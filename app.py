# app.py — ADRC RAG Bot (Semantic + ReRank) — English-only comments

import os, json, re
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()  # ensure .env is loaded before importing settings

from settings import (
    DATA_DIR, INDEX_DIR, EMBED_MODEL,
    INTERNAL_OPENAI_API_KEY, INTERNAL_OPENAI_API_BASE, INTERNAL_OPENAI_MODEL,
    EXTERNAL_OPENAI_API_KEY, EXTERNAL_OPENAI_API_BASE, EXTERNAL_OPENAI_MODEL,
    HOST, PORT, CORS_ORIGINS, BASE_URL_STATIC_HTML
)

# Embeddings: sentence-transformers (multilingual if you set EMBED_MODEL accordingly)
from sentence_transformers import SentenceTransformer
import numpy as np
import httpx

from index_htmls import build_corpus
import re

# ------------------------ link sanitizers ------------------------
_EXAMPLE_HOST_RE = re.compile(r"https?://example\.com", re.IGNORECASE)

def sanitize_context_links(txt: str) -> str:
    if not txt:
        return ""
    return _EXAMPLE_HOST_RE.sub("", txt)

def sanitize_answer_links(md: str) -> str:
    if not md:
        return md
    md = _EXAMPLE_HOST_RE.sub("", md)  # strip example.com
    md = md.replace("http:///html", "/html").replace("https:///html", "/html")
    md = re.sub(r"\((?:html/)", r"(/html/", md)
    return md


def strip_bare_html_paths(md: str) -> str:
    """
    Remove any bare /html/... paths that appear inline in the answer text,
    especially patterns like "(Source: /html/xxx#anchor)" or "مصدر] /html/...".
    Preserve proper markdown links [Title](/html/...).
    """
    if not md:
        return md
    # Remove patterns like "(Source: /html/...)" or "(المصدر: /html/...)"
    md = re.sub(r"\((?:Source|المصدر)\s*:\s*/html/[^)]+\)", "", md, flags=re.IGNORECASE)
    # Remove fragments like "مصدر] /html/..", "Source] /html/.."
    md = re.sub(r"(?:Source|مصدر)\]\s*/html/[^\s)]+", "", md, flags=re.IGNORECASE)
    # Remove standalone bare paths that are not part of markdown links
    # Negative lookbehind ensures we don't strip markdown ")(" part
    md = re.sub(r"(?<!\])\s*\(?/html/[^\s)]+\)?", "", md)
    # Collapse redundant spaces created by removals
    md = re.sub(r"\s{2,}", " ", md).strip()
    return md
# ------------------------ app & static mounts ------------------------
app = FastAPI(title="RAG Bot (Semi-Offline, No-FAISS)", version="1.1.0")

APP_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = APP_DIR / "public"
if PUBLIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")
app.mount("/html",   StaticFiles(directory=DATA_DIR, html=True),             name="html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Include the external agent router (cloud LLM)
#
# This router provides endpoints for uploading files to the provider and for
# streaming answers via Server‑Sent Events (SSE). To enable these endpoints,
# ensure that ``server/agent.py`` is present and importable. If the import
# fails (e.g. missing provider configuration), the agent endpoints will not
# be available but the rest of the application will still function.
try:
    from .agent import router as agent_router  # type: ignore
    app.include_router(agent_router)
except Exception as _e:
    # It's okay if the agent module cannot be imported; log for debugging
    print("[agent] router not loaded:", _e)

# No-cache for HTML (index + all /html/*.html)
from starlette.responses import FileResponse

@app.get("/html/index.html")
def no_cache_index():
    return FileResponse(
        os.path.join(DATA_DIR, "index.html"),
        headers={"Cache-Control": "no-store, max-age=0"}
    )

@app.middleware("http")
async def add_nocache_headers(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/html/") and getattr(resp, "media_type", "") == "text/html":
        resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

# ------------------------ globals ------------------------
embed_model = None
vectors: np.ndarray | None = None
metadata: List[Dict[str, Any]] = []

# ------------------------ helpers: embeddings/index ------------------------
def load_embedder():
    global embed_model
    if embed_model is None:
        # Tip: choose a multilingual model in EMBED_MODEL, e.g.:
        # "intfloat/multilingual-e5-base" or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embed_model = SentenceTransformer(EMBED_MODEL)
    return embed_model

def ensure_index_dir():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

def _paths():
    return {
        "vectors_npy": str(Path(INDEX_DIR) / "vectors.npy"),
        "meta_json":   str(Path(INDEX_DIR) / "metadata.json"),
    }

def save_index(X: np.ndarray, meta: List[Dict[str, Any]]):
    ensure_index_dir()
    p = _paths()
    np.save(p["vectors_npy"], X)
    Path(p["meta_json"]).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def load_index() -> bool:
    global vectors, metadata
    p = _paths()
    if Path(p["vectors_npy"]).exists() and Path(p["meta_json"]).exists():
        vectors = np.load(p["vectors_npy"])
        metadata = json.loads(Path(p["meta_json"]).read_text(encoding="utf-8"))
        return True
    return False

def encode(texts: List[str]) -> np.ndarray:
    model = load_embedder()
    emb = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return emb.astype("float32")

def build_index() -> Dict[str, Any]:
    # Build corpus from HTMLs. Each item includes: text, doc, section_path, anchor, etc.
    corpus = build_corpus(DATA_DIR)
    if not corpus:
        return {"ok": False, "error": "No HTML files found in DATA_DIR."}
    texts = [c["text"] for c in corpus]
    X = encode(texts)  # normalized embeddings
    save_index(X, corpus)
    return {"ok": True, "size": len(corpus)}


# ------------------------ OpenAI clients (HTTPX) ------------------------
def _openai_request(path: str, payload: Dict[str, Any], timeout: int = 60, *, api_key: str, api_base: str) -> Dict[str, Any]:
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    url = api_base.rstrip("/") + path
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

def call_openai_chat_external(messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 60) -> str:
    payload = {"model": INTERNAL_OPENAI_MODEL, "messages": messages, "temperature": temperature}
    data = _openai_request("/chat/completions", payload, timeout=timeout, api_key=INTERNAL_OPENAI_API_KEY, api_base=INTERNAL_OPENAI_API_BASE)
    return data["choices"][0]["message"]["content"]

def call_openai_chat_external(messages: List[Dict[str, str]], temperature: float = 0.3, timeout: int = 60) -> str:
    payload = {"model": EXTERNAL_OPENAI_MODEL, "messages": messages, "temperature": temperature}
    data = _openai_request("/chat/completions", payload, timeout=timeout, api_key=EXTERNAL_OPENAI_API_KEY, api_base=EXTERNAL_OPENAI_API_BASE)
    return data["choices"][0]["message"]["content"]

# ------------------------ semantic retrieval utils ------------------------

def topk_cosine(q_vec: np.ndarray, X: np.ndarray, k: int) -> List[int]:
    sims = X @ q_vec
    k = min(k, sims.size)
    if k <= 0:
        return []
    idx_part = np.argpartition(-sims, k - 1)[:k]
    idx_sorted = idx_part[np.argsort(-sims[idx_part])]
    return idx_sorted.tolist()

def build_citation_link(item: Dict[str, Any]) -> str:
    """
    Build a safe URL to the cited HTML page.

    Historically, the `doc` field in the metadata might only contain the filename
    (e.g. ``credit_policy.html``) instead of a path relative to ``DATA_DIR``
    (e.g. ``policies/credit_policy.html``). If the file is not found at the root
    of ``DATA_DIR``, this function will attempt to locate it in the ``policies``
    or ``quick_guides`` subdirectories and prefix the path accordingly. This
    ensures that citations always link to an existing file under ``/html``.

    Args:
        item: A dictionary containing metadata for a retrieved passage. Keys
          include ``doc`` (relative filename) and optional ``anchor``.

    Returns:
        A path-only URL (starting with ``/html``) pointing to the correct
        location of the HTML file, including the anchor if present.
    """
    base = (BASE_URL_STATIC_HTML or "/html").strip()
    # Build the base URL prefix. If ``base`` starts with a protocol (http/https)
    # we preserve it, otherwise we ensure it starts with a leading slash.
    prefix = base.rstrip("/") if base.startswith(("http://", "https://")) else (base if base.startswith("/") else "/" + base)

    # Extract the document name; strip leading slashes for safety.
    doc = (item.get("doc") or "").lstrip("/")

    # If the doc already contains a subdirectory (e.g. policies/filename.html), do not
    # modify it. Otherwise, attempt to locate it in known subdirectories. We check
    # ``policies/`` first, then ``quick_guides/``. If neither exists, leave ``doc`` as-is.
    if doc and "/" not in doc:
        from pathlib import Path
        # ``DATA_DIR`` is imported from settings; ensure it's a Path
        # Try policies folder
        policies_path = DATA_DIR / "policies" / doc
        if policies_path.exists():
            doc = f"policies/{doc}"
        else:
            quick_path = DATA_DIR / "quick_guides" / doc
            if quick_path.exists():
                doc = f"quick_guides/{doc}"

    href = f"{prefix}/{doc}" if doc else prefix
    anchor = (item.get("anchor") or "").strip()
    if anchor:
        href += f"#{anchor}"
    return href

# ------------------------ semantic query understanding ------------------------
def understand_query_semantically(q: str) -> Dict[str, Any]:
    """
    Returns a small JSON with intents/entities and 2-3 semantic paraphrases (same language).
    No translation; preserve language of the input.
    """
    sys = "You convert a user policy question into semantic search intents. Return compact JSON only."
    usr = f"""Question: {q}
Return JSON with fields:
- "intents": 1-2 short intents
- "entities": key entities/sections if any
- "queries": 2-3 short semantic paraphrases in the SAME language of the question (no translation)"""
    try:
        msg = [
            {"role": "system", "content": sys},
            {"role": "user", "content": usr},
        ]
        content = call_openai_chat_external(msg, temperature=0.2)
        # tolerant JSON parsing
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            return json.loads(content[brace_start:brace_end+1])
    except Exception:
        pass
    return {"intents": [], "entities": [], "queries": []}

def llm_rerank(question: str, candidates: List[Dict[str, Any]]) -> List[int]:
    """
    Ask LLM to rank passages by semantic relevance.
    Returns 0-based indices in best-to-worst order.
    """
    # Build passages preview (trim aggressively to keep prompt small)
    blocks = []
    for i, c in enumerate(candidates, start=1):
        title = c.get("section_path", "") or c.get("doc", "")
        text = sanitize_context_links(c.get("text", ""))[:900]
        blocks.append(f"[{i}] Title: {title}\n{text}")
    passages_blob = "\n\n".join(blocks)

    sys = ("You are a multilingual re-ranker for company policy passages. "
           "Rank by semantic relevance to the question. Return JSON list of indices (1-based).")
    usr = f"Question:\n{question}\n\nPassages:\n{passages_blob}\n\nReturn JSON like: [3,1,2]"
    try:
        content = call_openai_chat_external([{"role":"system","content":sys},{"role":"user","content":usr}], temperature=0.0)
        # tolerant parsing
        bracket_start = content.find("[")
        bracket_end = content.rfind("]")
        order = []
        if bracket_start != -1 and bracket_end != -1:
            arr = json.loads(content[bracket_start: bracket_end+1])
            if isinstance(arr, list):
                for v in arr:
                    try:
                        iv = int(v) - 1
                        if 0 <= iv < len(candidates):
                            order.append(iv)
                    except Exception:
                        continue
        # fallback: if empty, keep original order
        return order if order else list(range(len(candidates)))
    except Exception:
        return list(range(len(candidates)))

# ------------------------ API models ------------------------
class AskIn(BaseModel):
    question: str
    top_k: int = 8  # final top-K after rerank

# Additional model for v2 API (dual-source chat)
class Ask2In(BaseModel):
    query: str
    conversation_id: str | None = None
    lang: str | None = None
    history: list | None = None
    # Optional intent hint (e.g. "summary") from the frontend to adjust response mode
    intent: str | None = None

# ------------------------ routes ------------------------

@app.get("/favicon.ico")
def favicon():
    from starlette.responses import FileResponse
    p = APP_DIR.parent / "public" / "favicon.ico"
    return FileResponse(str(p))

@app.get("/health")
def health():
    ok = load_index()
    return {"ok": ok, "docs_indexed": (len(metadata) if ok else 0)}

@app.post("/ingest")
def ingest():
    return build_index()

@app.post("/ask")
def ask(inp: AskIn = Body(...)):
    if not load_index():
        return {"ok": False, "error": "Index not built yet. Call /ingest first."}

    # 1) Query understanding (no translation; semantic paraphrases in same language)
    qinfo = understand_query_semantically(inp.question)
    paraphrases: List[str] = []
    qs = [inp.question]
    if isinstance(qinfo.get("queries"), list):
        paraphrases = [q for q in qinfo["queries"] if isinstance(q, str) and q.strip()]
    # Keep up to 2 extra semantic queries (short)
    qs += paraphrases[:2]

    # 2) Dense semantic retrieval for each query; union of hits
    #    For efficiency we embed all queries at once then search for each vector.
    q_embs = encode(qs)
    pool_indices: List[int] = []
    # per-query top size slightly higher than final K
    per_k = max(inp.top_k + 4, 12)
    for i in range(q_embs.shape[0]):
        idxs = topk_cosine(q_embs[i], vectors, per_k)
        pool_indices.extend(idxs)
    # deduplicate while preserving order
    seen = set()
    uniq = []
    for ix in pool_indices:
        if ix not in seen:
            seen.add(ix)
            uniq.append(ix)

    # build candidate passages
    candidates = [metadata[i] for i in uniq]
    # Filter out quick-guide or other non-policy docs if they somehow exist in the index.
    # We rely on filename patterns; any doc containing "quick" (case-insensitive) is excluded.
    filtered_candidates = []
    for c in candidates:
        docname = (c.get("doc") or "").lower()
        # exclude quick guides and general quick guide pages
        if "quick" in docname:
            continue
        filtered_candidates.append(c)
    candidates = filtered_candidates

    # 3) LLM Re-ranking by semantic relevance
    order = llm_rerank(inp.question, candidates)
    ranked = [candidates[i] for i in order][: int(inp.top_k)]

    # 4) Build answer prompt with only the ranked passages
    system = (
        "You are Tito — the ADRC Policy Bot. "
        "You answer about ADRC policies/SOPs. Be concise. "
        "Always cite sources with markdown links. "
        "Never use absolute domains in links; only use path links that start with /html/filename.html#anchor."
    )

    context_lines = []
    citations = []
    for h in ranked:
        link = build_citation_link(h)
        safe_text = sanitize_context_links(h.get("text", ""))
        context_lines.append(
            "[{doc} :: {section}]\n{text}\n(Source: {link})".format(
                doc=h.get("doc", ""),
                section=h.get("section_path", ""),
                text=safe_text,
                link=link,
            )
        )
        citations.append({"title": h.get("section_path", ""), "doc": h.get("doc", ""), "href": link})

    context_text = sanitize_context_links("\n\n---\n\n".join(context_lines))
    user_msg = (
        "Question: {q}\n\nUse only the provided context to answer. "
        "If not sure, say you are not sure. Cite sources inline with [title](link)."
    ).format(q=inp.question)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",  "content": "Context:\n" + context_text + "\n\n" + user_msg},
    ]

    try:
        answer = call_openai_chat_external(messages, temperature=0.1)
        answer = sanitize_answer_links(answer)
        return {"ok": True, "answer_markdown": answer, "citations": citations, "passages": ranked, "queries_used": qs}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ------------------------ Dual-Source API ------------------------


def rewrite_query_with_history(history: List[Dict[str,str]], current: str, lang: str) -> str:
    if not history:
        return current
    # Build messages for the rewriter
    sys_ar = "أنت مساعد لإعادة صياغة الأسئلة. حوّل سؤال المتابعة إلى سؤال مستقل ومكتمل المعنى، باستخدام المعلومات الضمنية من الحوار السابق. أعد السؤال فقط."
    sys_en = "You are a query rewriter. Turn the follow‑up question into a standalone, fully‑specified question using prior chat context. Return ONLY the rewritten question."
    sys = sys_ar if (lang or '').lower().startswith('ar') else sys_en
    msgs = [{"role":"system","content":sys}]
    # Include limited recent history (last 6 exchanges) to stay within budget
    recent = []
    for m in history[-12:]:
        role = m.get("role")
        content = m.get("content","")
        if role in ("user","bot") and content:
            msgs.append({"role":"user" if role=="user" else "assistant", "content": content})
    msgs.append({"role":"user","content": current})
    try:
        rewritten = call_openai_chat_external(msgs, temperature=0.0)
        # Keep it single line
        return rewritten.strip().replace("\n", " ")
    except Exception:
        return current
@app.post("/api/ask_local")
async def api_ask_local(inp: Ask2In = Body(...)):
    """
    Local RAG endpoint for Tito v2. Accepts query, returns answer and enriched HTML sources.
    - Builds a combined query from history (last user msg + current query).
    - Delegates retrieval/answering to /ask().
    - Returns sources with: title, source_url, snippet, heading_id.
    """
    try:
        # Preserve context on follow-ups
        combined_query = rewrite_query_with_history(inp.history or [], inp.query, inp.lang or '')

        # Prefer the most recent user message explicitly if present
        if inp.history:
            for m in reversed(inp.history):
                if isinstance(m, dict) and m.get("role") == "user":
                    prev_q = m.get("content")
                    if isinstance(prev_q, str) and prev_q.strip():
                        combined_query = prev_q.strip() + " " + combined_query
                    break

        res = ask(AskIn(question=combined_query, top_k=8))  # type: ignore[call-arg]
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": []}

    # Validate downstream response
    if not isinstance(res, dict) or not res.get("ok"):
        msg = res.get("error") if isinstance(res, dict) else "Error"
        return {"answer": msg, "sources": []}

    # Assemble enriched sources
    answer_md = res.get("answer_markdown") or ""
    citations = res.get("citations") or []
    passages = res.get("passages") or []

    sources = []
    for i, c in enumerate(citations):
        url = c.get("href") or c.get("url") or ""
        title = c.get("title") or c.get("doc") or url

        # Pull precise snippet from the matching passage
        snippet = ""
        if i < len(passages):
            snippet = (passages[i].get("text") or "").strip()
            if len(snippet) > 240:
                snippet = snippet[:240].rstrip() + "…"

        # Extract anchor id & strip query params; keep path + hash only
        heading_id = ""
        if isinstance(url, str):
            if "#" in url:
                heading_id = url.split("#", 1)[1]
            if "?" in url:
                url = url.split("?", 1)[0]

        sources.append({
            "title": title,
            "source_url": url,
            "snippet": snippet,
            "heading_id": heading_id
        })

    # Optional summarization mode
    if (inp.intent or "").lower() == "summary":
        try:
            if (inp.lang or "").lower().startswith("ar"):
                sys = "أنت مساعد متخصص في التلخيص. قدم ملخصًا موجزًا ودقيقًا باللغة العربية."
                user_prompt = f"الجواب: {answer_md}\n\nلخص هذا الجواب في نقطتين أو ثلاث أسطر كحد أقصى."
            else:
                sys = "You are a helpful summarization assistant."
                user_prompt = f"Answer: {answer_md}\n\nSummarize into two bullet points or three short sentences, in the same language."
            summary = call_openai_chat_internal([
                {"role": "system", "content": sys},
                {"role": "user",  "content": user_prompt},
            ], temperature=0.3)
            answer_md = (summary or "").strip()
        except Exception:
            pass

    answer_md = strip_bare_html_paths(answer_md)
    return {"answer": answer_md, "sources": sources}


@app.post("/api/rag")
async def api_rag(inp: Ask2In = Body(...)):
    """
    Compatibility alias for frontend expecting POST /api/rag.
    Delegates to local RAG endpoint (/api/ask_local).
    """
    return await api_ask_local(inp)
@app.post("/api/ask_external")
async def api_ask_external(inp: Ask2In = Body(...)):
    """
    External agent endpoint for Tito v2. Uses OpenAI to answer general questions.
    Includes conversation history to allow follow-up questions. Supports summarisation via intent.
    Attempts to extract markdown links from the response as citations.
    """
    # conversation history passed from frontend (list of dicts with role and content)
    hist = []
    # Build OpenAI messages from conversation history, if provided
    if isinstance(inp.history, list):
        for m in inp.history:
            role = m.get("role")
            content = m.get("content", "")
            if not content:
                continue
            # Convert roles: user's messages stay as "user", bot messages as "assistant"
            if role == "user":
                hist.append({"role": "user", "content": content})
            elif role == "bot":
                hist.append({"role": "assistant", "content": content})

    # Determine language and whether summarisation is requested
    lang = (inp.lang or "").lower()
    wants_summary = (inp.intent or "").lower() == "summary"

    # Combine last user question with follow-up query for better context (if any)
    combined_query = inp.query
    if hist:
        # Find last user content in history
        for m in reversed(hist):
            if m["role"] == "user":
                combined_query = m["content"] + "\n\n" + inp.query
                break

    # General assistant prompt
    system = (
        "You are a multilingual general knowledge assistant. "
        "Answer the user's question thoroughly in the same language if possible. "
        "If you know relevant credible sources on the public web that support your answer, "
        "include them at the end of your response as markdown links of the form [Title](https://example.com). "
        "If no sources are known, you may omit them."
    )

    messages = [ {"role": "system", "content": system} ]
    messages.extend(hist[-12:])  # include up to 12 previous interactions to keep prompt small
    messages.append({"role": "user", "content": combined_query})

    try:
        content = call_openai_chat_external(messages, temperature=0.3)
        import re as _re
        # Extract markdown links from the response
        cite_pairs = _re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", content)
        sources = [{"title": t, "url": u} for (t, u) in cite_pairs]
        # Remove links from the answer text
        answer_no_links = _re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", content).strip()
        # Summarise if requested
        if wants_summary:
            try:
                if lang.startswith("ar"):
                    sys_msg = "أنت مساعد متخصص في التلخيص. قدم ملخصًا موجزًا باللغة العربية."
                    usr_prompt = f"الجواب: {answer_no_links}\n\nلخص هذا الجواب في نقطتين أو ثلاث أسطر كحد أقصى."
                else:
                    sys_msg = "You are a helpful summarization assistant."
                    usr_prompt = f"Answer: {answer_no_links}\n\nSummarize the above answer into at most two bullet points or three short sentences, in the same language."
                summary = call_openai_chat_external([
                    {"role": "system", "content": sys_msg},
                    {"role": "user",  "content": usr_prompt},
                ], temperature=0.3)
                if summary:
                    answer_no_links = summary.strip()
            except Exception:
                pass
        return {"answer": answer_no_links, "sources": sources}
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": []}

# (optional) auto-ingest if missing
@app.on_event("startup")
def _auto_ingest():
    try:
        if not load_index() or not metadata:
            build_index()
    except Exception as e:
        print("[auto-ingest] skipped:", e)

# ---- SSE Heartbeat Endpoint (stability) --------------------------------------
@app.get("/sse/heartbeat")
async def sse_heartbeat():
    """Lightweight SSE stream for liveness and latency monitoring.
    Sends a ping every 15 seconds. Clients should auto-reconnect with backoff.
    """
    from fastapi.responses import StreamingResponse
    import asyncio
    async def event_gen():
        try:
            yield "event: ping\ndata: {\"ts\": %d}\n\n" % int(__import__('time').time()*1000)
            while True:
                await asyncio.sleep(15)
                yield "event: ping\ndata: {\"ts\": %d}\n\n" % int(__import__('time').time()*1000)
        except asyncio.CancelledError:
            return
    headers = {
        "Cache-Control": "no-store",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

# ---- SSE Ask Endpoint (streaming answers) -----------------------------------
@app.get("/sse/ask")
async def sse_ask(
    q: str,
    scope: str = "local",
    topk: int = 6,
    conversation_id: str = "",
    lang: str = "",
    intent: str = "",
    history: str = "",
):
    """
    Stream answers token-by-token via Server-Sent Events. This endpoint wraps
    the existing ask/local functionality to provide incremental responses to
    the frontend. Only the internal (policy) source is streamed; any external
    agent should be queried separately via the regular POST endpoint.

    Parameters:
      q: The current user question. Will be combined with history on the server side.
      scope: Currently unused; reserved for future dual streaming.  Default "local".
      topk: Top‑K passages for retrieval.  Default 6.
      conversation_id: An opaque identifier to correlate with client conversations.
      lang: Detected language code (e.g. "ar" or "en"). Used for summarisation.
      intent: Optional hint such as "summary".
      history: JSON‑encoded list of previous messages (user and bot) with keys
        "role" and "content". Sent from the frontend to support follow‑up questions.

    The SSE stream sends a sequence of events:
      event: start\n
      data: {"sources": [...]}\n\n
      data: {"token": "..."}\n\n  (repeated for each chunk)

      event: end\n
      data: {}\n\n
    Clients should append each received token to the current answer buffer and
    update the UI. When the "sources" object is delivered, the client should
    display citations accordingly.
    """
    from fastapi.responses import StreamingResponse
    import asyncio
    # Parse history JSON if provided
    hist_list: List[Dict[str, str]] = []
    if history:
        try:
            hist_list = json.loads(history)
            if not isinstance(hist_list, list):
                hist_list = []
        except Exception:
            hist_list = []

    async def generate_stream():
        try:
            # Immediately send start event
            yield "event: start\ndata: {}\n\n"
            # Build the input model for the local API; reuse existing ask_local implementation.
            from pydantic import ValidationError
            try:
                inp = Ask2In(query=q, conversation_id=conversation_id or None, lang=lang or None, history=hist_list or None, intent=intent or None)
            except ValidationError as ve:
                err_msg = f"Invalid input: {ve}"
                for ch in err_msg:
                    yield f"data: {{\"token\": {json.dumps(ch)} }}\n\n"
                yield "event: end\ndata: {}\n\n"
                return
            # Invoke the local ask endpoint directly
            try:
                res = await api_ask_local(inp)  # type: ignore[arg-type]
            except Exception as e:
                err_msg = f"Error: {e}"
                for ch in err_msg:
                    yield f"data: {{\"token\": {json.dumps(ch)} }}\n\n"
                yield "event: end\ndata: {}\n\n"
                return
            answer = ""
            sources = []
            if isinstance(res, dict):
                answer = res.get("answer") or res.get("answer_markdown") or ""
                sources = res.get("sources") or res.get("citations") or []
            # send sources event
            try:
                yield f"event: sources\ndata: {json.dumps({'sources': sources})}\n\n"
            except Exception:
                pass
            # Stream answer
            if answer:
                text = str(answer)
                chunk_size = 30
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    yield f"data: {{\"token\": {json.dumps(chunk)} }}\n\n"
            # end event
            yield "event: end\ndata: {}\n\n"
        except asyncio.CancelledError:
            return

    headers = {
        "Cache-Control": "no-store",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(generate_stream(), media_type="text/event-stream", headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST or "127.0.0.1", port=PORT, reload=False)


@app.get("/api/health")
async def api_health():
    return {"ok": True, "service": "tito", "version": "v7"}
