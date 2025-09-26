
# English-only comments.
from bs4 import BeautifulSoup
import os, re, json, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

def slugify(text: str) -> str:
    text = re.sub(r"\s+", "-", text.strip().lower())
    text = re.sub(r"[^a-z0-9\-]", "", text)
    return text

def read_html_files(data_dir: str) -> List[Path]:
    p = Path(data_dir)
    return [f for f in p.glob("*.html") if f.is_file()]

def extract_blocks(html_path: Path) -> List[Dict[str, Any]]:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Ensure each heading has an id anchor
    changed = False
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        if not h.get("id"):
            anchor = slugify(h.get_text()[:80])
            if not anchor:
                anchor = hashlib.md5(h.get_text().encode("utf-8")).hexdigest()[:8]
            h["id"] = anchor
            changed = True
    # Persist anchors back to the file so links & :target work in the browser
    if changed:
        try:
            html_path.write_text(str(soup), encoding="utf-8")
        except Exception:
            pass

    # Extract text blocks by heading sections
    blocks = []
    doc_title = soup.title.get_text().strip() if soup.title else html_path.name
    headings = soup.find_all(re.compile(r"^h[1-6]$"))
    if not headings:
        # Fallback: whole body as one block
        body_text = soup.get_text(separator=" ", strip=True)
        blocks.append({
            "doc": html_path.name,
            "title": doc_title,
            "anchor": "",
            "section_path": doc_title,
            "text": body_text[:5000]
        })
        return blocks

    for i, h in enumerate(headings):
        start = h
        end = headings[i+1] if i+1 < len(headings) else None
        section_nodes = []
        node = start
        while node and node != end:
            section_nodes.append(node)
            node = node.find_next_sibling()
        section_soup = BeautifulSoup("", "html.parser")
        for n in section_nodes:
            section_soup.append(n)
        section_text = section_soup.get_text(separator=" ", strip=True)
        blocks.append({
            "doc": html_path.name,
            "title": doc_title,
            "anchor": h.get("id") or "",
            "section_path": h.get_text().strip(),
            "text": section_text[:8000],  # safety
        })

    return blocks

def chunk_text(text: str, max_len: int = 900, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = i + max_len
        chunk_words = words[i:j]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += (max_len - overlap)
    return chunks

def build_corpus(data_dir: str) -> List[Dict[str, Any]]:
    """
    Build a corpus of text chunks from HTML files. If a ``policies`` subdirectory
    exists inside the provided data directory, only HTML files inside that
    subdirectory are indexed. Otherwise all HTML files in ``data_dir`` are used.

    The intent is to allow separating long-form policy/SOP documents from
    supplemental quick guides. When a ``policies`` directory exists, the search
    index will be restricted to those documents alone, preventing accidental
    inclusion of quick guide pages or other non-authoritative sources.
    """
    corpus = []
    data_path = Path(data_dir)
    # Prefer a dedicated policies folder if present
    policies_dir = data_path / "policies"
    target_dir = policies_dir if policies_dir.exists() else data_path
    for f in read_html_files(str(target_dir)):
        # Derive relative path of HTML (e.g. "policies/credit_policy.html") for citation links
        try:
            rel_path = f.relative_to(data_path).as_posix()
        except Exception:
            rel_path = f.name
        blocks = extract_blocks(f)
        for b in blocks:
            parts = chunk_text(b["text"], max_len=300, overlap=60)  # shorter chunks → better pinpointing
            for idx, part in enumerate(parts):
                # Overwrite "doc" with the relative HTML path. This ensures that
                # citations link correctly after moving HTML files into subfolders
                item = {
                    "doc": rel_path,
                    "title": b["title"],
                    "anchor": b["anchor"],
                    "section_path": b["section_path"],
                    "chunk_id": idx,
                    "text": part
                }
                corpus.append(item)
    return corpus
