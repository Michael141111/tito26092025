"""
FastAPI router for a cloud agent that streams responses via SSE.

This module defines endpoints to upload and delete files to a remote provider
and to stream answers for questions using an external LLM with optional file
attachments. It currently implements support for the OpenAI API using the
`responses` endpoint with retrieval attachments. To enable support for other
providers, set the environment variable `LLM_PROVIDER` accordingly and add
matching helper functions below.

The SSE endpoint yields a ``start`` event, the raw streamed JSON lines from
the provider (each line is passed through unchanged), and an ``end`` event
when the stream completes. The frontend is responsible for parsing the
``token`` fields from the streamed JSON objects.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
import os
import json
from typing import List, Optional
import httpx

# Define the router with a prefix so it mounts under /agent in the main app
router = APIRouter(prefix="/agent", tags=["agent"])

# Determine which provider to use for LLM calls. Currently only "openai"
# is supported. You can set ``LLM_PROVIDER=openai`` in your environment.
PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()


# ---------------------------------------------------------------------------
# Provider-specific helpers
#
# These functions implement file upload, deletion and chat streaming for the
# configured provider. They should raise an HTTPException on failure. If you
# wish to add support for additional providers (e.g. Azure OpenAI), implement
# corresponding upload/delete/stream functions and extend the conditional
# blocks below.

async def _openai_upload(file: UploadFile) -> str:
    """Upload a file to the OpenAI API and return its file_id."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    url = "https://api.openai.com/v1/files"
    headers = {"Authorization": f"Bearer {api_key}"}
    # Multipart form: purpose and file content
    form = {
        "purpose": (None, "assistants"),
        "file": (file.filename, await file.read(), file.content_type or "application/octet-stream"),
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, headers=headers, files=form)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"OpenAI upload failed: {e}")
    data = resp.json()
    return data.get("id")


async def _openai_delete(file_id: str) -> bool:
    """Delete a previously uploaded file from the OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    url = f"https://api.openai.com/v1/files/{file_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.delete(url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"OpenAI delete failed: {e}")
    return True


async def _openai_stream_chat(question: str, file_ids: Optional[List[str]]) -> httpx.AsyncClient:
    """
    Stream a chat completion from the OpenAI Responses API.

    The question is passed as the ``input`` field. If ``file_ids`` are
    provided, they are attached via the ``attachments`` field to enable
    retrieval from uploaded files. The environment variable ``OPENAI_MODEL``
    may be used to set the model name; it defaults to ``gpt-4o-mini``.

    This generator yields each line of the response as-is. The caller is
    responsible for prefixing the lines with ``data:`` if desired.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": question,
        "stream": True,
    }
    if file_ids:
        attachments = [
            {"file_id": fid, "tools": [{"type": "retrieval"}]} for fid in file_ids
        ]
        payload["attachments"] = attachments
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=f"OpenAI stream failed: {e}")
            async for line in resp.aiter_lines():
                # The responses API returns lines beginning with 'data:' or empty
                if not line:
                    continue
                yield f"{line}\n"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.post("/files")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the configured provider and return its ID."""
    if PROVIDER == "openai":
        file_id = await _openai_upload(file)
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    return {"file_id": file_id, "filename": file.filename}


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a previously uploaded file from the configured provider."""
    if PROVIDER == "openai":
        await _openai_delete(file_id)
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    return {"ok": True}


@router.get("/sse")
async def ask_sse(
    q: str = Query(..., min_length=1),
    files: Optional[str] = Query(None, description="JSON list of file IDs to attach"),
):
    """
    Stream an answer from the external LLM using Server Sent Events.

    The ``q`` parameter holds the user question. The optional ``files``
    parameter should contain a JSON-encoded array of file IDs previously
    uploaded via POST /agent/files; these IDs will be attached to the request
    so the LLM can retrieve information from the files.
    """
    try:
        file_ids = json.loads(files) if files else None
        if file_ids is not None and not isinstance(file_ids, list):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid files parameter; must be JSON list")

    async def event_generator():
        # Indicate the stream has started so the frontend can hide typing indicator
        yield "event: start\n" + "data: {}\n\n"
        # Forward provider stream; note: provider lines may include their own 'data:' prefix
        if PROVIDER == "openai":
            async for chunk in _openai_stream_chat(q, file_ids):
                # Forward the chunk unmodified; do not prepend event name so 'message' is used
                yield chunk
        else:
            # Should never happen due to earlier check, but included for completeness
            yield "data: {\"error\": \"Unsupported provider\"}\n\n"
        # Signal completion
        yield "event: end\n" + "data: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")