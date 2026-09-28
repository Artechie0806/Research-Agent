"""FastAPI server: streams the agent's pipeline to the browser over SSE.

The agent runs sync (blocking LLM calls + a verify thread pool), so it runs in a
worker thread; its on_event callback pushes dicts onto an asyncio queue that the
SSE endpoint drains. A fresh agent per request keeps source/claim state isolated.

    uvicorn server:app --reload      # from backend/
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse

from agent import ResearchAgent
from context import Budget
from llm import missing_config
from search import DuckDuckGoSearch
from trending import get_trending_claims, is_live

load_dotenv()  # read .env for LLM settings (see llm.py; search is keyless)

def _prewarm_trending() -> None:
    try:
        get_trending_claims(n=6)  # populate the cache so the landing loads fast
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the trending-claims cache in the background at startup; the first
    # fetch does a live search + LLM call, so this keeps the landing page snappy.
    threading.Thread(target=_prewarm_trending, daemon=True).start()
    yield


app = FastAPI(title="Groundwork — Research Verifier", lifespan=lifespan)
# index.html sits next to this file (flat layout).
FRONTEND = Path(__file__).resolve().parent


@app.get("/")
def index() -> FileResponse:
    # no-cache so an updated index.html is always picked up on refresh
    return FileResponse(FRONTEND / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/trending")
def trending() -> dict:
    """Latest trending/controversial claims for the landing page's floating boxes.

    Best-effort and cached. `live` is False while a cold cache is still being
    filled in the background (the claims served meanwhile are fallbacks), so the
    UI knows to poll again for the real news."""
    try:
        return {"claims": get_trending_claims(n=6), "live": is_live()}
    except Exception:
        return {"claims": [], "live": False}


@app.get("/api/research")
async def research(request: Request, query: str, max_rounds: int = 2,
                   keep_partial: bool = False, max_context: int = 16000):
    def sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    missing = missing_config()
    if missing:
        async def err():
            yield sse({"type": "error",
                       "message": f"Set these env vars and restart: "
                                  f"{', '.join(missing)}."})
        return StreamingResponse(err(), media_type="text/event-stream")

    max_rounds = max(1, min(max_rounds, 4))
    max_context = max(4000, min(max_context, 200000))

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def work() -> None:
        try:
            agent = ResearchAgent(search=DuckDuckGoSearch(),
                                  budget=Budget(max_context=max_context))
            agent.run(query, max_rounds=max_rounds, keep_partial=keep_partial,
                      on_event=emit)
        except Exception as exc:  # surface failures to the UI, don't hang it
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            emit({"type": "__end__"})

    loop.run_in_executor(None, work)

    async def stream():
        while True:
            event = await queue.get()
            if event.get("type") == "__end__":
                break
            yield sse(event)
            if await request.is_disconnected():
                break

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
