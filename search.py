"""Search + fetch, keyless. DuckDuckGo finds URLs; trafilatura extracts page text.

No API key required (replaces Tavily). `ddgs` returns result URLs + snippets;
each page is then fetched with a browser User-Agent and reduced to its main
article text with trafilatura. When a page bot-walls or has no extractable
article, we fall back to the search snippet so the Source still carries something.

Swap providers by implementing the same `.search(query, k) -> list[Source]`.
Store generously here; context.py trims at prompt-build time so the verifier can
window into the relevant passage rather than being stuck with the first N chars.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import httpx
import trafilatura
from ddgs import DDGS

from models import Source

STORE_CHARS = 16000   # keep a lot per source; budgeting happens at use-time
FETCH_WORKERS = 5     # pages are fetched concurrently
FETCH_TIMEOUT = 20    # seconds per page

# A real browser UA + Accept header — trafilatura's own fetcher gets bot-walled
# on some sites; httpx with these headers reliably returns the article HTML.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class SearchProvider(Protocol):
    def search(self, query: str, k: int = 5) -> list[Source]: ...


def _fetch_text(url: str) -> str:
    """Fetch a page and return its main article text, or '' on any failure."""
    if not url:
        return ""
    try:
        r = httpx.get(url, headers=_HEADERS, follow_redirects=True,
                      timeout=FETCH_TIMEOUT)
        r.raise_for_status()
    except Exception:
        return ""
    return trafilatura.extract(r.text, include_comments=False,
                               favor_recall=True) or ""


class DuckDuckGoSearch:
    """Keyless search via DuckDuckGo + trafilatura page-text extraction."""

    def __init__(self, store_chars: int = STORE_CHARS):
        self.store_chars = store_chars

    def search(self, query: str, k: int = 5) -> list[Source]:
        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=k)  # each: {title, href, body}
        if not hits:
            return []

        urls = [h.get("href", "") for h in hits]
        with ThreadPoolExecutor(max_workers=min(FETCH_WORKERS, len(urls))) as pool:
            texts = list(pool.map(_fetch_text, urls))

        out: list[Source] = []
        for h, text in zip(hits, texts):
            url = h.get("href", "")
            if not url:
                continue
            body = text or h.get("body", "")  # fall back to the search snippet
            out.append(Source(
                id="",  # assigned globally by the orchestrator
                url=url,
                title=h.get("title", "") or url,
                content=body[: self.store_chars],
            ))
        return out
