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

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from urllib.parse import urlparse

import httpx
import trafilatura
from ddgs import DDGS
from htmldate import find_date

from models import Source

STORE_CHARS = 16000   # keep a lot per source; budgeting happens at use-time
FETCH_WORKERS = 5     # pages are fetched concurrently
FETCH_TIMEOUT = 20    # seconds per page
SEARCH_RETRY_DELAY = 3  # seconds before the one retry (DDG rate-limits bursts)

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


def _fetch_page(url: str) -> tuple[str, str]:
    """Fetch a page and return (main article text, publish date), or ('', '')
    on any failure. The date comes from page metadata (meta tags, JSON-LD, URL)
    and is '' when none is found. Wikipedia articles are living documents whose
    "original" date is the day the article was created, so for them we take the
    last-edited date instead; elsewhere "latest date on the page" is unreliable
    (it picks up sidebar links), so we keep the publish date."""
    if not url:
        return "", ""
    try:
        r = httpx.get(url, headers=_HEADERS, follow_redirects=True,
                      timeout=FETCH_TIMEOUT)
        r.raise_for_status()
    except Exception:
        return "", ""
    text = trafilatura.extract(r.text, include_comments=False,
                               favor_recall=True) or ""
    try:
        wiki = urlparse(url).netloc.endswith("wikipedia.org")
        published = find_date(r.text, url=url, original_date=not wiki) or ""
    except Exception:
        published = ""
    return text, published


class DuckDuckGoSearch:
    """Keyless search via DuckDuckGo + trafilatura page-text extraction."""

    def __init__(self, store_chars: int = STORE_CHARS):
        self.store_chars = store_chars

    def _hits(self, query: str, k: int) -> list[dict]:
        """DDG raises on both "no results" and rate limiting; retry once, then
        give up with [] so one failed search doesn't sink the whole run."""
        for attempt in range(2):
            try:
                with DDGS() as ddgs:
                    return ddgs.text(query, max_results=k) or []  # {title, href, body}
            except Exception:
                if attempt == 0:
                    time.sleep(SEARCH_RETRY_DELAY)
        return []

    def search(self, query: str, k: int = 5) -> list[Source]:
        hits = self._hits(query, k)
        if not hits:
            return []

        urls = [h.get("href", "") for h in hits]
        with ThreadPoolExecutor(max_workers=min(FETCH_WORKERS, len(urls))) as pool:
            pages = list(pool.map(_fetch_page, urls))

        out: list[Source] = []
        for h, (text, published) in zip(hits, pages):
            url = h.get("href", "")
            if not url:
                continue
            body = text or h.get("body", "")  # fall back to the search snippet
            out.append(Source(
                id="",  # assigned globally by the orchestrator
                url=url,
                title=h.get("title", "") or url,
                content=body[: self.store_chars],
                published=published,
            ))
        return out
