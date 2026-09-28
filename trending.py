"""Latest trending / controversial checkable claims for the landing page.

Searches DuckDuckGo *news* (keyless) for the last few days of contested stories,
keeps only the freshest headlines, then asks the LLM to distill them
into short, self-contained claims a user could actually verify. Results are
cached in-process so page loads don't re-run search + LLM every time.

Freshness is the whole point here, so the degradation path stays newsy: if the
LLM call fails we build claims straight from the headline titles, and only fall
back to static evergreen claims when search itself returns nothing (offline).
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from ddgs import DDGS

from llm import LLMClient, make_client

# Angles on "currently contested", spread across domains so the six boxes don't
# all end up being the same story.
_SEEDS = [
    "controversy sparks backlash",
    "disputed claim fact check",
    "policy decision criticized debate",
    "study findings disputed scientists",
]

_SYSTEM = "You turn the latest news headlines into short, checkable factual claims."

_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
}

_TTL = 900  # seconds — news goes stale fast, so re-fetch every 15 minutes
_MAX_AGE_HOURS = 96  # drop anything older than ~4 days; "trending" has a shelf life
_MAX_CHARS = 120  # a claim has to fit in a ~180px floating box

_cache: dict = {"at": 0.0, "claims": [], "live": False}
_refresh_lock = threading.Lock()

# Last resort only — served when search itself is unavailable (offline, rate
# limited) so the landing page always has boxes. Not news, but still checkable.
_FALLBACK = [
    "The Great Wall of China is visible from space with the naked eye.",
    "Humans use only 10 percent of their brains.",
    "Goldfish have a memory span of just three seconds.",
    "Lightning never strikes the same place twice.",
    "Mount Everest is the tallest mountain on Earth measured from base to peak.",
    "Napoleon Bonaparte was unusually short for his time.",
]

_REL = re.compile(r"(\d+)\s*(minute|min|hour|hr|day|week|month)s?\s+ago", re.I)
_ISO = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?"
)
_UNIT_HOURS = {"minute": 1 / 60, "min": 1 / 60, "hour": 1, "hr": 1,
               "day": 24, "week": 168, "month": 720}


def _age_hours(raw: str) -> float | None:
    """Age of a news item in hours, or None if its date can't be read.

    ddgs mixes formats across engines: ISO timestamps ("2026-07-09T14:24:45+00:00")
    and relative strings, sometimes glued to a label ("Opinion1 day ago").
    """
    s = (raw or "").strip()
    if not s:
        return None
    m = _REL.search(s)
    if m:
        return int(m.group(1)) * _UNIT_HOURS[m.group(2).lower()]
    m = _ISO.search(s)
    if not m:
        return None
    try:
        dt = datetime.fromisoformat(m.group(0).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600


def _shorten(text: str, limit: int = _MAX_CHARS) -> str:
    """Trim to `limit` chars on a word boundary rather than mid-word."""
    t = " ".join(text.split())
    if len(t) <= limit:
        return t
    cut = t[:limit].rsplit(" ", 1)[0].rstrip(" ,;:—-")
    return (cut or t[:limit]) + "…"


def _ago(hours: float | None) -> str:
    if hours is None:
        return "recent"
    if hours < 1:
        return "just now"
    if hours < 24:
        return f"{int(hours)}h ago"
    return f"{int(hours // 24)}d ago"


def _one_seed(seed: str, per_seed: int) -> list[dict]:
    """News hits for one seed. timelimit is a hint the engines honour loosely,
    so the real age filter happens in _headlines()."""
    out: list[dict] = []
    try:
        with DDGS() as d:
            hits = d.news(seed, timelimit="w", max_results=per_seed,
                          region="wt-wt")
    except Exception:
        return out
    for h in hits:
        title = (h.get("title") or "").strip()
        if not title:
            continue
        out.append({"title": title,
                    "body": (h.get("body") or "").strip(),
                    "age": _age_hours(h.get("date", ""))})
    return out


def _headlines(per_seed: int = 6, keep: int = 20) -> list[dict]:
    """Freshest-first contested headlines across all seeds, deduped."""
    with ThreadPoolExecutor(max_workers=len(_SEEDS)) as pool:
        batches = pool.map(lambda s: _one_seed(s, per_seed), _SEEDS)

    items: list[dict] = []
    seen: set[str] = set()
    for b in batches:
        for h in b:
            if h["age"] is not None and h["age"] > _MAX_AGE_HOURS:
                continue
            key = re.sub(r"[^a-z0-9]+", " ", h["title"].lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            items.append(h)

    # Undated items sort just past the age cutoff: worth keeping, never ahead of
    # a headline we know is fresh.
    items.sort(key=lambda h: h["age"] if h["age"] is not None
               else _MAX_AGE_HOURS + 1)
    return items[:keep]


def _headline_claims(items: list[dict], n: int) -> list[str]:
    """Claims straight from the freshest headline titles — used when the LLM is
    unavailable. Rougher phrasing than the model's, but still current news."""
    return [_shorten(h["title"]) for h in items[:n]]


def _do_fetch(n: int, client: LLMClient | None) -> None:
    """Blocking live search + LLM; update the cache on success."""
    items = _headlines()
    if not items:
        return
    client = client or make_client()
    today = datetime.now(timezone.utc).strftime("%d %B %Y")
    listing = "\n".join(f"- [{_ago(h['age'])}] {h['title']}"
                        + (f" — {h['body']}" if h["body"] else "")
                        for h in items)
    user = (
        f"Today is {today}. Below are the latest news headlines about contested, "
        f"controversial stories.\n\nWrite {n} short, self-contained, CHECKABLE "
        "factual claims drawn from these headlines. Rules:\n"
        "- Every claim must come from a RECENT event in the headlines below — "
        "never general knowledge, history, or well-known myths.\n"
        "- Prefer the freshest and most contested items; the ones at the top of "
        "the list are the most recent.\n"
        "- One sentence, under 16 words, phrased as a neutral statement a "
        "fact-checker could verify or refute — not a question, not an opinion.\n"
        "- Name the specific person, company, country, or organisation involved, "
        "so the claim stands alone without its headline.\n"
        "- Cover a variety of topics: no two claims from the same story.\n\n"
        f"Headlines:\n{listing}"
    )
    try:
        data, _ = client.structured(_SYSTEM, user, _SCHEMA, max_tokens=500)
        claims = [_shorten(c) for c in data.get("claims", [])
                  if isinstance(c, str) and c.strip()][:n]
    except Exception:
        claims = []

    live = bool(claims)
    if not claims:
        claims = _headline_claims(items, n)  # still current, just unpolished
    if claims:
        _cache.update(claims=claims, at=time.time(), live=live)


def _kick_refresh(n: int, client: LLMClient | None) -> None:
    """Refresh the cache in the background; at most one refresh runs at a time."""
    if not _refresh_lock.acquire(blocking=False):
        return
    def job() -> None:
        try:
            _do_fetch(n, client)
        finally:
            _refresh_lock.release()
    threading.Thread(target=job, daemon=True).start()


def is_live() -> bool:
    """True once the cache holds model-written claims from a live news fetch, as
    opposed to headline-derived or static fallbacks. The UI polls on this."""
    return bool(_cache["claims"]) and _cache["live"]


def get_trending_claims(n: int = 6, client: LLMClient | None = None) -> list[str]:
    """Return claims immediately — never blocks on the network, never empty.

    A fresh cache is served as-is; a stale or empty cache is refreshed in the
    background while we serve the last good batch (or built-in fallbacks). So the
    landing page's boxes always have something to show."""
    now = time.time()
    if _cache["claims"] and now - _cache["at"] < _TTL:
        return _cache["claims"]
    _kick_refresh(n, client)
    return _cache["claims"] or _FALLBACK[:n]
