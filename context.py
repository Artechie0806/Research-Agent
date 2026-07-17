"""
Context management for a small (e.g. 16k-token) window.

The whole problem this module solves: you cannot fit five full web pages into a
16k window alongside a system prompt AND leave room for the model to answer. So
every prompt is budgeted before it is sent:

  input_allowance = max_context - reserved_output - safety_margin

Sources are then trimmed to fit that allowance. Two strategies are used:

  fit_sources()      -- for the RESEARCHER, which needs breadth: split the
                        allowance evenly across sources and truncate each.
  window_for_claim() -- for the VERIFIER, which needs the ONE relevant passage:
                        score paragraphs by overlap with the claim and keep only
                        the best ones, in reading order.

Token counts are estimated with a chars-per-token heuristic. It is intentionally
conservative (over-counts slightly) so the safety margin protects us from the
estimate being wrong. Swap in an exact tokenizer by replacing estimate_tokens().
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from models import Source

CHARS_PER_TOKEN = 3.8  # conservative for English; over-estimates a little

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "was", "were", "are",
    "has", "have", "had", "not", "but", "its", "into", "than", "then", "they",
    "their", "which", "who", "what", "when", "where", "will", "would", "can",
    "could", "also", "such", "been", "being", "about", "over", "under", "these",
}


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.8:  # avoid slicing mid-word
        cut = cut[:space]
    return cut.rstrip() + " …[truncated]"


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


_SENT = re.compile(r"(?<=[.!?])\s+")


def _segment(content: str, unit_cap: int) -> list[str]:
    """Split into paragraphs, and further split any paragraph larger than
    `unit_cap` tokens into sentences, so scoring works on either newline-
    delimited or single-spaced pages."""
    blocks = [b.strip() for b in re.split(r"\n{2,}", content) if b.strip()]
    if len(blocks) <= 1:  # no paragraph breaks — go straight to sentences
        blocks = [content]
    units: list[str] = []
    for b in blocks:
        if estimate_tokens(b) <= unit_cap:
            units.append(b)
        else:
            units.extend(s.strip() for s in _SENT.split(b) if s.strip())
    return units


def window_for_claim(content: str, claim_text: str, max_tokens: int) -> str:
    """Return the passages of `content` most relevant to `claim_text`, capped at
    `max_tokens`, preserving reading order. This lets the verifier see the part
    of a long page that actually matters instead of just its first N chars."""
    if estimate_tokens(content) <= max_tokens:
        return content

    kws = _keywords(claim_text)
    # segment finely enough that individual units fit within the cap
    paras = _segment(content, max(20, max_tokens // 2))
    if not paras:
        return truncate_to_tokens(content, max_tokens)

    scored = [(len(kws & _keywords(p)), i, p) for i, p in enumerate(paras)]
    scored.sort(key=lambda x: (-x[0], x[1]))  # best overlap first, tie by order

    picked: list[tuple[int, str]] = []
    total = 0
    for score, i, p in scored:
        t = estimate_tokens(p)
        if total + t > max_tokens:
            continue
        if score == 0 and picked:  # stop adding irrelevant filler once we have some
            break
        picked.append((i, p))
        total += t
        if total >= max_tokens * 0.9:
            break

    if not picked:
        return truncate_to_tokens(content, max_tokens)
    picked.sort(key=lambda x: x[0])  # restore reading order
    return " […] ".join(p for _, p in picked)


@dataclass
class Budget:
    max_context: int = 16000
    safety_margin: int = 500

    def input_allowance(self, output_tokens: int) -> int:
        """Tokens available for the prompt if we reserve `output_tokens` for the
        model's reply and keep a safety margin."""
        return max(0, self.max_context - output_tokens - self.safety_margin)

    def fit_sources(self, sources: list[Source], output_tokens: int,
                    overhead_tokens: int) -> tuple[list[Source], int]:
        """Trim sources so system + query + sources + reply all fit the window.

        Returns (trimmed_sources, tokens_used_by_sources). Drops the least
        sources needed so each surviving source still gets a usable share."""
        allowance = self.input_allowance(output_tokens) - overhead_tokens
        if allowance <= 0 or not sources:
            return [], 0

        MIN_PER_SOURCE = 120
        n = len(sources)
        while n > 1 and allowance // n < MIN_PER_SOURCE:
            n -= 1  # drop the weakest tail; search returns most-relevant first
        share = allowance // n

        fitted: list[Source] = []
        used = 0
        for s in sources[:n]:
            header = f"[{s.id}] {s.title}\n{s.url}\n"
            body_budget = share - estimate_tokens(header)
            body = truncate_to_tokens(s.content, body_budget)
            fitted.append(Source(s.id, s.url, s.title, body))
            used += estimate_tokens(header) + estimate_tokens(body)
        return fitted, used
