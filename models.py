"""Shared domain models for the research + verifier agent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    id: str
    url: str
    title: str
    content: str  # extracted page text (may be large; budgeted at use-time)

    def to_public(self) -> dict:
        """What we ship to the browser — never the full page text."""
        snippet = self.content[:220].strip()
        if len(self.content) > 220:
            snippet += "…"
        return {"id": self.id, "url": self.url, "title": self.title,
                "snippet": snippet}


@dataclass
class Claim:
    id: str
    text: str
    source_ids: list[str]

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "source_ids": self.source_ids}


@dataclass
class Verdict:
    claim_id: str
    verdict: str            # SUPPORTED | PARTIAL | UNSUPPORTED
    reason: str
    supporting_quote: str
    quote_found: bool

    def to_dict(self) -> dict:
        return {"claim_id": self.claim_id, "verdict": self.verdict,
                "reason": self.reason, "supporting_quote": self.supporting_quote,
                "quote_found": self.quote_found}


@dataclass
class Metrics:
    rounds: int = 0
    total_claims: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    dropped: int = 0
    input_tokens: int = 0        # estimated (wrapper reports no usage)
    output_tokens: int = 0
    context_peak: int = 0        # largest single-call prompt we sent (tokens)

    @property
    def grounding_rate(self) -> float:
        # Partial claims earn half credit (supported=1, partial=0.5, unsupported=0),
        # so an all-partial run scores 0.5 instead of collapsing to 0.
        if not self.total_claims:
            return 0.0
        return (self.supported + 0.5 * self.partial) / self.total_claims

    def to_dict(self) -> dict:
        return {"rounds": self.rounds, "total_claims": self.total_claims,
                "supported": self.supported, "partial": self.partial,
                "unsupported": self.unsupported, "dropped": self.dropped,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "context_peak": self.context_peak,
                "grounding_rate": self.grounding_rate}


@dataclass
class ResearchResult:
    query: str
    report: str
    references: str
    verified: list[Claim] = field(default_factory=list)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    sources: dict[str, Source] = field(default_factory=dict)
    metrics: Metrics = field(default_factory=Metrics)
