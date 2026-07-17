"""
Research agent with a verifier pass — context-budgeted for a small window.

Actor (Researcher) drafts atomic claims from budgeted sources. Critic (Verifier)
checks each claim against ONLY its cited sources, windowed to the relevant
passage. Unsupported claims are dropped or re-researched for a bounded number of
rounds. Every LLM call is sized to fit `Budget.max_context` first.

Pass `on_event` to stream progress: {"type": ...} dicts for search, sources,
claims, per-claim verdicts, context usage, the final report, and metrics.
"""

from __future__ import annotations

import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Callable

import context as ctx
from context import Budget
from llm import QwenClient
from models import Claim, Metrics, ResearchResult, Source, Verdict
from search import SearchProvider

# The self-hosted Qwen wrapper serves ONE model for every call. Verifier
# ISOLATION (it sees only the claim + its cited source) still holds; the old
# "different, cheaper model for the critic" property does not — there is one model.

# Output reserves per call (leaves room in the window for the reply).
DRAFT_OUTPUT = 2200
VERIFY_OUTPUT = 500
WRITE_OUTPUT = 1500

VERIFY_WORKERS = 6
Emit = Callable[[dict], None]


# --- prompts ----------------------------------------------------------------
RESEARCHER_SYSTEM = textwrap.dedent("""
    You are a research analyst. Using ONLY the numbered sources provided, produce
    atomic, self-contained factual claims that answer the question.
    Rules:
    - One fact per claim. Each must stand alone (no "it"/"this"/"the above").
    - In source_ids, cite ONLY sources whose text actually contains the fact.
      Never invent a source id. Never cite a source you did not use.
    - Preserve tense and time exactly. If a source describes something planned,
      announced, expected, or in the future, phrase the claim that way ("plans
      to", "announced", "is expected to") — never state it as already done or
      currently true. Include the relevant date when the source gives one.
    - The sources may disagree or update one another. When a later source
      reverses, corrects, or supersedes an earlier one, draft a claim for the
      CURRENT state, and when useful a separate claim capturing the reversal.
    - If the sources do not cover part of the question, omit it. Never guess or
      add outside knowledge.
""").strip()

VERIFIER_SYSTEM = textwrap.dedent("""
    You are a fact-checking verifier. You get ONE claim, today's date, and the
    text of the source(s) it cites — nothing else. Judge ONLY on what the source
    literally says, and be strict about time.
      SUPPORTED   - the source directly and fully states the claim, in the SAME
                    tense and time frame. "Will marry" / "plans to" / "announced"
                    does NOT support a claim that it has already happened or is
                    currently true.
      PARTIAL     - related or once-true but not currently established: the source
                    supports only a past, planned, or announced version; or the
                    situation was later changed, reversed, or superseded; or a
                    number, qualifier, or step is missing.
      UNSUPPORTED - not established by the source, or contradicted.
    Check tense and dates: a claim stated as current or completed needs a source
    that says so as of now — not a future plan, and not an older statement that
    was later reversed. You MUST paste a supporting_quote copied VERBATIM
    (<=25 words) from the source. If nothing supports it, return an empty quote
    and UNSUPPORTED. No outside knowledge. Do not be charitable.
""").strip()

WRITER_SYSTEM = textwrap.dedent("""
    Write a concise, well-organized answer using ONLY the verified claims supplied.
    Every sentence must trace to a claim; cite claim ids inline like [c3]. Do not
    introduce any fact not present in the verified claims. If the claims include a
    later correction, reversal, or update, reflect the CURRENT state and say what
    changed and when. If the verified claims are thin or conflict, say so plainly
    rather than padding.
""").strip()


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip().lower()


class ResearchAgent:
    def __init__(self, search: SearchProvider, budget: Budget | None = None,
                 client: QwenClient | None = None, k: int = 5):
        self.search = search
        self.budget = budget or Budget()
        self.client = client or QwenClient()
        self.k = k
        self.sources: dict[str, Source] = {}
        self._src_n = 0
        self._claim_n = 1
        self._cumulative_tokens = 0

    # --- structured LLM call (schema-in-prompt → parsed JSON) ---------------
    def _structured(self, system: str, user: str, schema: dict,
                    output_reserve: int, m: Metrics, emit: Emit, stage: str) -> dict:
        prompt_tokens = ctx.estimate_tokens(system) + ctx.estimate_tokens(user)
        m.context_peak = max(m.context_peak, prompt_tokens + output_reserve)
        self._cumulative_tokens += prompt_tokens
        emit({"type": "context", "stage": stage,
              "call_tokens": prompt_tokens + output_reserve,
              "prompt_tokens": prompt_tokens, "output_reserve": output_reserve,
              "max_context": self.budget.max_context,
              "cumulative_tokens": self._cumulative_tokens})

        # The wrapper has no tool-use mode: ask for JSON, parse it back. Usage is
        # not reported by the wrapper, so estimate it from the prompt and reply.
        data, text = self.client.structured(system, user, schema, output_reserve)
        m.input_tokens += prompt_tokens
        m.output_tokens += ctx.estimate_tokens(text)
        return data

    # --- source registry (global ids, dedup by url) -------------------------
    def _ingest(self, raw: list[Source]) -> list[Source]:
        by_url = {s.url: s for s in self.sources.values()}
        out: list[Source] = []
        for s in raw:
            if s.url in by_url:
                out.append(by_url[s.url])
                continue
            self._src_n += 1
            s.id = f"s{self._src_n}"
            self.sources[s.id] = s
            by_url[s.url] = s
            out.append(s)
        return out

    # --- balanced gathering (supporting + contradicting evidence) -----------
    def _counter_query(self, query: str) -> str:
        """Ask the model for a search that surfaces contradicting/updated evidence."""
        schema = {"type": "object", "properties": {"query": {"type": "string"}},
                  "required": ["query"]}
        system = "You write concise web-search queries."
        user = ("Write ONE short web search query to surface evidence that "
                "CONTRADICTS, corrects, reverses, updates, or fact-checks the most "
                "likely answer to this question — including newer developments that "
                f"might overturn it.\n\nQuestion: {query}")
        try:
            data, _ = self.client.structured(system, user, schema, 120)
            return (data.get("query") or "").strip()
        except Exception:
            return ""

    def _gather(self, query: str, emit: Emit) -> list[Source]:
        """Search for supporting AND contradicting/updated evidence, then merge."""
        raw = list(self.search.search(query, k=self.k))
        counter = self._counter_query(query)
        if counter and _norm(counter) != _norm(query):
            emit({"type": "note",
                  "message": f'Also searching for contradicting or updated '
                             f'evidence: "{counter}"'})
            raw += self.search.search(counter, k=max(2, self.k - 2))
        return self._ingest(raw)

    # --- researcher (budgeted breadth) --------------------------------------
    def _draft(self, query: str, sources: list[Source], m: Metrics,
               emit: Emit) -> list[Claim]:
        today = date.today().isoformat()
        head = f"Today's date is {today}.\n\n"
        overhead = (ctx.estimate_tokens(RESEARCHER_SYSTEM)
                    + ctx.estimate_tokens(head + f"Question: {query}\n\nSources:\n\nEmit the claims."))
        fitted, _ = self.budget.fit_sources(sources, DRAFT_OUTPUT, overhead)
        if len(fitted) < len(sources):
            emit({"type": "note",
                  "message": f"Context budget: fed {len(fitted)}/{len(sources)} "
                             f"sources to the researcher to stay under "
                             f"{self.budget.max_context} tokens."})

        block = "\n\n".join(f"[{s.id}] {s.title}\n{s.url}\n{s.content}"
                            for s in fitted)
        user = f"{head}Question: {query}\n\nSources:\n{block}\n\nEmit the claims."
        schema = {"type": "object", "properties": {"claims": {"type": "array",
                  "items": {"type": "object", "properties": {
                      "text": {"type": "string"},
                      "source_ids": {"type": "array", "items": {"type": "string"}}},
                      "required": ["text", "source_ids"]}}}, "required": ["claims"]}
        data = self._structured(RESEARCHER_SYSTEM, user, schema,
                                DRAFT_OUTPUT, m, emit, stage="draft")
        claims: list[Claim] = []
        for c in data.get("claims", []):
            claims.append(Claim(id=f"c{self._claim_n}", text=c["text"].strip(),
                                source_ids=list(dict.fromkeys(c["source_ids"]))))
            self._claim_n += 1
        return claims

    # --- verifier (isolated + windowed to the relevant passage) -------------
    def _verify_one(self, claim: Claim, m: Metrics, emit: Emit) -> Verdict:
        cited = [self.sources[sid] for sid in claim.source_ids
                 if sid in self.sources]
        if not cited:  # researcher cited a non-existent source → uncheckable
            return Verdict(claim.id, "UNSUPPORTED", "Claim cited no valid source.",
                           "", False)

        today = date.today().isoformat()
        head = f"Today's date is {today}.\n\n"
        overhead = (ctx.estimate_tokens(VERIFIER_SYSTEM)
                    + ctx.estimate_tokens(head + f"Claim: {claim.text}\n\nCited source(s):\n"))
        allowance = self.budget.input_allowance(VERIFY_OUTPUT) - overhead
        per_source = min(3000, max(200, allowance // len(cited)))

        parts = []
        for s in cited:
            windowed = ctx.window_for_claim(s.content, claim.text, per_source)
            parts.append(f"[{s.id}] {s.title}\n{windowed}")
        user = f"{head}Claim: {claim.text}\n\nCited source(s):\n" + "\n\n".join(parts)

        schema = {"type": "object", "properties": {
            "verdict": {"type": "string",
                        "enum": ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]},
            "reason": {"type": "string"},
            "supporting_quote": {"type": "string"}},
            "required": ["verdict", "reason", "supporting_quote"]}
        data = self._structured(VERIFIER_SYSTEM, user, schema,
                                VERIFY_OUTPUT, m, emit, stage="verify")

        quote = data.get("supporting_quote", "").strip()
        verdict = data["verdict"]
        reason = data["reason"]
        found = bool(quote) and any(_norm(quote) in _norm(s.content) for s in cited)
        # verify the verifier: SUPPORTED needs a real, locatable quote
        if verdict == "SUPPORTED" and not found:
            verdict = "PARTIAL"
            reason += " [downgraded: supporting quote not found in source]"
        return Verdict(claim.id, verdict, reason, quote, found)

    def _verify_all(self, claims: list[Claim], m: Metrics,
                    emit: Emit) -> dict[str, Verdict]:
        verdicts: dict[str, Verdict] = {}
        with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as pool:
            futures = {pool.submit(self._verify_one, c, m, emit): c for c in claims}
            for fut in as_completed(futures):  # emit as each check lands
                v = fut.result()
                verdicts[v.claim_id] = v
                emit({"type": "verdict", **v.to_dict()})
        return verdicts

    # --- assembly (verified claims only) ------------------------------------
    def _assemble(self, query: str, verified: list[Claim], m: Metrics,
                  emit: Emit) -> tuple[str, str]:
        if not verified:
            return ("No claim survived verification. The sources did not support a "
                    "grounded answer to this question.", "")
        block = "\n".join(f"[{c.id}] {c.text}  (sources: {', '.join(c.source_ids)})"
                          for c in verified)
        schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"]}
        data = self._structured(WRITER_SYSTEM,
                                f"Question: {query}\n\nVerified claims:\n{block}",
                                schema, WRITE_OUTPUT, m, emit, stage="assemble")
        return data["answer"].strip(), self._references(verified)

    def _references(self, verified: list[Claim]) -> str:
        used = {sid for c in verified for sid in c.source_ids if sid in self.sources}
        lines = []
        for sid in sorted(used, key=lambda x: int(x[1:])):
            s = self.sources[sid]
            lines.append(f"[{sid}] {s.title} — {s.url}")
        return "\n".join(lines)

    # --- main loop ----------------------------------------------------------
    def run(self, query: str, max_rounds: int = 2, keep_partial: bool = False,
            on_event: Emit | None = None) -> ResearchResult:
        emit: Emit = on_event or (lambda e: None)
        m = Metrics()
        verified: list[Claim] = []
        all_verdicts: dict[str, Verdict] = {}
        pending_query = query

        emit({"type": "start", "query": query,
              "max_context": self.budget.max_context, "max_rounds": max_rounds})

        for rnd in range(max_rounds):
            m.rounds = rnd + 1
            emit({"type": "status", "stage": "search", "round": m.rounds,
                  "query": pending_query})
            sources = self._gather(pending_query, emit)
            emit({"type": "sources", "round": m.rounds,
                  "sources": [s.to_public() for s in sources]})

            emit({"type": "status", "stage": "draft", "round": m.rounds})
            claims = self._draft(pending_query, sources, m, emit)
            m.total_claims += len(claims)
            if not claims:
                emit({"type": "note", "message": "No claims drafted this round."})
                break
            emit({"type": "claims", "round": m.rounds,
                  "claims": [c.to_dict() for c in claims]})

            emit({"type": "status", "stage": "verify", "round": m.rounds})
            verdicts = self._verify_all(claims, m, emit)
            all_verdicts.update(verdicts)

            failed: list[Claim] = []
            for c in claims:
                v = verdicts[c.id]
                if v.verdict == "SUPPORTED":
                    m.supported += 1
                    verified.append(c)
                elif v.verdict == "PARTIAL":
                    m.partial += 1
                    (verified if keep_partial else failed).append(c)
                else:
                    m.unsupported += 1
                    failed.append(c)

            if failed and rnd < max_rounds - 1:
                topics = "; ".join(c.text for c in failed[:4])
                pending_query = f"{query} -- verify specifically: {topics}"
                emit({"type": "requery", "round": m.rounds, "failed": len(failed)})
            else:
                m.dropped += len(failed)
                break

        emit({"type": "status", "stage": "assemble"})
        report, refs = self._assemble(query, verified, m, emit)
        emit({"type": "report", "report": report, "references": refs})
        emit({"type": "metrics", **m.to_dict()})
        emit({"type": "done"})

        return ResearchResult(query, report, refs, verified, all_verdicts,
                              self.sources, m)
