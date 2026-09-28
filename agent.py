"""
Research agent with a verifier pass — context-budgeted for a small window.

Planner reads the input (a question, or a statement to fact-check) and splits it
into 1-4 sub-questions. One Researcher per sub-question searches and drafts
atomic claims from its own budgeted sources, in parallel. Critic (Verifier)
checks each claim against ONLY its cited sources, windowed to the relevant
passage. Unsupported claims are dropped or re-researched for a bounded number of
rounds. Every LLM call is sized to fit `Budget.max_context` first; sources per
run and claims per round are capped to keep runs fast.

Pass `on_event` to stream progress: {"type": ...} dicts for search, sources,
claims, per-claim verdicts, context usage, the final report, and metrics.
"""

from __future__ import annotations

import re
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import context as ctx
from context import Budget
from llm import LLMClient, make_client
from models import Claim, Conflict, Metrics, ResearchResult, Source, Verdict
from search import SearchProvider

# One configured LLM (see llm.py) serves ONE model for every call. Verifier
# ISOLATION (it sees only the claim + its cited source) still holds; the old
# "different, cheaper model for the critic" property does not — there is one model.

# Output reserves per call (leaves room in the window for the reply).
DRAFT_OUTPUT = 1600
VERIFY_OUTPUT = 500
WRITE_OUTPUT = 1500
FOLLOWUP_OUTPUT = 200
CONFLICT_OUTPUT = 800
PLAN_OUTPUT = 500

# Run-size caps: these, not the context window, decide how long a run takes.
MAX_SOURCES = 20        # distinct pages per run, all rounds together
FOLLOWUP_SOURCES = 5    # of those, held back for the follow-up round
MAX_CLAIMS = 12         # claims drafted per round (each costs a verify call)
MAX_SUBQUESTIONS = 4
ENOUGH_VERIFIED = 6     # skip the follow-up round once this many claims hold up
                        # (or when 3/4 of the round's claims did)

VERIFY_WORKERS = 6
RESEARCH_WORKERS = 4    # researchers drafting at once
SEARCH_CONCURRENCY = 1  # web searches at once (DDG drops parallel bursts)
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
    - Each source shows its publish date when known. When sources disagree,
      prefer the most recent for the current state, and put the date in the
      claim ("As of March 2025, ...") when the fact can change over time.
    - If the sources do not cover part of the question, omit it. Never guess or
      add outside knowledge.
    - Stay on topic: only draft claims that help answer the question. Skip
      background (founders, funding, history, product trivia) unless it bears
      directly on the answer. Respect the claim limit you are given; pick the
      most relevant facts, not the first ones you see.
""").strip()

PLANNER_SYSTEM = textwrap.dedent("""
    You plan web research. The input is either a QUESTION, or a STATEMENT the
    user wants fact-checked (e.g. "X is the best Y and will replace Z").
    - kind: "question" or "statement".
    - focus: the single question the final answer must address. For a
      statement, rephrase it as a checkable question ("Is X ...? Will X ...?")
      keeping all its parts.
    - sub_questions: 1 to 4 narrower questions that together answer the focus,
      each with a short web search (3-8 words, no quotes or operators). Use ONE
      for a simple factual lookup; more only when the input has several parts
      or needs a comparison. For a statement or anything contested, make one
      sub-question look for criticism, limitations, or evidence against it.
      Sub-questions must not overlap.
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
    was later reversed. Each source shows its publish date (or "unknown"): a
    claim about the present backed only by an OLD source describing a situation
    that can change (office holders, prices, rankings, laws, records) is at most
    PARTIAL. You MUST paste a supporting_quote copied VERBATIM
    (<=25 words) from the source. If nothing supports it, return an empty quote
    and UNSUPPORTED. No outside knowledge. Do not be charitable.
""").strip()

WRITER_SYSTEM = textwrap.dedent("""
    Write a concise, well-organized answer using ONLY the verified claims supplied.
    Your FIRST sentence must answer the focus question directly. If the input
    was a STATEMENT to fact-check, begin with a verdict — "Supported.",
    "Partly supported.", "Not supported.", or "Unverifiable." (for opinions
    and predictions no source can settle) — then explain why, part by part.
    Leave out claims that do not bear on the focus; do not summarise
    everything you were given.
    Every sentence must trace to a claim; cite claim ids inline like [c3]. Do not
    introduce any fact not present in the verified claims. If the claims include a
    later correction, reversal, or update, reflect the CURRENT state and say what
    changed and when. If the verified claims are thin or conflict, say so plainly
    rather than padding. When KNOWN CONFLICTS are listed, never state both sides
    as fact: present the current/most recent one, and say the sources disagree
    (citing both) when it is unclear which is right.
""").strip()

CONFLICT_SYSTEM = textwrap.dedent("""
    You are a consistency checker. You get a numbered list of claims that were
    each verified against a source, with the publish date of those sources.
    Find claims that CANNOT all be true at the same time: different numbers or
    dates for the same thing, one says X happened and another says it did not,
    an old state vs a newer state of the same fact, etc.
    Claims that are merely about different aspects, or that are compatible
    are NOT conflicts: one more specific than another, or an approximate figure
    ("about", "roughly", "over", "nearly") that the exact figure falls within
    (e.g. "about 5,000" and "5,120" agree). Be precise: report only
    real contradictions; an empty list is a normal answer.
    For each candidate give the claim ids involved, a one-sentence explanation,
    then "contradicts": true only if the claims truly cannot both be true
    (false if on reflection they are compatible — then better to leave the
    candidate out entirely), and in "current" the id of the claim that
    reflects the latest state if the dates make that clear, else "".
    "X runs the company" and "the company has no named CEO" are compatible.
    A claim explicitly framed as PAST ("Before 2024, X was ...", "X served
    until ...") is background and compatible with the current state. But a
    claim stated in the PRESENT tense ("X is the CEO") that a newer claim shows
    has since changed IS a conflict — the older claim is stale.
""").strip()

FOLLOWUP_SYSTEM = textwrap.dedent("""
    You write web search queries. Given a research question and claims that
    could NOT be verified from the sources found so far, write 1-3 short search
    queries (3-8 words each, like a person would type into a search engine)
    that would find authoritative, recent sources to confirm or refute them.
    Target the specific entity, number, or event in doubt. No quotes, no
    operators, no full sentences.
""").strip()


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip().lower()


@dataclass
class SubQuestion:
    question: str
    search: str


@dataclass
class Plan:
    kind: str                     # "question" | "statement"
    focus: str                    # what the final answer must address
    subs: list[SubQuestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "focus": self.focus,
                "sub_questions": [{"question": q.question, "search": q.search}
                                  for q in self.subs]}


class ResearchAgent:
    def __init__(self, search: SearchProvider, budget: Budget | None = None,
                 client: LLMClient | None = None, k: int = 5,
                 max_sources: int = MAX_SOURCES, max_claims: int = MAX_CLAIMS):
        self.search = search
        self.budget = budget or Budget()
        self.client = client or make_client()
        self.k = k
        self.max_sources = max_sources
        self.max_claims = max_claims
        self.sources: dict[str, Source] = {}
        self._src_n = 0
        self._claim_n = 1
        self._cumulative_tokens = 0
        self._cap_noted = False
        # researchers and verifiers run in threads and share these counters
        self._lock = threading.Lock()
        self._search_slots = threading.Semaphore(SEARCH_CONCURRENCY)

    # --- structured LLM call (schema-in-prompt → parsed JSON) ---------------
    def _structured(self, system: str, user: str, schema: dict,
                    output_reserve: int, m: Metrics, emit: Emit, stage: str) -> dict:
        prompt_tokens = ctx.estimate_tokens(system) + ctx.estimate_tokens(user)
        with self._lock:
            m.context_peak = max(m.context_peak, prompt_tokens + output_reserve)
            self._cumulative_tokens += prompt_tokens
            cumulative = self._cumulative_tokens
        emit({"type": "context", "stage": stage,
              "call_tokens": prompt_tokens + output_reserve,
              "prompt_tokens": prompt_tokens, "output_reserve": output_reserve,
              "max_context": self.budget.max_context,
              "cumulative_tokens": cumulative})

        # The wrapper has no tool-use mode: ask for JSON, parse it back. Usage is
        # not reported by the wrapper, so estimate it from the prompt and reply.
        data, text = self.client.structured(system, user, schema, output_reserve)
        with self._lock:
            m.input_tokens += prompt_tokens
            m.output_tokens += ctx.estimate_tokens(text)
        return data

    # --- source registry (global ids, dedup by url, capped) -----------------
    def _ingest(self, raw: list[Source], emit: Emit, limit: int | None = None) -> list[Source]:
        """Register new sources (dedup by URL) until the run holds `limit`
        (default: max_sources). Already-known pages are always returned."""
        limit = self.max_sources if limit is None else limit
        out: list[Source] = []
        with self._lock:
            by_url = {s.url: s for s in self.sources.values()}
            for s in raw:
                if s.url in by_url:
                    if by_url[s.url] not in out:  # same page from two searches
                        out.append(by_url[s.url])
                    continue
                if len(self.sources) >= limit:
                    if not self._cap_noted:
                        self._cap_noted = True
                        emit({"type": "note", "message": f"Source limit reached "
                              f"({limit}); skipping further pages."})
                    continue
                self._src_n += 1
                s.id = f"s{self._src_n}"
                self.sources[s.id] = s
                by_url[s.url] = s
                out.append(s)
        return out

    def _search(self, query: str, k: int) -> list[Source]:
        with self._search_slots:
            return list(self.search.search(query, k=k))

    # --- planner ------------------------------------------------------------
    def _plan(self, query: str, m: Metrics, emit: Emit) -> Plan:
        schema = {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["question", "statement"]},
            "focus": {"type": "string"},
            "sub_questions": {"type": "array", "items": {"type": "object",
                "properties": {"question": {"type": "string"},
                               "search": {"type": "string"}},
                "required": ["question", "search"]}}},
            "required": ["kind", "focus", "sub_questions"]}
        today = date.today().isoformat()
        try:
            data = self._structured(PLANNER_SYSTEM,
                                    f"Today's date is {today}.\n\nInput: {query}",
                                    schema, PLAN_OUTPUT, m, emit, stage="plan")
        except Exception as exc:
            emit({"type": "note", "message": f"Planner failed ({exc}); "
                  "researching the input as-is."})
            data = {}
        subs, seen = [], set()
        for q in data.get("sub_questions") or []:
            if not isinstance(q, dict):
                continue
            question = str(q.get("question", "")).strip()
            search = str(q.get("search", "")).strip()[:120] or question[:120]
            if question and _norm(search) not in seen:
                seen.add(_norm(search))
                subs.append(SubQuestion(question, search))
        if not subs:
            subs = [SubQuestion(query, query[:120])]
        kind = data.get("kind") if data.get("kind") in ("question", "statement") else "question"
        focus = str(data.get("focus") or "").strip() or query
        return Plan(kind, focus, subs[:MAX_SUBQUESTIONS])

    # --- researchers (one per sub-question, run in parallel) ----------------
    def _research_one(self, plan: Plan, sub: SubQuestion, k: int, max_claims: int,
                      m: Metrics, emit: Emit) -> tuple[list[Source], list[Claim]]:
        sources = self._ingest(self._search(sub.search, k), emit)
        emit({"type": "sources", "round": m.rounds,
              "sources": [s.to_public() for s in sources]})
        if not sources:
            return [], []
        question = sub.question
        if _norm(sub.question) != _norm(plan.focus):
            question = f"{sub.question}\n(Part of the larger question: {plan.focus})"
        return sources, self._draft(question, sources, m, emit, max_claims=max_claims)

    def _research(self, plan: Plan, m: Metrics, emit: Emit) -> list[Claim]:
        n = len(plan.subs)
        budget = self.max_sources - FOLLOWUP_SOURCES
        k = min(8, max(3, budget // n))
        per = max(3, self.max_claims // n)
        results: list[tuple[list[Source], list[Claim]]] = []
        with ThreadPoolExecutor(max_workers=min(n, RESEARCH_WORKERS)) as pool:
            futures = [pool.submit(self._research_one, plan, sub, k, per, m, emit)
                       for sub in plan.subs]
            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception as exc:  # one researcher failing isn't fatal
                    emit({"type": "note", "message": f"A researcher failed: {exc}"})
        claims = [c for _, cs in results for c in cs]
        return _dedupe_claims(claims)[: self.max_claims]

    def _followup_queries(self, query: str, failed: list[Claim]) -> list[str]:
        """Turn failed claims into a few short, targeted search queries."""
        schema = {"type": "object", "properties": {"queries": {
            "type": "array", "items": {"type": "string"}}}, "required": ["queries"]}
        listing = "\n".join(f"- {c.text}" for c in failed[:6])
        user = f"Question: {query}\n\nUnverified claims:\n{listing}"
        try:
            data, _ = self.client.structured(FOLLOWUP_SYSTEM, user, schema,
                                             FOLLOWUP_OUTPUT)
            qs = [q.strip() for q in data.get("queries", [])
                  if isinstance(q, str) and q.strip()]
        except Exception:
            qs = []
        seen, out = {_norm(query)}, []
        for q in qs:
            if _norm(q) not in seen and len(q) <= 120:
                seen.add(_norm(q))
                out.append(q)
        # Fall back to the old behaviour rather than doing nothing.
        return out[:3] or [f"{query} {failed[0].text}"[:200]]

    def _gather_followups(self, queries: list[str], emit: Emit) -> list[Source]:
        emit({"type": "note", "message": "Follow-up searches: "
              + "; ".join(f'"{q}"' for q in queries)})
        raw: list[Source] = []
        for q in queries:
            raw += self._search(q, max(2, self.k - 2))
        return self._ingest(raw, emit)

    # --- researcher (budgeted breadth) --------------------------------------
    def _draft(self, query: str, sources: list[Source], m: Metrics,
               emit: Emit, focus: list[Claim] | None = None,
               max_claims: int = MAX_CLAIMS) -> list[Claim]:
        today = date.today().isoformat()
        head = f"Today's date is {today}. Write at most {max_claims} claims.\n\n"
        if focus:
            head += ("Earlier claims could not be verified. Using these NEW "
                     "sources, confirm, correct, or replace them:\n"
                     + "\n".join(f"- {c.text}" for c in focus[:6]) + "\n\n")
        overhead = (ctx.estimate_tokens(RESEARCHER_SYSTEM)
                    + ctx.estimate_tokens(head + f"Question: {query}\n\nSources:\n\nEmit the claims."))
        fitted, _ = self.budget.fit_sources(sources, DRAFT_OUTPUT, overhead)
        if len(fitted) < len(sources):
            emit({"type": "note",
                  "message": f"Context budget: fed {len(fitted)}/{len(sources)} "
                             f"sources to the researcher to stay under "
                             f"{self.budget.max_context} tokens."})

        block = "\n\n".join(f"[{s.id}] {s.title} ({s.dated})\n{s.url}\n{s.content}"
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
        for c in data.get("claims", [])[:max_claims]:
            if not isinstance(c, dict) or not str(c.get("text", "")).strip():
                continue
            with self._lock:
                cid = f"c{self._claim_n}"
                self._claim_n += 1
            claims.append(Claim(id=cid, text=c["text"].strip(),
                                source_ids=list(dict.fromkeys(c.get("source_ids") or []))))
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
            parts.append(f"[{s.id}] {s.title} ({s.dated})\n{windowed}")
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

    # --- cross-claim consistency --------------------------------------------
    def _claim_line(self, c: Claim) -> str:
        dates = ", ".join(f"{sid} {self.sources[sid].published or 'undated'}"
                          for sid in c.source_ids if sid in self.sources)
        return f"[{c.id}] {c.text}  (sources: {dates})"

    def _find_conflicts(self, query: str, verified: list[Claim], m: Metrics,
                        emit: Emit) -> list[Conflict]:
        """One pass over the verified claims to catch ones that contradict."""
        if len(verified) < 2:
            return []
        today = date.today().isoformat()
        block = "\n".join(self._claim_line(c) for c in verified)
        schema = {"type": "object", "properties": {"conflicts": {
            "type": "array", "items": {"type": "object", "properties": {
                "claim_ids": {"type": "array", "items": {"type": "string"}},
                "explanation": {"type": "string"},
                "contradicts": {"type": "boolean"},
                "current": {"type": "string"}},
                "required": ["claim_ids", "explanation", "contradicts", "current"]}}},
            "required": ["conflicts"]}
        try:
            data = self._structured(
                CONFLICT_SYSTEM,
                f"Today's date is {today}.\nQuestion: {query}\n\nClaims:\n{block}",
                schema, CONFLICT_OUTPUT, m, emit, stage="consistency")
        except Exception as exc:
            emit({"type": "note", "message": f"Consistency check skipped: {exc}"})
            return []
        ids = {c.id for c in verified}
        out: list[Conflict] = []
        for c in data.get("conflicts", []):
            if not isinstance(c, dict) or c.get("contradicts") is False:
                continue
            cids = [i for i in dict.fromkeys(c.get("claim_ids") or []) if i in ids]
            if len(cids) < 2:  # a "conflict" needs two real claims
                continue
            cur = c.get("current") or ""
            out.append(Conflict(cids, str(c.get("explanation", "")).strip(),
                                cur if cur in cids else ""))
        return out

    # --- assembly (verified claims only) ------------------------------------
    def _assemble(self, query: str, verified: list[Claim], m: Metrics,
                  emit: Emit, conflicts: list[Conflict] | None = None,
                  plan: Plan | None = None) -> tuple[str, str]:
        if not verified:
            return ("No claim survived verification. The sources did not support a "
                    "grounded answer to this question.", "")
        block = "\n".join(self._claim_line(c) for c in verified)
        if conflicts:
            block += "\n\nKNOWN CONFLICTS between these claims:\n" + "\n".join(
                f"- {', '.join(c.claim_ids)}: {c.explanation}"
                + (f" (current: {c.current})" if c.current else " (unresolved)")
                for c in conflicts)
        schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"]}
        if plan and plan.kind == "statement":
            head = (f"STATEMENT to fact-check: {query}\n"
                    f"Focus question: {plan.focus}")
        else:
            head = f"Question: {query}"
            if plan and _norm(plan.focus) != _norm(query):
                head += f"\nFocus question: {plan.focus}"
        data = self._structured(WRITER_SYSTEM,
                                f"{head}\n\nVerified claims:\n{block}",
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
        focus: list[Claim] = []
        followups: list[str] = []

        emit({"type": "start", "query": query,
              "max_context": self.budget.max_context, "max_rounds": max_rounds})

        emit({"type": "status", "stage": "plan"})
        plan = self._plan(query, m, emit)
        emit({"type": "plan", **plan.to_dict()})

        for rnd in range(max_rounds):
            m.rounds = rnd + 1
            emit({"type": "status", "stage": "search", "round": m.rounds,
                  "query": "; ".join(followups) or query})
            if followups:
                sources = self._gather_followups(followups, emit)
                if sources:
                    emit({"type": "sources", "round": m.rounds,
                          "sources": [s.to_public() for s in sources]})
                    emit({"type": "status", "stage": "draft", "round": m.rounds})
                    claims = self._draft(plan.focus, sources, m, emit, focus=focus,
                                         max_claims=min(6, max(3, len(focus))))
                    done = {_claim_key(c) for c in verified}
                    claims = [c for c in claims if _claim_key(c) not in done]
                else:
                    claims = []
            else:
                emit({"type": "status", "stage": "draft", "round": m.rounds})
                claims = self._research(plan, m, emit)
            if not self.sources:
                emit({"type": "note", "message": "Search returned nothing "
                      "(the search engine may be rate-limiting)."})
            if followups and not claims:
                m.dropped += len(focus)  # keep what earlier rounds verified
                break
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

            mostly_held = len(claims) - len(failed) >= 0.75 * len(claims)
            if failed and (len(verified) >= ENOUGH_VERIFIED or mostly_held):
                emit({"type": "note", "message": f"{len(verified)} claims verified "
                      f"— enough to answer, skipping the follow-up round."})
                m.dropped += len(failed)
                break
            if failed and rnd < max_rounds - 1:
                focus = failed
                followups = self._followup_queries(plan.focus, failed)
                emit({"type": "requery", "round": m.rounds, "failed": len(failed),
                      "queries": followups})
            else:
                m.dropped += len(failed)
                break

        conflicts = self._find_conflicts(plan.focus, verified, m, emit)
        m.conflicts = len(conflicts)
        if conflicts:
            emit({"type": "conflicts",
                  "conflicts": [c.to_dict() for c in conflicts]})

        emit({"type": "status", "stage": "assemble"})
        report, refs = self._assemble(query, verified, m, emit, conflicts, plan)
        emit({"type": "report", "report": report, "references": refs})
        emit({"type": "metrics", **m.to_dict()})
        emit({"type": "done"})

        return ResearchResult(query, report, refs, verified, all_verdicts,
                              self.sources, m, conflicts)


def _claim_key(c: Claim) -> str:
    return re.sub(r"[^a-z0-9 ]", "", _norm(c.text))


def _dedupe_claims(claims: list[Claim]) -> list[Claim]:
    """Drop near-identical claims from different researchers (same wording
    after normalising), keeping the first and merging its citations."""
    out: dict[str, Claim] = {}
    for c in claims:
        key = _claim_key(c)
        if key in out:
            kept = out[key]
            kept.source_ids = list(dict.fromkeys(kept.source_ids + c.source_ids))
        else:
            out[key] = c
    return list(out.values())
