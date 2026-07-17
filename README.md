# Groundwork — research agent with a verifier pass

A research agent that must cite its evidence. It searches, drafts **atomic claims**,
then a **separate verifier** checks each claim against *only* its cited sources and
drops whatever it can't back up. Everything is streamed to a browser UI, and every
LLM call is **budgeted to fit a small (16k-token) context window**.S

```
query ─▶ Researcher ─▶ Verifier ─▶ Orchestrator ─▶ report + metrics
        (drafts claims) (per-claim   (drop / re-research
                         entailment)  unsupported, bounded)
```

## Why it's more than "chain a search API"

- **Verifier isolation.** The verifier sees one claim and the text of the sources
  that claim cites — never the researcher's other claims, reasoning, or the query
  framing. A critic that can read the actor's confident prose gets talked into
  agreeing.
- **Verify the verifier.** The verifier must paste a verbatim supporting quote.
  If that quote can't be located in the cited source, the verdict is downgraded —
  because critics hallucinate evidence too.
- **Independent verifier call.** Verification is a separate, high-volume call per claim.
  This project targets one self-hosted Qwen model for every stage, so actor and critic
  share a model — the isolation above (not model diversity) is what keeps the critic honest.
- **Bounded re-research.** Failed claims spawn one focused follow-up query per round,
  capped so the loop always terminates.
- **Numbers.** Grounding rate, per-verdict counts, rounds, peak context, and real
  token usage are reported every run.

## Context management (the 16k constraint)

You can't fit five full web pages + a system prompt + room to answer into 16k tokens,
so `context.py` budgets every call before it's sent:

```
input_allowance = max_context − reserved_output − safety_margin
```

Two strategies (see `context.py`):

- **`fit_sources()`** — for the researcher, which needs breadth. Splits the allowance
  evenly across sources and truncates each; drops the weakest tail if even the shares
  would be too small (search returns most-relevant first).
- **`window_for_claim()`** — for the verifier, which needs the *one* relevant passage.
  Segments a page (paragraphs, falling back to sentences), scores segments by keyword
  overlap with the claim, and keeps only the best ones in reading order — so the
  verifier sees the part of a long page that matters, not just its first N chars.

Token counts use a conservative chars-per-token heuristic; the safety margin covers
estimate error. Swap in an exact tokenizer by replacing `estimate_tokens()`.

## Files

```
backend/
  models.py       dataclasses (Source, Claim, Verdict, Metrics, ResearchResult)
  context.py      token budgeting, truncation, relevance windowing  ← 16k logic
  search.py       SearchProvider protocol + DuckDuckGo (keyless) + trafilatura fetch
  agent.py        Researcher + Verifier + Orchestrator, emits stream events
  trending.py     latest contested headlines → checkable claims for the landing boxes
  server.py       FastAPI + Server-Sent Events, serves the UI
  requirements.txt
frontend/
  index.html      streaming console UI: pipeline, budget gauge, verdict cards, paper report
```

## Run

```bash
pip install -r requirements.txt

# Fill in .env (already scaffolded in this folder):
#   QWEN_API_URL    your self-hosted Qwen wrapper base URL (default http://127.0.0.1:8000)
#   QWEN_API_KEY    the api-key your wrapper expects

uvicorn server:app --reload
```

The agent talks to your self-hosted Qwen wrapper's `POST /chat/text` endpoint (see
`llm.py`) instead of a hosted LLM API. Web search is **keyless** — DuckDuckGo (the
`ddgs` package) finds URLs and `trafilatura` extracts each page's article text
(see `search.py`), so no search API key is needed.

Open http://127.0.0.1:8000 and ask something checkable, e.g.
*"did the EU AI Act ban emotion recognition at work?"*

Watch claims arrive as pending cards, then flip to SUPPORTED / PARTIAL / UNSUPPORTED
with their evidence quote, while the context gauge shows each call's size against the
16k ceiling. The verified answer prints on the paper card at the end.

> You can open `frontend/index.html` directly (no backend) just to see the idle shell;
> the live pipeline needs the server running.

## Swapping pieces

- **Search provider:** implement `.search(query, k) -> list[Source]` (see `search.py`)
  for Brave, SerpAPI, or DuckDuckGo + your own fetcher. Everything else is unchanged.
- **LLM endpoint / budgets:** point `QWEN_API_URL` at any wrapper that exposes the same
  `/chat/text` contract; the `*_OUTPUT` constants and `Budget(...)` are the other knobs
  (`max_context` is passed per request from the UI).

## Known limits / next steps

- Heuristic token counting (swap for an exact tokenizer for tighter packing).
- Re-research derives sub-queries naively from failed claim text.
- No eval harness yet — a frozen 30-task set with a scorer that reports grounding
  rate would let you prove a prompt change actually helped, and catch regressions.
