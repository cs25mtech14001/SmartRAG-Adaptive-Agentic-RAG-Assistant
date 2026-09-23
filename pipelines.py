"""The three answering strategies and the adaptive orchestrator.

direct      -> ask the LLM with no retrieval
single      -> retrieve once, answer from the passages
multi       -> agent loop: retrieve, let the LLM decide to ANSWER or SEARCH again (up to MAX_STEPS searches)

`run()` picks a strategy (via the router when mode="auto"), and escalates from single to multi
when the single-step answer says the passages don't contain the answer.
"""
import re
import time

import config

MODES = ("direct", "single", "multi")
UNANSWERED_MARKER = "i don't know based on the documents"

DIRECT_PROMPT = """Answer the question in 1-3 sentences.

Question: {question}
Answer:"""

GROUNDED_PROMPT = """Answer the question using ONLY the passages below. Cite the passages you use like [1] or [2].
If the passages do not contain the answer, reply exactly: I don't know based on the documents.

Passages:
{context}

Question: {question}
Answer:"""

DECIDE_PROMPT = """You are researching a question that may need several searches.

Question: {question}

Passages found so far:
{context}

If the passages are enough to answer the full question, reply in this form:
ANSWER: <answer in 1-4 sentences, citing passages like [1]>

Otherwise, reply with ONE short search query for the missing information, in this form:
SEARCH: <query>"""

_ACTION = re.compile(r"\b(ANSWER|SEARCH)\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


class _CountingLLM:
    """Counts LLM calls for one request (safe even when the underlying client is shared)."""

    def __init__(self, llm):
        self.llm = llm
        self.calls = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return self.llm(prompt, **kwargs)


def format_context(passages: list[dict]) -> str:
    if not passages:
        return "(none yet)"
    return "\n\n".join(f"[{i}] ({p['title']}) {p['text']}" for i, p in enumerate(passages, 1))


def parse_action(reply: str) -> tuple[str, str]:
    """Return ("answer", text) or ("search", query). Anything unparseable counts as an answer."""
    cleaned = reply.replace("*", "")
    match = _ACTION.search(cleaned)
    if not match:
        return "answer", reply.strip()
    action, content = match.group(1).lower(), match.group(2).strip()
    if action == "search":
        lines = content.splitlines()
        content = lines[0].strip().strip('"').strip() if lines else ""
    return action, content


def is_unanswered(answer: str) -> bool:
    return UNANSWERED_MARKER in answer.lower().replace("’", "'")


def direct(question: str, llm) -> dict:
    answer = llm(DIRECT_PROMPT.format(question=question))
    return {"answer": answer, "sources": [], "searches": 0, "trace": [{"action": "answer from model knowledge"}]}


def single_step(question: str, llm, retriever, top_k: int = config.TOP_K) -> dict:
    passages = retriever.search(question, top_k)
    answer = llm(GROUNDED_PROMPT.format(context=format_context(passages), question=question))
    return {
        "answer": answer,
        "sources": passages,
        "searches": 1,
        "trace": [{"action": "search", "query": question, "found": len(passages)}, {"action": "answer"}],
    }


def multi_step(
    question: str,
    llm,
    retriever,
    top_k: int = config.TOP_K,
    max_steps: int = config.MAX_STEPS,
    max_notes: int = config.MAX_NOTES,
) -> dict:
    notes, seen_ids, queries, trace = [], set(), [], []

    def search(query: str):
        hits = retriever.search(query, top_k)
        new = [h for h in hits if h["id"] not in seen_ids][: max(0, max_notes - len(notes))]
        for h in new:
            seen_ids.add(h["id"])
            notes.append(h)
        queries.append(query.lower())
        trace.append({"action": "search", "query": query, "found": len(new)})

    search(question)  # always start with one search on the original question
    for _ in range(max_steps - 1):
        action, content = parse_action(llm(DECIDE_PROMPT.format(question=question, context=format_context(notes))))
        if action == "answer" and content:
            trace.append({"action": "answer"})
            return {"answer": content, "sources": notes, "searches": len(queries), "trace": trace}
        if not content or content.lower() in queries:
            break  # no new query to try
        search(content)

    answer = llm(GROUNDED_PROMPT.format(context=format_context(notes), question=question))
    trace.append({"action": "final answer (step limit reached)"})
    return {"answer": answer, "sources": notes, "searches": len(queries), "trace": trace}


def run(question: str, llm, retriever, router=None, mode: str = "auto", allow_escalation: bool = True) -> dict:
    """Answer a question. mode is "auto" (router decides) or one of MODES."""
    start = time.perf_counter()
    counted = _CountingLLM(llm)
    router_probs, routed_by = None, "manual"

    if mode == "auto":
        if router is None:
            raise ValueError("mode='auto' needs a router")
        mode, router_probs = router.predict(question)
        routed_by = router.kind
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")

    chosen_mode, escalated = mode, False
    if mode == "direct":
        result = direct(question, counted)
    elif mode == "single":
        result = single_step(question, counted, retriever)
        if allow_escalation and is_unanswered(result["answer"]):
            first_trace = result["trace"]
            result = multi_step(question, counted, retriever)
            result["trace"] = first_trace + [{"action": "escalate to multi-step"}] + result["trace"]
            result["searches"] += 1
            mode, escalated = "multi", True
    else:
        result = multi_step(question, counted, retriever)

    result.update(
        question=question,
        mode=mode,
        routed_mode=chosen_mode,
        routed_by=routed_by,
        router_probs=router_probs,
        escalated=escalated,
        llm_calls=counted.calls,
        seconds=round(time.perf_counter() - start, 2),
    )
    return result
