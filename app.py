"""SmartRAG web app. Run with:  streamlit run app.py"""
import os

import streamlit as st

import config
from pipelines import run

MODE_INFO = {
    "direct": ("⚡", "Direct answer", "Answered from the model's own knowledge, no search."),
    "single": ("🔍", "Single search", "Searched the documents once, then answered."),
    "multi": ("🧠", "Multi-step research", "Searched, reasoned, and searched again until it could answer."),
}
MODE_CHOICES = {"Auto (router decides)": "auto", "Direct answer": "direct", "Single search": "single", "Multi-step research": "multi"}
EXAMPLES = [
    "What does DNS stand for?",
    "What port does HTTPS use by default?",
    "Which routing protocol connects autonomous systems, and is it link-state, distance-vector, or path-vector?",
]

st.set_page_config(page_title="SmartRAG", page_icon="🧭", layout="centered")


@st.cache_resource(show_spinner="Loading models and index...")
def load_components():
    from llm import LLM
    from retriever import Retriever
    from router import Router

    return LLM(), Retriever(), Router()


def check_setup():
    problems = []
    if not os.path.exists(os.path.join(config.INDEX_DIR, "chunks.faiss")):
        problems.append("The search index is missing. Run `python build_index.py` first.")
    if not config.LLM_API_KEY:
        problems.append("No LLM API key found. Copy `.env.example` to `.env` and set `LLM_API_KEY`.")
    for p in problems:
        st.error(p)
    if problems:
        st.stop()


def sidebar(router_kind: str):
    with st.sidebar:
        st.header("Settings")
        choice = st.radio("Answering mode", list(MODE_CHOICES), index=0)
        st.caption(f"Router: `{router_kind}` · LLM: `{config.LLM_MODEL}`")

        history = st.session_state.get("history", [])
        if history:
            st.header("This session")
            n = len(history)
            st.metric("Questions asked", n)
            st.metric("Avg LLM calls", f"{sum(r['llm_calls'] for r in history) / n:.1f}")
            st.metric("Avg time", f"{sum(r['seconds'] for r in history) / n:.1f}s")
            counts = {m: sum(r["mode"] == m for r in history) for m in MODE_INFO}
            st.write("  \n".join(f"{MODE_INFO[m][0]} {MODE_INFO[m][1]}: {c}" for m, c in counts.items()))
    return MODE_CHOICES[choice]


def show_result(r: dict):
    icon, name, description = MODE_INFO[r["mode"]]
    st.subheader(f"{icon} {name}")
    st.caption(description)
    if r["escalated"]:
        st.info("A single search wasn't enough, so the assistant escalated to multi-step research.")

    st.markdown(r["answer"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Time", f"{r['seconds']}s")
    c2.metric("LLM calls", r["llm_calls"])
    c3.metric("Searches", r["searches"])

    if r["router_probs"]:
        with st.expander(f"Why this mode? (router: {r['routed_by']})"):
            for mode, p in sorted(r["router_probs"].items(), key=lambda kv: -kv[1]):
                st.progress(p, text=f"{MODE_INFO[mode][1]}: {p:.0%}")

    with st.expander("Steps taken"):
        for i, step in enumerate(r["trace"], 1):
            if step["action"] == "search":
                st.write(f"{i}. Searched for “{step['query']}” ({step['found']} new passages)")
            else:
                st.write(f"{i}. {step['action'].capitalize()}")

    if r["sources"]:
        with st.expander(f"Sources ({len(r['sources'])})"):
            for i, s in enumerate(r["sources"], 1):
                st.markdown(f"**[{i}] {s['title']}**")
                st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))


def main():
    st.title("SmartRAG")
    st.write(
        "Ask a question about computer networking. The assistant decides how much work it needs: "
        "answer directly, search once, or research in several steps."
    )
    check_setup()
    llm, retriever, router = load_components()
    mode = sidebar(router.kind)

    st.caption("Try: " + " · ".join(f"*{e}*" for e in EXAMPLES))
    with st.form("ask", clear_on_submit=False):
        question = st.text_input("Your question", placeholder="Which transport protocol does HTTP/3 use?")
        submitted = st.form_submit_button("Get answer")

    if submitted and question.strip():
        with st.spinner("Thinking..."):
            try:
                result = run(question.strip(), llm, retriever, router, mode=mode)
            except Exception as exc:
                st.error(f"Couldn't get an answer: {exc}")
                return
        st.session_state.setdefault("history", []).append(result)
        show_result(result)
    elif submitted:
        st.warning("Type a question first.")


main()
