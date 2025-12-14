#!/usr/bin/env python3
# MSK_Chat/Chatbot/mskbot.py
# Streamlit UI for MSK Neurology RAG (qaEngine v7.6)

import sys
import time
import json
from time import monotonic
from pathlib import Path
import streamlit as st
import html

# ----------------------------------------------------------------------
# Path setup - make VectorDB importable
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "VectorDB"))

from qaEngine import run_qa, agentic_run, detect_embed_model, _backend, QAConfig  # noqa


# ----------------------------------------------------------------------
# Page setup + CSS
# ----------------------------------------------------------------------

st.set_page_config(page_title="MSK Triage Chatbot", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #f2f3f5;
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    section[data-testid="stSidebar"] {
        background-color: #0c0f14;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stChatMessage"] {
        padding: 0.6rem 1rem;
        border-radius: 1rem;
        margin-bottom: 0.75rem;
        max-width: 85%;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset;
        background: #171a21;
        border: 1px solid #262b33;
        color: #f0f3f8;
    }
    .chat-user {
        background: linear-gradient(135deg, #1b64d8 0%, #0f5bc8 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.12);
        margin-left: auto;
        padding: 0.4rem 0.75rem;
        border-radius: 0.9rem;
    }
    .chat-assistant {
        margin-right: auto;
        padding: 0.4rem 0.75rem;
        border-radius: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.4rem;
        font-size: 0.75rem;
        background: #1a2130;
        border: 1px solid #2a3444;
        color: #cbd3df;
        margin-right: 0.4rem;
        margin-bottom: 0.25rem;
    }

    /* Slider thumb: make the knob gray instead of black */
    input[type="range"] {
        -webkit-appearance: none;
        appearance: none;
        height: 6px;
        background: transparent; /* keep default track styling */
    }
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #9aa1a8; /* desired gray color */
        border: 1px solid rgba(0,0,0,0.25);
        box-shadow: none;
        cursor: pointer;
        margin-top: -5px; /* vertically center for many browsers */
    }
    input[type="range"]::-moz-range-thumb {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #9aa1a8;
        border: 1px solid rgba(0,0,0,0.25);
        box-shadow: none;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MSK Triage Chatbot")
st.caption("Evidence-grounded assistant using the MSK Neurology corpus")


# ----------------------------------------------------------------------
# Load backend (cached)
# ----------------------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_backend():
    model_name = detect_embed_model()
    st.write(f"Loading embedder: `{model_name}`")
    embedder = _backend.load_embedder(model_name)
    collection = _backend.load_collection()
    return embedder, collection


EMBEDDER, COLLECTION = load_backend()


# ----------------------------------------------------------------------
# Sidebar: Controls
# ----------------------------------------------------------------------

st.sidebar.header("Retrieval & Context")

top_k = st.sidebar.slider(
    "Sources to sample", 1, 10, 4, 1,
)

per_source_max = st.sidebar.slider(
    "Chunks per source", 1, 5, 3, 1,
)

budget_tokens = st.sidebar.slider(
    "Token Budget", 256, 20000, 10000, 64,
)

neighbor_headroom = st.sidebar.slider(
    "Neighbor Headroom (tokens)", 0, 400, 150, 10,
)

st.sidebar.header("Generation")

openai_model = st.sidebar.text_input(
    "OpenAI Model",
    value="gpt-4.1-mini",
)

num_predict = st.sidebar.slider(
    "Max answer tokens", 64, 10000, 2048, 16,
)

st.sidebar.header("Options")

use_reranker = st.sidebar.checkbox(
    "LLM-based reranker (4.1-mini)", value=True,
)

reranker_top_n = st.sidebar.slider(
    "Reranker top N", 1, 40, 10, 1,
)

include_history = st.sidebar.checkbox(
    "Use conversation memory", value=False,
)

disable_bias = st.sidebar.checkbox(
    "Disable retrieval biases", value=False,
)

stat_view = st.sidebar.checkbox(
    "Show Statistics", value=True,
)

with st.sidebar.expander("Advanced — Conversation memory", expanded=False):
    history_top_entries = st.slider("History entries", 0, 5, 2, 1)
    history_decay = st.slider("History decay", 0.40, 0.95, 0.65, 0.01)
    history_scale = st.slider("History scale", 0.10, 1.00, 0.30, 0.05)
    history_dist_penalty = st.slider("History penalty", 0.00, 0.40, 0.20, 0.01)
    history_use_threshold = st.slider("Memory gating threshold", 0.00, 0.95, 0.55, 0.01)

st.sidebar.header("Session")
if st.sidebar.button("New conversation"):
    st.session_state.clear()
    st.rerun()


# ----------------------------------------------------------------------
# Demo Scenarios
# ----------------------------------------------------------------------

st.sidebar.header("Demo Scenarios")

if st.sidebar.button("Scenario: Scapular Dyskinesis"):
    st.session_state["inject"] = "Explain the biomechanics of scapular dyskinesis."

if st.sidebar.button("Scenario: TOS Mechanism"):
    st.session_state["inject"] = "What structures are usually responsible for thoracic outlet symptoms?"

if st.sidebar.button("Scenario: Atlas Dysfunction"):
    st.session_state["inject"] = "How can atlas rotation contribute to headache patterns?"

if "inject" in st.session_state:
    st.session_state["pending_query"] = st.session_state.pop("inject")
    st.rerun()


# ----------------------------------------------------------------------
# Chat state
# ----------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": (
            "Ask about symptoms, biomechanics, or exercise progressions.\n\n"
            "All answers are grounded strictly in the MSK Neurology dataset."
        ),
    }]

if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None


# ----------------------------------------------------------------------
# Layout: Split-Screen
# ----------------------------------------------------------------------

col_chat, col_eng = st.columns([3, 2])

answer_block = None
context_used = None
trace_data = None
retrieval_confidence = None


# ----------------------------------------------------------------------
# Left Column: Chat UI
# ----------------------------------------------------------------------

with col_chat:

    # Replay history with Markdown inside bubbles
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                # Wrap user text in one HTML block (no Markdown parsing needed for user input)
                safe = html.escape(msg["content"]).replace("\n", "<br/>")
                st.markdown(f'<div class="chat-user">{safe}</div>', unsafe_allow_html=True)
            else:
                # Assistant: render Markdown normally (no HTML wrapper)
                st.markdown(msg["content"])

    pending_query = st.session_state.pop("pending_query", None)

    if pending_query:

        # Add user message to history
        st.session_state["messages"].append({"role": "user", "content": pending_query})

        # Render user bubble (single call so content is inside the wrapper)
        with st.chat_message("user"):
            safe = html.escape(pending_query).replace("\n", "<br/>")
            st.markdown(f'<div class="chat-user">{safe}</div>', unsafe_allow_html=True)

        # Assistant streaming reply
        with st.chat_message("assistant"):

            # Remove the wrapper open/close to avoid empty box
            # Placeholder for streaming inside the assistant message
            placeholder = st.empty()
            placeholder.markdown("_Thinking…_")

            cfg = QAConfig(
                top_k=top_k,
                per_source_max=per_source_max,
                budget_tokens=budget_tokens,
                neighbor_headroom=neighbor_headroom,
                num_predict=num_predict,
                openai_model=openai_model,
                use_reranker=use_reranker,
                reranker_top_n=reranker_top_n,
                include_history=include_history,
                history_top_entries=history_top_entries,
                history_decay=history_decay,
                history_scale=history_scale,
                history_dist_penalty=history_dist_penalty,
                history_use_threshold=history_use_threshold,
                use_bias=not disable_bias,
            )

            answer_accum = [""]
            first_token_time = [None]
            _last_redraw = [0.0]

            # 20ms throttled streaming update
            def on_token(tok: str):
                if first_token_time[0] is None:
                    first_token_time[0] = time.time()

                answer_accum[0] += tok

                now = monotonic()
                if now - _last_redraw[0] > 0.020:  # ~50 FPS
                    # Interpret as Markdown (so headings, lists work)
                    placeholder.markdown(answer_accum[0])
                    _last_redraw[0] = now

            history_state = st.session_state["messages"] if include_history else None

            t0 = time.time()

            try:
                res = agentic_run(
                    pending_query,
                    cfg=cfg,
                    history=history_state,
                    on_token=on_token,
                )

                raw_answer = res.get("answer", None)
                answer_text = raw_answer if raw_answer is not None else answer_accum[0]

                # Final render of full answer (Markdown)
                placeholder.markdown(answer_text)

                # No closing wrapper needed

                answer_block = answer_text
                context_used = res.get("contexts", [])
                retrieval_confidence = res.get("retrieval_confidence", 0.0)
                agentic_category = res.get("category", None)
                agentic_refined_q = res.get("refined_query", None)
                agentic_category_label = res.get("category_label", None)


                # ------------------------------------------------------
                # Source previews / citations
                # ------------------------------------------------------
                citation_previews = {}
                for ctx in context_used:
                    meta = ctx.get("meta", {})
                    label = f"{meta.get('source_relpath','unknown')} — {meta.get('section','n/a')}"
                    txt = ctx.get("text") or ""
                    citation_previews[label] = html.escape(txt.replace("\n", " "), quote=True)

                citations = res.get("citations", [])
                if citations:
                    citation_html = ""
                    for c in citations:
                        # escape the visible label to prevent HTML injection
                        safe_label = html.escape(c, quote=True)
                        preview = citation_previews.get(c, "No preview available")
                        citation_html += (
                            f'<span style="border-bottom:1px dashed #888; cursor:pointer;" '
                            f'title="{preview}">{safe_label}</span> · '
                        )

                    st.markdown(
                        f"<div style='margin-top:0.5rem; font-size:0.85rem; color:#bbb;'>"
                        f"<b>Sources:</b> {citation_html}</div>",
                        unsafe_allow_html=True,
                    )

                # ------------------------------------------------------
                # Timing / token stats
                # ------------------------------------------------------
                rt = float(res.get("retrieval_time", 0.0))
                gt = float(res.get("generation_time", 0.0))
                t_first = first_token_time[0] - t0 if first_token_time[0] else float("nan")
                t_total = time.time() - t0

                p_tok = int(res.get("prompt_tokens", 0))
                o_tok = int(res.get("output_tokens", 0))
                c_tok = int(res.get("context_tokens", 0))
                q_tok = int(res.get("question_tokens", 0))

                st.markdown(
                    f"""
                    <div style="display:flex;gap:.5rem;flex-wrap:wrap;
                                margin:.5rem 0 .75rem 0;font-size:0.85rem">
                      <span class="badge">Retrieval: {rt:.2f}s</span>
                      <span class="badge">First token: {t_first:.2f}s</span>
                      <span class="badge">LLM total: {gt:.2f}s</span>
                      <span class="badge">End-to-end: {t_total:.2f}s</span>
                      <span class="badge">Prompt: {p_tok}</span>
                      <span class="badge">Output: {o_tok}</span>
                      <span class="badge">Context: {c_tok}</span>
                      <span class="badge">Question: {q_tok}</span>
                      <span class="badge">Confidence: {retrieval_confidence:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                trace_data = {
                    "category": agentic_category,
                    "category_label": agentic_category_label,
                    "refined_query": agentic_refined_q,
                    "retrieval_time": rt,
                    "generation_time": gt,
                    "first_token": t_first,
                    "end_to_end": t_total,
                    "prompt_tokens": p_tok,
                    "output_tokens": o_tok,
                    "context_tokens": c_tok,
                    "question_tokens": q_tok,
                    "citations": citations,
                    "confidence": retrieval_confidence,
                }

                # Persist assistant message (Markdown) into history
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer_text}
                )

            except Exception as e:
                # Show error inside the assistant bubble
                placeholder.markdown(f"_Engine error: {e}_")

    user_query = st.chat_input("Ask about symptoms, structures, or treatments...")


if user_query:
    st.session_state["pending_query"] = user_query
    st.rerun()


# ----------------------------------------------------------------------
# Right Column: Engineering Telemetry
# ----------------------------------------------------------------------

with col_eng:
    if stat_view:
        st.subheader("Engine Telemetry")
    
        tab1, tab2, tab3 = st.tabs([
            "Retrieval Metrics",
            "Chunk Inspector",
            "Trace"
        ])

        with tab1:
            st.caption("Measures retrieval confidence, LLM timing, token usage.")
            if trace_data:
                st.metric("Retrieval Confidence", f"{trace_data['confidence']:.2f}")
                st.write("Timing statistics:")
                st.metric("Retrieval time", f"{trace_data['retrieval_time']:.4f}s")
                st.metric("Generation time", f"{trace_data['generation_time']:.4f}s")
                st.metric("End-to-end", f"{trace_data['end_to_end']:.4f}s")

                st.write("Token usage:")
                st.json({
                    "prompt_tokens": trace_data["prompt_tokens"],
                    "output_tokens": trace_data["output_tokens"],
                    "context_tokens": trace_data["context_tokens"],
                    "question_tokens": trace_data["question_tokens"],
                })
                # --- Agentic classification telemetry ---
                st.write("---")
                st.subheader("Agentic Query Processing")

                st.metric("Classification Letter", trace_data.get("category", "N/A"))
                st.write(f"**Meaning:** {trace_data.get("category_label", "N/A")}")

                st.write("**Refined retrieval query:**")
                st.text_area(
                "",
                trace_data.get("refined_query", "N/A"),
                height=150
            )


            else:
                st.info("Ask a question to populate telemetry.")

        with tab2:
            st.caption("Shows the top retrieved chunks from Chroma.")
            if context_used:
                for i, ctx in enumerate(context_used, 1):
                    meta = ctx.get("meta", {})
                    with st.expander(
                        f"[{i}] {meta.get('section','(no section)')} — score={ctx.get('dist'):.3f}"
                    ):
                        st.caption(f"Source: `{meta.get('source_relpath','unknown')}`")
                        st.text(ctx.get("text", ""))
            else:
                st.info("No context available yet.")

        with tab3:
            st.caption("Raw trace JSON for debugging.")
            if trace_data:
                st.json(trace_data)
                st.download_button(
                    "Export trace JSON",
                    data=json.dumps(trace_data, indent=2, ensure_ascii=False),
                    file_name="trace.json",
                    mime="application/json",
                )
            else:
                st.info("No trace available yet.")
    else:
        st.info("Statistics Disabled")
