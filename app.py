import streamlit as st
from core_engine import (
    query_rag,
    generate_related_questions,
    get_embedding_function,
)
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ────────────────────────────  App configuration  ─────────────────────────────
st.set_page_config(page_title="The Neural Intelligence Lab", layout="wide")

# ─────────────────────────────  State management  ─────────────────────────────
def initialize_state() -> None:
    if "response" not in st.session_state:
        st.session_state.response = None
    if "related_questions" not in st.session_state:
        st.session_state.related_questions = []
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""


# ─────────────────────────────  Helper functions  ─────────────────────────────
def sent_tokenize_regex(text: str) -> list[str]:
    """Very small sentence splitter."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def highlight_text(source_text: str, generated_answer: str, threshold: float = 0.70) -> str:
    """Highlight source sentences that closely match answer sentences."""
    embed = get_embedding_function()

    src_sents = sent_tokenize_regex(source_text)
    ans_sents = sent_tokenize_regex(generated_answer)
    if not src_sents or not ans_sents:
        return source_text

    src_emb = embed.embed_documents(src_sents)
    ans_emb = embed.embed_documents(ans_sents)
    sim = cosine_similarity(ans_emb, src_emb)

    marked = {
        src_sents[np.argmax(row)]
        for row in sim
        if row.max() > threshold
    }

    out = []
    for s in src_sents:
        if s in marked:
            out.append(f"<mark style='background:yellow'>{s}</mark>")
        else:
            out.append(s)
    return " ".join(out)


# ─────────────────────────  Core RAG / LLM pipeline  ──────────────────────────
def handle_query(query: str) -> None:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("GROQ_API_KEY missing from Streamlit secrets.")
        return

    with st.spinner("Synthesizing answer…"):
        answer, sources = query_rag(query, api_key=api_key)
        st.session_state.response = {
            "query": query,
            "answer": answer,
            "sources": sources,
        }

    with st.spinner("Generating related questions…"):
        st.session_state.related_questions = generate_related_questions(
            query, answer, api_key=api_key
        )


# ───────────────────────────────  UI builders  ────────────────────────────────
def render_header() -> None:
    """Title + one-line explainer, centred."""
    st.markdown(
        """
        <h1 style='text-align:center;font-size:2.8rem;font-weight:700;margin:0'>
            The Neural Intelligence Lab
        </h1>
        <p style='text-align:center;font-size:1.25rem;color:#6e6e73;margin-top:8px'>
            Compare how biological brains and artificial intelligence actually work. 
            Ask questions about neurons, neural networks, learning, memory, or 
            decision‑making. Get answers that explore both worlds of intelligence.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.write("")  # 16 px spacer


def render_apple_style_input_area() -> None:
    """Three grey starter-question pills + centred search bar."""
    STARTER_QUESTIONS = [
        "Why can deep learning excel at pattern recognition yet still struggle with the common‑sense reasoning that comes naturally to humans?",
        "How does the brain consolidate memories during sleep, and how could replay‑style mechanisms inspire more robust continual‑learning in AI?",
        "What lessons from human attention can help us design faster, energy‑efficient AI models that run directly on edge devices?",
    ]

    # CSS for Apple-grey pills (#F5F5F7) and larger search bar
    st.markdown(
        """
        <style>
        /* starter-question pills */
        .pill-row > div[data-testid="stButton"] > button{
            background:#F5F5F7 !important;
            border:1px solid #D0D0D0;
            border-radius:16px;
            padding:24px 28px;
            font:600 1.1rem -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
            color:#1D1D1F;
            width:100%;
            height:100%;
            transition:.2s;
        }
        .pill-row > div[data-testid="stButton"] > button:hover{
            border-color:#007aff;color:#007aff;
        }

        /* wider taller search bar */
        input[data-testid="stTextInput"]{
            font-size:1.2rem;
            padding:22px 24px !important;
            height:72px !important;
            border-radius:12px !important;
            width:100% !important;           /* max width */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use the full‑width container (no outer columns)
    with st.container():
        # ——— three-wide pill grid ———
        cols = st.columns([1, 1, 1], gap="small")
        for i, q in enumerate(STARTER_QUESTIONS):
            with cols[i]:
                st.markdown('<div class="pill-row">', unsafe_allow_html=True)
                if st.button(q, key=f"starter_{i}"):
                    st.session_state.user_query = q
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.write("")  # spacer

        # ——— centred label ———
        st.markdown(
            "<h4 style='text-align:center;color:#1D1D1F;margin-bottom:16px;"
            "font-size:2rem;font-weight:700'>"
            "Ask another question</h4>",
            unsafe_allow_html=True,
        )

        # ——— full‑width search bar ———
        with st.container():

            def set_query_from_input():
                st.session_state.user_query = st.session_state.input_query

            st.text_input(
                "Ask your question",
                key="input_query",
                placeholder="Type here…",
                on_change=set_query_from_input,
                label_visibility="collapsed",
            )


def render_response_area() -> None:
    """Answer, sources, and feedback block."""
    st.markdown("---")
    resp = st.session_state.response
    st.markdown(
        "<h3 style='text-align:center;color:#1D1D1F;margin-bottom:12px;"
        "font-size:1.8rem;font-weight:700'>Answer</h3>",
        unsafe_allow_html=True,
    )
    st.write(resp["answer"])
    st.write("")

    if st.session_state.related_questions:
        with st.expander("Explore Related Concepts", expanded=False):
            for q in st.session_state.related_questions:
                if st.button(q, key=f"rel_q_{q}"):
                    st.session_state.user_query = q
                    st.rerun()

    st.markdown("---")
    st.subheader("Sources")
    full_text = "\n\n".join(doc.page_content for doc in resp["sources"])
    with st.expander("View Highlighted Source Text"):
        st.markdown(highlight_text(full_text, resp["answer"]), unsafe_allow_html=True)

    st.markdown("---")
    st.write("Was this answer helpful? (Your feedback is not saved).")
    yes, no, _ = st.columns([1, 1, 5])
    yes.button("Yes", use_container_width=True)
    no.button("No", use_container_width=True)


# ────────────────────────────────  Main flow  ────────────────────────────────
initialize_state()
render_header()
render_apple_style_input_area()

if st.session_state.user_query:
    handle_query(st.session_state.user_query)
    st.session_state.user_query = ""

if st.session_state.response:
    render_response_area()