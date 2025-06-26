import streamlit as st
from core_engine import (
    query_rag,
    generate_related_questions,
    get_embedding_function,
)
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ──────────────────────────────  Custom feedback/related button CSS  ──────────────────────────────
st.markdown("""
    <style>
    .feedback-btn button, .related-q-btn button {
        padding: 12px 20px !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)

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
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False
    if "active_starter" not in st.session_state:
        st.session_state.active_starter = ""


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


# ─────────────────────────  Source display helper  ───────────────────────────
def render_sources(retrieved_docs: list, answer: str) -> None:
    """
    Display each retrieved chunk as a neat, collapsible excerpt that
    shows the heading (if available) or a snippet of the first sentence.
    The chunk text is highlighted relative to the generated answer.
    """
    if not retrieved_docs:
        return

    for idx, doc in enumerate(retrieved_docs, start=1):
        # -------- Robust heading extraction --------
        heading = ""

        #  LangChain / LlamaIndex Document objects
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            heading = doc.metadata.get("heading", "")

        # Plain dicts returned by some retrievers
        elif isinstance(doc, dict):
            heading = (
                doc.get("heading", "")
                or doc.get("metadata", {}).get("heading", "")
            )

        #  Fallback to first sentence of the chunk
        if not heading:
            text = doc.get("page_content", "") if isinstance(doc, dict) else doc.page_content
            heading = (text.split(".")[0][:80] + "…") if text else "Untitled"

        st.markdown(f"**Excerpt {idx}: {heading}**")
        highlighted = highlight_text(doc.page_content, answer)
        st.markdown(highlighted, unsafe_allow_html=True)
        if idx < len(retrieved_docs):
            st.markdown("---")


# ─────────────────────────  Core RAG / LLM pipeline  ──────────────────────────
def handle_query(query: str, from_starter: bool = False) -> None:
    st.session_state.feedback_given = False
    st.session_state.active_starter = query if from_starter else ""

    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("API Key is missing. Please create a .env file with GROQ_API_KEY or set Streamlit secrets.")
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
            Compare how brains and artificial intelligence actually work. 
            Ask questions about neurons, neural networks, learning, memory, or 
            decision‑making. Get answers that explore both worlds of intelligence.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.write("")  # 16 px spacer


def render_apple_style_input_area() -> None:
    STARTER_QUESTIONS = [
        "Why can deep learning excel at pattern recognition yet still struggle with the common‑sense reasoning that comes naturally to humans?",
        "How does the brain consolidate memories during sleep, and how could replay‑style mechanisms inspire more robust continual‑learning in AI?",
        "What lessons from human attention can help us design faster, energy‑efficient AI models that run directly on edge devices?",
    ]

    st.markdown(
        """
        <style>
        /* --- Apple‑style pill button for Streamlit native buttons --- */
        div.stButton > button:first-child {
            background:#FFFFFF !important;
            border:1px solid #D0D0D0;
            border-radius:20px;
            padding:36px 48px;
            font:600 1.15rem -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
            color:#1D1D1F;
            text-align:center;
            transition:0.2s;
            cursor:pointer;
            box-shadow:0 4px 12px rgba(0,0,0,0.06);
            margin:16px;
            width:100%;
        }
        div.stButton > button:first-child:hover{
            border-color:#007aff;
            color:#007aff;
            background:#f7f9fc !important;
        }
        div.stButton > button:first-child:focus{
            border:1.5px solid #007aff;
            color:#007aff;
            background:#f0f8ff !important;
        }
        input[data-testid="stTextInput"]{
            font-size:1.2rem;
            padding:22px 24px !important;
            height:72px !important;
            border-radius:12px !important;
            width:100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        cols = st.columns([1, 1, 1], gap="medium")
        for i, q in enumerate(STARTER_QUESTIONS):
            with cols[i]:
                if st.button(q, key=f"starter_{i}", use_container_width=True):
                    st.session_state.user_query = q
                    handle_query(q, from_starter=True)
                    st.rerun()

        st.write("")
        st.markdown(
            "<h4 style='text-align:center;color:#1D1D1F;margin-bottom:16px;"
            "font-size:2rem;font-weight:700'>"
            "Ask another question</h4>",
            unsafe_allow_html=True,
        )

        with st.container():
            def set_query_from_input():
                st.session_state.user_query = st.session_state.input_query.strip()
            st.text_input(
                "Ask your question",
                key="input_query",
                placeholder="Type here…",
                on_change=set_query_from_input,
                label_visibility="collapsed",
            )


def render_response_area() -> None:
    """Answer, sources, and feedback block with one‑time feedback buttons."""
    st.markdown("---")
    resp = st.session_state.response

    # Answer heading
    st.markdown(
        "<h3 style='text-align:center;color:#1D1D1F;margin-bottom:12px;"
        "font-size:1.6rem;font-weight:700'>Answer</h3>",
        unsafe_allow_html=True,
    )
    st.write(resp["answer"])
    st.write("")

    # Related concepts expander (collapsed by default)
    if st.session_state.related_questions:
        with st.expander("Explore Related Concepts", expanded=False):
            for q in st.session_state.related_questions:
                st.markdown('<div class="related-q-btn">', unsafe_allow_html=True)
                if st.button(q, key=f"rel_q_{q}"):
                    st.session_state.user_query = q
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # Sources section
    st.markdown("---")
    st.subheader("Sources")
    with st.expander("View Retrieved Context"):
        retrieved_docs = resp.get("sources", [])
        render_sources(retrieved_docs, resp["answer"])

    st.markdown("---")

    # ------------- Feedback logic -------------
    def set_feedback():
        st.session_state.feedback_given = True

    if st.session_state.feedback_given:
        st.success("Thank you for your feedback!")
    else:
        st.write("Was this answer helpful?")
        col_yes, col_no, _ = st.columns([1, 1, 5])
        with col_yes:
            st.markdown('<div class="feedback-btn">', unsafe_allow_html=True)
            st.button("Yes", key="feedback_yes", on_click=set_feedback)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_no:
            st.markdown('<div class="feedback-btn">', unsafe_allow_html=True)
            st.button("No", key="feedback_no", on_click=set_feedback)
            st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────  Main flow  ────────────────────────────────
initialize_state()
render_header()
render_apple_style_input_area()

if st.session_state.user_query and st.session_state.user_query != st.session_state.active_starter:
    handle_query(st.session_state.user_query, from_starter=False)
    st.session_state.user_query = ""

if st.session_state.response:
    render_response_area()