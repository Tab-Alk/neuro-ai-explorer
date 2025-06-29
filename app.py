import streamlit as st
from core_engine import (
    query_rag,
    generate_related_questions,
    get_embedding_function,
)
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ──────────────────────────── App configuration & GLOBAL STYLES ─────────────────────────────

st.set_page_config(
    page_title="The Neural Intelligence Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UNIFIED CSS BLOCK FOR ALL SPACING FIXES ---
# This single block now controls the layout for the entire app.
st.markdown("""
<style>
/* --- 1. GLOBAL LAYOUT & SPACING RESETS --- */

/* Target the main container for more precise control */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* Remove default Streamlit spacing between elements in the main area */
.main [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div[data-testid^="stMarkdownContainer"] {
    margin-bottom: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

/* Main page title */
h1 {
    margin-bottom: 0.25rem !important;
}

/* Main page subtitle */
.main [data-testid="stMarkdownContainer"] p {
    font-size: 1.1rem;
    line-height: 1.6;
    margin-bottom: 1.5rem !important; /* Controlled space between subtitle and first button */
}

/* --- 2. SIDEBAR SPACING FIXES --- */

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    margin-bottom: 0.25rem !important; /* Space after sidebar titles */
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] hr {
    margin: 0.75rem 0 !important; /* Tighter horizontal lines */
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    line-height: 1.4 !important; /* Tighter line spacing for text */
    font-size: 0.95rem;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    padding: 0 !important;
    margin-bottom: 0.75rem !important; /* Space after each markdown block */
}

/* --- 3. STARTER QUESTION BUTTONS & SECTION --- */

/* This container controls the space BETWEEN buttons */
.starter-btn-container {
    margin-bottom: 0.6rem !important;
}

.starter-btn-container div.stButton > button:first-child {
    background: #FFFFFF !important;
    border: 1px solid #D0D0D0 !important;
    border-radius: 0.5rem !important;
    padding: 1rem 1.25rem !important;
    font: 500 1rem -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    color: #1D1D1F !important;
    text-align: left !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
    margin: 0 !important;
    height: auto !important;
    white-space: normal !important;
    width: 100% !important;
}

.starter-btn-container div.stButton > button:first-child:hover {
    border-color: #007aff !important;
    color: #007aff !important;
    background: #f7f9fc !important;
}

/* --- 4. "ASK ANOTHER QUESTION" SECTION --- */

.question-input-section {
    padding-top: 2.5rem !important; /* Controls space above the heading */
}

.question-input-section h4 {
    text-align: center;
    color: #1D1D1F;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem !important; /* Space between heading and input box */
}

div[data-testid="stTextInput"] input {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    font-size: 1.05rem;
}

/* --- 5. VERTICAL SPACER FIX --- */

/* This empty div will expand to push all content up, solving the vertical alignment issue. */
.main > div {
    display: flex;
    flex-direction: column;
}
.spacer {
    flex-grow: 1;
}

/* --- App background colors --- */
section[data-testid="stSidebar"] {
    background-color: #F0EEEB !important;
}
.main {
    background-color: #FBF8F6;
}

</style>
""", unsafe_allow_html=True)


# Add title and description at the top of the main script
st.title("The Neural Intelligence Lab")
st.markdown(
    "Compare how biological brains and artificial intelligence actually work. "
    "Ask questions about neurons, neural networks, learning, memory, or decision‑making. "
    "Get answers that explore both worlds of intelligence."
)

# ─────────────────────────────  State management & Helpers (NO CHANGES) ─────────────────────────────
def initialize_state():
    # ... (code is correct)
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


def sent_tokenize_regex(text: str) -> list[str]:
    # ... (code is correct)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def highlight_text(source_text: str, generated_answer: str, threshold: float = 0.70) -> str:
    # ... (code is correct)
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


def render_sources(retrieved_docs: list, answer: str):
    # ... (code is correct)
    if not retrieved_docs: return
    for idx, doc in enumerate(retrieved_docs, start=1):
        text = doc.page_content
        heading = doc.metadata.get("heading", "")
        if not heading and text:
            first_sentence = text.split(".")[0][:80]
            heading = first_sentence + "…" if len(first_sentence) == 80 else first_sentence
        elif not heading:
            heading = "Untitled"
        st.markdown(f"**Excerpt {idx}: {heading}**")
        if text:
            highlighted = highlight_text(text, answer)
            st.markdown(highlighted, unsafe_allow_html=True)
        else:
            st.markdown("*No content available*")
        if idx < len(retrieved_docs): st.markdown("---")


def handle_query(query: str, from_starter: bool = False):
    # ... (code is correct)
    st.session_state.feedback_given = False
    st.session_state.active_starter = query if from_starter else ""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("API Key is missing. Please set it in your environment or Streamlit secrets.")
        return
    with st.spinner("Synthesizing answer…"):
        answer, sources = query_rag(query, api_key=api_key)
        st.session_state.response = {"query": query, "answer": answer, "sources": sources}
    with st.spinner("Generating related questions…"):
        st.session_state.related_questions = generate_related_questions(query, answer, api_key=api_key)

# ───────────────────────────────  UI builders (REVISED)  ────────────────────────────────

def render_header() -> None:
    """Layered sidebar introduction."""
    with st.sidebar:
        st.markdown("### What is this App?")
        st.write(
            "The Neural Intelligence Lab is a specialized AI-powered research tool designed to accelerate literature reviews and comparative analyses in neuroscience and artificial intelligence."
        )
        st.write(
            "It solves the challenge of navigating vast scientific data by intelligently parsing and synthesizing diverse research, delivering accurate, source-backed answers."
        )
        st.write(
            "Every response features highlighted sources and intelligent follow-up questions, providing transparent, verifiable, and guided research sessions."
        )
        st.markdown("**Ready to begin?**  \nStart by clicking a starter question or asking your own!", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Technical Implementation")
        st.markdown("""
        * **Multi-Format Document Ingestion (LlamaParse):** Robustly parses complex PDFs and other research documents, expanding the knowledge base for comprehensive insights.  
        * **Intelligent RAG Pipeline (LangChain, Groq LLMs):** Combines advanced retrieval (ChromaDB) with powerful Llama 3 models for precise, hallucination-free answers.  
        * **Explainability & Discovery:** Provides semantic source highlighting and AI-generated follow-up questions for transparent and guided exploration.  
        * **Quantitative Evaluation (Ragas):** Ensures continuous quality assurance of answer faithfulness and relevance.
        """)
        st.markdown("---")
        st.markdown("### Created By: Tabarek Alkhalidi")
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Tab--Alk-black?style=for-the-badge&logo=github)](https://github.com/Tab-Alk)"
        )
        st.markdown(
            "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tabarek%20Alkhalidi-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/tabarek-alkhalidi/)"
        )
        st.markdown(
            "[![Project Repo](https://img.shields.io/badge/Project%20Repo-Neural%20Intel%20Lab-purple?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Tab-Alk/Neural-Intel-Lab)"
        )
        st.markdown("---")
        st.caption("© 2025 The Neural Intelligence Lab. V2.0")


def render_apple_style_input_area() -> None:
    STARTER_QUESTIONS = [
        "Why can deep learning excel at pattern recognition yet still struggle with the common‑sense reasoning that comes naturally to humans?",
        "How does the brain consolidate memories during sleep, and how could replay‑style mechanisms inspire more robust continual‑learning in AI?",
        "What lessons from human attention can help us design faster, energy‑efficient AI models that run directly on edge devices?",
    ]

    # --- Vertical layout for starter questions ---
    for q in STARTER_QUESTIONS:
        # We now wrap each button in a div that our CSS can target for spacing.
        st.markdown('<div class="starter-btn-container">', unsafe_allow_html=True)
        if st.button(q, key=f"starter_{q}", use_container_width=True):
            st.session_state.user_query = q
            handle_query(q, from_starter=True)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- "Ask another question" section ---
    st.markdown('<div class="question-input-section">', unsafe_allow_html=True)
    # The h4 style is now controlled by the global CSS, not inline styles.
    st.markdown("<h4>Ask another question</h4>", unsafe_allow_html=True)

    def set_query_from_input():
        st.session_state.user_query = st.session_state.input_query.strip()
    st.text_input(
        "Ask your question", key="input_query", placeholder="Type here…",
        on_change=set_query_from_input, label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)


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
    with st.container():
        st.markdown(
            """
            <div style="
                border: 1px solid rgba(0, 0, 0, 0.1);
                background-color: #ffffff;
                border-radius: 0.5rem;
                padding: 1.2rem;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
                font-size: 1.1rem;
                line-height: 1.6;
                color: #1D1D1F;
            ">
            """ + resp["answer"] + "</div>",
            unsafe_allow_html=True
        )
    st.write("")

    # Related concepts expander (collapsed by default)
    if st.session_state.related_questions:
        with st.expander("Related Questions to Explore", expanded=False):
            for i, q in enumerate(st.session_state.related_questions):
                st.markdown('<div class="related-q-btn">', unsafe_allow_html=True)
                if st.button(q, key=f"rel_q_{i}_{hash(q)}", use_container_width=True):
                    st.session_state.user_query = q
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # Sources section
    st.subheader("Sources")
    with st.expander("View Retrieved Sources"):
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
            st.button("Yes", key="feedback_yes", on_click=set_feedback, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_no:
            st.markdown('<div class="feedback-btn">', unsafe_allow_html=True)
            st.button("No", key="feedback_no", on_click=set_feedback, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────  Main flow (REVISED) ────────────────────────────────
initialize_state()
render_header()
render_apple_style_input_area()

# This is the key fix for vertical alignment. This spacer div grows to fill
# all available space, pushing the content above it to the top.
if not st.session_state.response:
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

if st.session_state.user_query and st.session_state.user_query != st.session_state.active_starter:
    handle_query(st.session_state.user_query, from_starter=False)
    st.session_state.user_query = ""

if st.session_state.response:
    render_response_area()