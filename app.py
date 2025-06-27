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
    /* Compact styling for related questions and feedback buttons */
    .feedback-btn button, .related-q-btn button {
        padding: 6px 12px !important;
        font-size: 0.9rem !important;
        border-radius: 6px !important;
        height: auto !important;
        width: 100% !important;
        min-height: 0 !important;
        line-height: 1.2 !important;
        white-space: normal !important;
        margin: 2px 0 !important;
        border: 1px solid #D0D0D0 !important;
        background: #FFFFFF !important;
        color: #1D1D1F !important;
        transition: 0.2s !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    .feedback-btn button:hover, .related-q-btn button:hover {
        border-color: #007aff !important;
        color: #007aff !important;
        background: #f7f9fc !important;
    }
    
    /* Remove extra spacing from button containers */
    .feedback-btn > div, .related-q-btn > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .feedback-btn, .related-q-btn {
        margin: 4px 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────  App configuration  ─────────────────────────────
st.set_page_config(
    page_title="The Neural Intelligence Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add title and description at the top of the main script
st.title("The Neural Intelligence Lab")
st.markdown(
    "Compare how biological brains and artificial intelligence actually work. "
    "Ask questions about neurons, neural networks, learning, memory, or decision‑making. "
    "Get answers that explore both worlds of intelligence."
)
st.markdown("<br>", unsafe_allow_html=True)

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
        text = ""

        # Handle different document formats
        if hasattr(doc, "page_content"):
            # LangChain Document object
            text = doc.page_content
            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                heading = doc.metadata.get("heading", "")
        elif isinstance(doc, dict):
            # Plain dict format
            text = doc.get("page_content", "") or doc.get("text", "")
            metadata = doc.get("metadata", {})
            if isinstance(metadata, dict):
                heading = metadata.get("heading", "")
            else:
                # Sometimes metadata might be stored directly in the doc
                heading = doc.get("heading", "")

        # Fallback to first sentence if no heading found
        if not heading and text:
            first_sentence = text.split(".")[0][:80]
            heading = first_sentence + "…" if len(first_sentence) == 80 else first_sentence
        elif not heading:
            heading = "Untitled"

        # Display the source with proper heading
        st.markdown(f"**Excerpt {idx}: {heading}**")
        
        # Highlight the text relative to the answer
        if text:
            highlighted = highlight_text(text, answer)
            st.markdown(highlighted, unsafe_allow_html=True)
        else:
            st.markdown("*No content available*")
            
        # Add separator between sources (except for the last one)
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
    """Layered sidebar introduction."""
    with st.sidebar:
        st.markdown("### What is this App?")
        st.markdown("---")

        # Executive Summary
        st.write(
            "Welcome to the **Neural Intelligence Lab**! This interactive web application "
            "allows you to explore the fascinating connections and distinctions between "
            "**biological brains** and **artificial intelligence**. Ask questions, "
            "receive AI-generated answers, and discover new concepts in this dynamic "
            "knowledge platform."
        )
        st.write(
            "It's designed for anyone curious about the cutting edge of AI and "
            "neuroscience, offering a transparent 'glass box' approach to understanding "
            "where information comes from."
        )
        st.markdown("---")

        # Technical Details (now a regular sidebar block)
        st.markdown("### Technical Details:")
        st.markdown(
            "This application is powered by a modern **Retrieval-Augmented Generation (RAG) pipeline** "
            "designed for explainable and high-quality knowledge discovery. Here’s a quick overview of its core components:"
        )
        st.markdown("""
        * **Brain (LLM):** Leverages **Llama 3** via **Groq** for high-speed, intelligent answer generation.  
        * **Memory (Vector DB):** A curated knowledge base is vectorized using **Hugging Face's `all-MiniLM-L6-v2`** embeddings and stored in **ChromaDB**.  
        * **Reasoning (LangChain):** Orchestrates the entire RAG workflow, from retrieval to answer synthesis.  
        * **Transparency (Semantic Highlighting):** Employs **scikit-learn** for cosine similarity to visually highlight exact source sentences that inform the AI's answer.  
        * **Quality Assurance (Ragas):** Includes a quantitative evaluation framework using the **Ragas** library to measure answer faithfulness, relevance, and context utilization.
        """)
        st.markdown("This ensures not only accurate answers but also a clear understanding of their provenance.")

        st.markdown("---")

        # Creator + Links
        st.markdown("### Created By: Your Name")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourname)")
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourname)")
        st.markdown("[![Project Repo](https://img.shields.io/badge/Project%20Repo-purple?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourname/neural-intelligence-lab)")

        st.markdown("---")
        st.caption("© 2025 Neural Intelligence Lab. All rights reserved.")
        st.caption("Version 1.0")

# Remove the logo image if present (e.g. st.image("static/logo.png", width=120))


def render_apple_style_input_area() -> None:
    STARTER_QUESTIONS = [
        "Why can deep learning excel at pattern recognition yet still struggle with the common‑sense reasoning that comes naturally to humans?",
        "How does the brain consolidate memories during sleep, and how could replay‑style mechanisms inspire more robust continual‑learning in AI?",
        "What lessons from human attention can help us design faster, energy‑efficient AI models that run directly on edge devices?",
    ]

    st.markdown(
        """
        <style>
        /* --- Apple‑style pill button for starter questions only --- */
        .starter-btn div.stButton > button:first-child {
            background: #ECEFF1 !important;  /* more contrasting light gray-blue */
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
        .starter-btn div.stButton > button:first-child:hover{
            border-color:#007aff;
            color:#007aff;
            background:#f7f9fc !important;
        }
        .starter-btn div.stButton > button:first-child:focus{
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
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="starter-area">', unsafe_allow_html=True)
        cols = st.columns([1, 1, 1], gap="medium")
        for i, q in enumerate(STARTER_QUESTIONS):
            with cols[i]:
                st.markdown('<div class="starter-btn">', unsafe_allow_html=True)
                if st.button(q, key=f"starter_{i}", use_container_width=True):
                    st.session_state.user_query = q
                    handle_query(q, from_starter=True)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
            <style>
            .starter-area div.stButton > button:first-child {
                min-height: 140px !important;
                padding: 48px 56px !important;
                background: #ECEFF1 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown(
            "<h4 style='text-align:center;color:#1D1D1F;margin-bottom:16px;"
            "font-size:2rem;font-weight:700'>"
            "Ask another question</h4>",
            unsafe_allow_html=True,
        )

        col_input, col_button = st.columns([4, 1])
        with col_input:
            def set_query_from_input():
                st.session_state.user_query = st.session_state.input_query.strip()
            st.text_input(
                "Ask your question",
                key="input_query",
                placeholder="Type here…",
                on_change=set_query_from_input,
                label_visibility="collapsed",
            )
        with col_button:
            st.write("")  # spacer
            if st.button("Clear"):
                st.session_state.input_query = ""
                st.session_state.user_query = ""


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


# ────────────────────────────────  Main flow  ────────────────────────────────
initialize_state()
render_header()
st.markdown("<br>", unsafe_allow_html=True)
render_apple_style_input_area()

if st.session_state.user_query and st.session_state.user_query != st.session_state.active_starter:
    handle_query(st.session_state.user_query, from_starter=False)
    st.session_state.user_query = ""

if st.session_state.response:
    render_response_area()