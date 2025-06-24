import streamlit as st
from core_engine import query_rag, generate_related_questions, get_embedding_function
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# --- App State Management (Original) ---
def initialize_state():
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'related_questions' not in st.session_state:
        st.session_state.related_questions = []


# --- Text Processing and Highlighting (Original) ---
def sent_tokenize_regex(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def highlight_text(source_text, generated_answer, threshold=0.70):
    """Highlights source sentences with high semantic similarity to answer sentences."""
    embedding_function = get_embedding_function()

    source_sentences = sent_tokenize_regex(source_text)
    answer_sentences = sent_tokenize_regex(generated_answer)

    if not source_sentences or not answer_sentences:
        return source_text

    # Generate embeddings for all sentences
    source_embeddings = embedding_function.embed_documents(source_sentences)
    answer_embeddings = embedding_function.embed_documents(answer_sentences)

    # Calculate cosine similarity between each answer sentence and all source sentences
    similarity_matrix = cosine_similarity(answer_embeddings, source_embeddings)

    highlighted_sentences = set()
    for i in range(len(answer_sentences)):
        # Find the source sentence with the highest similarity for the current answer sentence
        best_match_index = np.argmax(similarity_matrix[i])
        if similarity_matrix[i][best_match_index] > threshold:
            highlighted_sentences.add(source_sentences[best_match_index])

    # Reconstruct the text with highlighting
    final_text = ""
    for s_sent in source_sentences:
        if s_sent in highlighted_sentences:
            final_text += f"<mark style='background-color: yellow;'>{s_sent}</mark> "
        else:
            final_text += f"{s_sent} "

    return final_text.strip()


# --- UI Rendering Functions (Cosmetic Changes Only) ---
def render_related_questions(questions: list[str], cols_per_row: int = 3):
    """
    Display the list of related‑question buttons in a responsive horizontal grid.
    Returns the question that was clicked, or None if no button was pressed.
    """
    clicked = None
    for start in range(0, len(questions), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, q in enumerate(questions[start:start + cols_per_row]):
            with cols[idx]:
                if st.button(q, use_container_width=True, key=f"related_q_{q}"):
                    clicked = q
    st.write("")  # small spacer below the grid
    return clicked

def display_header():
    st.title("The Neural Intelligence Lab")
    st.write(
        "Ask a question about the fascinating parallels and differences between "
        "biological brains and artificial intelligence."
    )
    # Changed to markdown for bolding
    st.markdown("**Example:** *How does memory in an AI compare to a human brain?*")
    # Added whitespace
    st.write("")

def display_response_area():
    response_data = st.session_state.response

    # Changed to header for better hierarchy
    st.header("Answer")
    st.write(response_data["answer"])
    
    # Added whitespace for better separation
    st.write("")

    # THIS SECTION IS IDENTICAL TO YOUR ORIGINAL CODE
    if st.session_state.related_questions:
        with st.expander("Explore Related Concepts", expanded=True):
            selected_q = render_related_questions(
                st.session_state.related_questions,
                cols_per_row=3
            )
            if selected_q:
                st.session_state.user_query = selected_q
                st.rerun()

    st.markdown("---")
    # Changed to subheader
    st.subheader("Sources")
    full_source_text = "\n\n".join([doc.page_content for doc in response_data["sources"]])
    highlighted_source = highlight_text(full_source_text, response_data["answer"])
    with st.expander("View Highlighted Source Text"):
        st.markdown(highlighted_source, unsafe_allow_html=True)

    st.markdown("---")
    st.write("Was this answer helpful? (Your feedback is not saved).")
    col1, col2, _ = st.columns([1, 1, 5]) # Adjusted columns for better spacing
    with col1:
        st.button("Yes", use_container_width=True)
    with col2:
        st.button("No", use_container_width=True)


# --- Core Logic Functions (Original) ---
def handle_query(query):
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it.")
        return

    with st.spinner("Synthesizing answer..."):
        answer, sources = query_rag(query, api_key=groq_api_key)
        st.session_state.response = {"query": query, "answer": answer, "sources": sources}

    with st.spinner("Generating related questions..."):
        st.session_state.related_questions = generate_related_questions(
            query, answer, api_key=groq_api_key
        )


# --- Main Application Execution (Minimal Changes for Layout) ---
st.set_page_config(page_title="The Neural Intelligence Lab", layout="wide")
initialize_state()
display_header()

# Added a header for the input section
st.header("Ask a Question")

if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

def set_query_from_input():
    st.session_state.user_query = st.session_state.input_query

st.text_input(
    "Your Question:",
    key="input_query",
    on_change=set_query_from_input,
    placeholder="Type your question here...",
    label_visibility="collapsed" # Hide label since we have a header
)

# Added a separator
st.markdown("---")

# This logic is identical to your original code
if st.session_state.user_query:
    handle_query(st.session_state.user_query)
    st.session_state.user_query = ""

if st.session_state.response:
    display_response_area()
