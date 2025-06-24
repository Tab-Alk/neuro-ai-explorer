import streamlit as st
from core_engine import query_rag, get_embedding_function
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="The Neural Intelligence Lab",
    layout="wide" # Use the full width of the page
)

# --- Curated Guided Tour Questions ---
GUIDED_QUESTIONS = [
    "How does human memory compare to AI memory storage?",
    "How does learning happen in brains vs. AI?",
    "Why can AI beat humans at chess but struggle with common sense?",
    "Could we download our memories into AI?"
]

# --- App State Management ---
def initialize_state():
    """Initializes session state variables."""
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ""

# --- Text Processing and Highlighting (No changes from original) ---
def sent_tokenize_regex(text: str) -> list[str]:
    """Splits text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def highlight_text(source_text, generated_answer, threshold=0.70):
    """Highlights source sentences with high semantic similarity to answer sentences."""
    embedding_function = get_embedding_function()
    
    source_sentences = sent_tokenize_regex(source_text)
    answer_sentences = sent_tokenize_regex(generated_answer)

    if not source_sentences or not answer_sentences:
        return source_text

    source_embeddings = embedding_function.embed_documents(source_sentences)
    answer_embeddings = embedding_function.embed_documents(answer_sentences)

    similarity_matrix = cosine_similarity(answer_embeddings, source_embeddings)
    
    highlighted_sentences = set()
    for i in range(len(answer_sentences)):
        best_match_index = np.argmax(similarity_matrix[i])
        if similarity_matrix[i][best_match_index] > threshold:
            highlighted_sentences.add(source_sentences[best_match_index])

    final_text = ""
    for s_sent in source_sentences:
        if s_sent in highlighted_sentences:
            final_text += f"<mark style='background-color: yellow;'>{s_sent}</mark> "
        else:
            final_text += f"{s_sent} "
            
    return final_text.strip()

# --- Core Logic Functions (UPDATED) ---
def handle_query(query):
    """Handles the query submission and RAG retrieval."""
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it to continue.")
        return

    # Clear previous response
    st.session_state.response = None
    
    with st.spinner("Synthesizing answer..."):
        answer, sources = query_rag(query, api_key=groq_api_key)
        # Store the new response
        st.session_state.response = { "query": query, "answer": answer, "sources": sources }


# --- UI Rendering Functions (REFACTORED FOR NEW QUESTION FLOW) ---
def render_header():
    """Renders the main title and introductory text of the app."""
    with st.container():
        st.title("The Neural Intelligence Lab")
        st.write("Ask a question about the fascinating parallels and differences between biological brains and artificial intelligence.")
        # Use the first guided question as the new example
        st.write(f"Example: *{GUIDED_QUESTIONS[0]}*")
        st.write("") # Strategic whitespace

def render_input_area():
    """Renders the user input text box."""
    with st.container():
        def set_query_from_input():
            st.session_state.user_query = st.session_state.input_query

        st.text_input(
            "Your Question:", 
            key="input_query",
            on_change=set_query_from_input,
            placeholder="Type your question here and press Enter..."
        )

def render_guided_tour():
    """Renders the curated list of guided questions as buttons."""
    with st.container():
        st.write("")
        st.subheader("Or, take a guided tour:")
        
        # Use columns for a cleaner, more organized button layout
        cols = st.columns(2) 
        for i, q in enumerate(GUIDED_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"guided_q_{q}", use_container_width=True):
                    st.session_state.input_query = q # Pre-fill the input box
                    st.session_state.user_query = q # Set query to be processed
                    st.rerun()

def render_response_area():
    """Renders the entire response section including answer, sources, and feedback."""
    response_data = st.session_state.response
    
    with st.container():
        st.write("") # Whitespace for separation
        st.subheader("Answer")
        st.write(response_data["answer"])
        
        st.markdown("---")
        
        # Display sources in a collapsible expander
        st.subheader("Sources")
        full_source_text = "\n\n".join([doc.page_content for doc in response_data["sources"]])
        highlighted_source = highlight_text(full_source_text, response_data["answer"])
        with st.expander("View Highlighted Source Text"):
            st.markdown(highlighted_source, unsafe_allow_html=True)
            
        st.write("") # Whitespace
        st.markdown("---")

        # Display feedback section
        st.write("Was this answer helpful? (Your feedback is not saved).")
        col1, col2, _ = st.columns([1, 1, 5]) # Use columns to control button width
        with col1:
            st.button("Yes", use_container_width=True)
        with col2:
            st.button("No", use_container_width=True)


# --- Main Application Execution ---
initialize_state()

# Render the static UI components
render_header()
render_input_area()

# Render the guided tour questions if no query has been made yet
if not st.session_state.response:
    render_guided_tour()

# Check if a new query has been submitted
if st.session_state.user_query:
    query_to_process = st.session_state.user_query
    st.session_state.user_query = "" 
    handle_query(query_to_process)

# Render the response area if a response exists in the session state
if st.session_state.response:
    render_response_area()

