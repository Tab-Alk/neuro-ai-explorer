import streamlit as st
from core_engine import query_rag, generate_related_questions, get_embedding_function
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="The Neural Intelligence Lab",
    layout="wide" # Use the full width of the page
)

# --- App State Management ---
def initialize_state():
    """Initializes session state variables."""
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'related_questions' not in st.session_state:
        st.session_state.related_questions = []
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

# --- Core Logic Functions (No changes from original) ---
def handle_query(query):
    """Handles the query submission, RAG retrieval, and question generation."""
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it to continue.")
        return

    # Clear previous results to make way for the new ones
    st.session_state.response = None
    st.session_state.related_questions = []

    with st.spinner("Synthesizing answer..."):
        answer, sources = query_rag(query, api_key=groq_api_key)
        st.session_state.response = { "query": query, "answer": answer, "sources": sources }
    with st.spinner("Generating related questions..."):
        st.session_state.related_questions = generate_related_questions(query, answer, api_key=groq_api_key)


# --- UI Rendering Functions (REFACTORED FOR BETTER LAYOUT) ---
def render_header():
    """Renders the main title and introductory text of the app."""
    with st.container():
        st.title("The Neural Intelligence Lab")
        st.write("Ask a question about the fascinating parallels and differences between biological brains and artificial intelligence.")
        st.write("Example: *How does memory in an AI compare to a human brain?*")
        st.write("") # Strategic whitespace

def render_input_area():
    """Renders the user input text box."""
    with st.container():
        # The on_change callback updates the session state when the user presses Enter or clicks away.
        def set_query_from_input():
            st.session_state.user_query = st.session_state.input_query

        st.text_input(
            "Your Question:", 
            key="input_query",
            on_change=set_query_from_input,
            placeholder="Type your question here and press Enter..."
        )

def render_response_area():
    """Renders the entire response section including answer, sources, and feedback."""
    response_data = st.session_state.response
    
    with st.container():
        st.write("") # Whitespace for separation
        st.subheader("Answer")
        st.write(response_data["answer"])

        # Display related questions if they exist
        if st.session_state.related_questions:
            st.write("") # Whitespace
            with st.expander("Explore Related Concepts", expanded=True):
                # Using columns for a cleaner, more organized button layout
                cols = st.columns(3)
                questions_to_display = st.session_state.related_questions[:3] 
                for i, q in enumerate(questions_to_display):
                    with cols[i]:
                        if st.button(q, key=f"related_q_{q}", use_container_width=True):
                            st.session_state.input_query = q # Pre-fill the input box
                            st.session_state.user_query = q # Set query to be processed
                            st.rerun() 
        
        st.markdown("---")
        
        # Display sources in a collapsible expander to keep the UI clean
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

# Render the static header and input areas
render_header()
render_input_area()

# Check if a new query has been submitted via the input's on_change callback
if st.session_state.user_query:
    query_to_process = st.session_state.user_query
    # IMPORTANT: Clear the user_query state immediately after capturing it.
    # This prevents the query from being re-processed on subsequent interactions (e.g., clicking a button).
    st.session_state.user_query = "" 
    handle_query(query_to_process)

# Render the response area only if a response exists in the session state
if st.session_state.response:
    render_response_area()

