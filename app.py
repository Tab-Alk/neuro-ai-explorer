import streamlit as st
from core_engine import query_rag, generate_related_questions
import time
import nltk
from thefuzz import fuzz
import os # Import the 'os' module for path operations

# --- App State Management ---
def initialize_state():
    """Initializes the session state for the app."""
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'related_questions' not in st.session_state:
        st.session_state.related_questions = []

# --- Text Processing and Highlighting ---
def highlight_text(source_text, generated_answer, threshold=85):
    """Highlights sentences in source_text that are similar to sentences in generated_answer."""
    # --- NEW: Manually load the tokenizer to bypass NLTK's failing search function ---
    try:
        # Construct the absolute path to the punkt tokenizer pickle file
        APP_ROOT = os.path.dirname(os.path.abspath(__file__))
        PUNKT_PATH = os.path.join(APP_ROOT, "nltk_data", "tokenizers", "punkt", "english.pickle")
        
        # Load the tokenizer manually from the pickle file
        tokenizer = nltk.data.load(PUNKT_PATH)
        
        # Use the tokenizer's .tokenize() method directly
        source_sentences = tokenizer.tokenize(source_text)
        answer_sentences = tokenizer.tokenize(generated_answer)
    except Exception as e:
        # If loading fails for any reason, fall back gracefully
        st.error(f"Error during sentence tokenization: {e}")
        return source_text
    # --- END NEW SECTION ---

    highlighted_sentences = set()
    for a_sent in answer_sentences:
        for s_sent in source_sentences:
            if fuzz.token_set_ratio(a_sent, s_sent) > threshold:
                highlighted_sentences.add(s_sent)

    final_text = ""
    for s_sent in source_sentences:
        if s_sent in highlighted_sentences:
            final_text += f"<mark style='background-color: yellow;'>{s_sent}</mark> "
        else:
            final_text += f"{s_sent} "
            
    return final_text.strip()


# --- UI Rendering Functions ---
# (The rest of the file remains the same)
def display_header():
    st.title("The Neural Intelligence Lab")
    st.write("Ask a question about the fascinating parallels and differences between biological brains and artificial intelligence.")
    st.write("Example: *How does memory in an AI compare to a human brain?*")

def display_response_area():
    response_data = st.session_state.response
    st.subheader("Answer:")
    st.write(response_data["answer"])

    if st.session_state.related_questions:
        with st.expander("Explore Related Concepts"):
            for q in st.session_state.related_questions:
                if st.button(q, key=f"related_q_{q}"):
                    st.session_state.user_query = q 
                    st.rerun() 
    
    st.markdown("---")
    st.subheader("Sources (with highlighting):")
    full_source_text = "\n\n".join([doc.page_content for doc in response_data["sources"]])
    highlighted_source = highlight_text(full_source_text, response_data["answer"])
    with st.expander("View Highlighted Source Text"):
        st.markdown(highlighted_source, unsafe_allow_html=True)
        
    st.markdown("---")
    st.write("Was this answer helpful? (Your feedback is not saved).")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Yes", use_container_width=True)
    with col2:
        st.button("No", use_container_width=True)


# --- Core Logic Functions ---
def handle_query(query):
    with st.spinner("Synthesizing answer..."):
        answer, sources = query_rag(query)
        st.session_state.response = {"query": query, "answer": answer, "sources": sources}
    with st.spinner("Generating related questions..."):
        st.session_state.related_questions = generate_related_questions(query, answer)

# --- Main Application Execution ---
st.set_page_config(page_title="The Neural Intelligence Lab")
initialize_state()
display_header()

if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

def set_query_from_input():
    st.session_state.user_query = st.session_state.input_query

st.text_input(
    "Your Question:", 
    key="input_query",
    on_change=set_query_from_input,
    placeholder="Type your question here..."
)

if st.session_state.user_query:
    handle_query(st.session_state.user_query)
    st.session_state.user_query = "" 

if st.session_state.response:
    display_response_area()