import streamlit as st
from core_engine import query_rag, generate_related_questions
import time
import re # Import the regular expressions library
from thefuzz import fuzz

# --- App State Management ---
def initialize_state():
    """Initializes the session state for the app."""
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'related_questions' not in st.session_state:
        st.session_state.related_questions = []

# --- Text Processing and Highlighting ---

def sent_tokenize_regex(text: str) -> list[str]:
    """A simple, self-contained sentence tokenizer using regular expressions."""
    # This regex splits the text at any occurrence of . ! ? followed by a space.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # It returns a list of sentences, removing any empty strings that might result.
    return [s.strip() for s in sentences if s.strip()]

def highlight_text(source_text, generated_answer, threshold=85):
    """Highlights sentences in source_text that are similar to sentences in generated_answer."""
    # We now use our own self-contained sentence tokenizer.
    source_sentences = sent_tokenize_regex(source_text)
    answer_sentences = sent_tokenize_regex(generated_answer)
    
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
    """Displays the generated answer, sources, related questions, and feedback."""
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