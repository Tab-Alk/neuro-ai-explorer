import streamlit as st
from core_engine import query_rag
import time

# --- App State Management ---
# This initializes the session state to store our app's data.
def initialize_state():
    """Initializes the session state for the app."""
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = False

# --- UI Rendering Functions ---

def display_header():
    """Displays the header and introduction of the app."""
    st.title("🧠 Neuro-AI Explorer 🤖")
    st.write(
        "Ask a question about the fascinating parallels and differences "
        "between biological brains and artificial intelligence."
    )
    st.write(
        "Example: *How does memory in an AI compare to a human brain?*"
    )

def display_response_and_feedback():
    """Displays the generated answer, sources, and feedback buttons."""
    response_data = st.session_state.response
    
    st.subheader("Answer:")
    st.write(response_data["answer"])
    
    st.markdown("---")
    st.subheader("Sources:")

    # --- NEW: Clickable, expandable sources ---
    # Get unique sources to avoid repetition
    unique_sources = {source.metadata['source'] for source in response_data["sources"]}

    for source_path in unique_sources:
        # Use an expander for each source document
        with st.expander(f"**Source: `{source_path}`**"):
            try:
                # Read and display the full content of the source file
                with open(source_path, 'r', encoding='utf-8') as f:
                    st.text(f.read())
            except FileNotFoundError:
                st.error(f"Error: Source file not found at {source_path}")
            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")
    # --- END NEW SECTION ---

    # Display feedback buttons if feedback has not been given yet
    if not st.session_state.feedback_given:
        st.markdown("---")
        st.write("Was this answer helpful?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes", use_container_width=True):
                handle_feedback(positive=True)
        with col2:
            if st.button("👎 No", use_container_width=True):
                handle_feedback(positive=False)

# --- Core Logic Functions ---

def generate_response(query):
    """Generates a response using the RAG engine and updates the session state."""
    with st.spinner("Synthesizing answer..."):
        st.session_state.feedback_given = False
        answer, sources = query_rag(query)
        st.session_state.response = {
            "query": query,
            "answer": answer,
            "sources": sources
        }

def handle_feedback(positive: bool):
    """Handles the user's feedback, logs it, and updates the UI."""
    st.session_state.feedback_given = True
    response_data = st.session_state.response
    feedback_type = "POSITIVE" if positive else "NEGATIVE"
    
    print(f"FEEDBACK: {feedback_type} for query: '{response_data['query']}'")
    
    if positive:
        st.success("Thank you for your feedback!")
    else:
        st.warning("Thank you for your feedback! We'll use it to improve.")
        
    time.sleep(2)
    st.rerun()

# --- Main Application Execution ---

st.set_page_config(page_title="Neuro-AI Explorer", page_icon="🤖")

# Initialize the app's state
initialize_state()

# Display the main header
display_header()

# Get user input
user_query = st.text_input(
    "Your Question:", 
    placeholder="Type your question here..."
)

# Handle the "Get Answer" button click
if st.button("Get Answer"):
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        generate_response(user_query)

# Display the response area if a response exists in the state
if st.session_state.response:
    display_response_and_feedback()