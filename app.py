import streamlit as st
from core_engine import query_rag
import time

# --- App UI ---

# Set the page title
st.set_page_config(page_title="Neuro-AI Explorer", page_icon="🤖")

# Display the title and a short introduction
st.title("Neuro-AI Explorer ")
st.write(
    "Ask a question about the fascinating parallels and differences "
    "between biological brains and artificial intelligence."
)
st.write(
    "Example: *How does memory in an AI compare to a human brain?*"
)

# --- Session State Management ---
# Initialize session state for holding the response and feedback
if 'response' not in st.session_state:
    st.session_state.response = None
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

# --- User Input and Response Generation ---
user_query = st.text_input(
    "Your Question:", 
    placeholder="Type your question here..."
)

if st.button("Get Answer"):
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Searching the knowledge base..."):
            # When a new question is asked, reset feedback state
            st.session_state.feedback_given = False
            # Call the RAG engine
            context, sources = query_rag(user_query)
            # Store the response in session state
            st.session_state.response = {
                "query": user_query,
                "context": context,
                "sources": sources
            }

# --- Display Response and Feedback Buttons ---
if st.session_state.response:
    response_data = st.session_state.response
    
    st.subheader("Retrieved Information:")
    st.info(response_data["context"])
    
    # Display sources
    st.markdown("---")
    st.subheader("Sources:")
    for i, source in enumerate(response_data["sources"]):
        st.write(f"**Source {i+1}:** `{source[0].metadata['source']}`")
        st.write(f"**Relevance Score:** `{source[1]:.2f}`")

    # --- Feedback Logic ---
    if not st.session_state.feedback_given:
        st.markdown("---")
        st.write("Was this answer helpful?")
        
        # Use columns for side-by-side buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Yes", use_container_width=True):
                st.session_state.feedback_given = True
                # In a real app, you would log this to a database
                print(f"FEEDBACK: POSITIVE for query: '{response_data['query']}'")
                st.success("Thank you for your feedback!")
                time.sleep(2) # Keep the message on screen for a moment
                st.rerun() # Rerun to hide the feedback buttons

        with col2:
            if st.button("👎 No", use_container_width=True):
                st.session_state.feedback_given = True
                # In a real app, you would log this to a database
                print(f"FEEDBACK: NEGATIVE for query: '{response_data['query']}'")
                st.warning("Thank you for your feedback! We'll use it to improve.")
                time.sleep(2) # Keep the message on screen for a moment
                st.rerun() # Rerun to hide the feedback buttons