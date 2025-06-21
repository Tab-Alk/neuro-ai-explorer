import streamlit as st
from core_engine import query_rag
import time

# --- App UI ---

st.set_page_config(page_title="Neuro-AI Explorer", page_icon="🤖")

st.title("🧠 Neuro-AI Explorer 🤖")
st.write(
    "Ask a question about the fascinating parallels and differences "
    "between biological brains and artificial intelligence."
)
st.write(
    "Example: *How does memory in an AI compare to a human brain?*"
)

# --- Session State Management ---
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
        with st.spinner("Synthesizing answer..."):
            st.session_state.feedback_given = False
            answer, sources = query_rag(user_query)
            st.session_state.response = {
                "query": user_query,
                "answer": answer,
                "sources": sources
            }

# --- Display Response and Feedback Buttons ---
if st.session_state.response:
    response_data = st.session_state.response
    
    st.subheader("Answer:")
    st.write(response_data["answer"])
    
    st.markdown("---")
    st.subheader("Sources:")
    for source in response_data["sources"]:
        st.write(f"- {source.metadata['source']}")

    # --- Feedback Logic ---
    if not st.session_state.feedback_given:
        st.markdown("---")
        st.write("Was this answer helpful?")
        
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Yes", use_container_width=True):
                st.session_state.feedback_given = True
                print(f"FEEDBACK: POSITIVE for query: '{response_data['query']}'")
                st.success("Thank you for your feedback!")
                time.sleep(2)
                st.rerun()

        with col2:
            if st.button("👎 No", use_container_width=True):
                st.session_state.feedback_given = True
                print(f"FEEDBACK: NEGATIVE for query: '{response_data['query']}'")
                st.warning("Thank you for your feedback! We'll use it to improve.")
                time.sleep(2)
                st.rerun()