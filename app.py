import streamlit as st
from core_engine import query_rag
import time
from nltk.tokenize import sent_tokenize
from thefuzz import fuzz

# --- App State Management ---
def initialize_state():
    """Initializes the session state for the app."""
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'feedback_given' not in st.session_state:
        st.session_state.feedback_given = False

# --- Text Processing and Highlighting ---
def highlight_text(source_text, generated_answer, threshold=85):
    """Highlights sentences in source_text that are similar to sentences in generated_answer."""
    source_sentences = sent_tokenize(source_text)
    answer_sentences = sent_tokenize(generated_answer)
    
    highlighted_sentences = set()

    # Find matches
    for a_sent in answer_sentences:
        for s_sent in source_sentences:
            # Using token_set_ratio for more flexible matching
            similarity = fuzz.token_set_ratio(a_sent, s_sent)
            if similarity > threshold:
                highlighted_sentences.add(s_sent)

    # Reconstruct the text with highlighting
    final_text = ""
    for s_sent in source_sentences:
        if s_sent in highlighted_sentences:
            # Using a yellow background for highlighting
            final_text += f"<mark style='background-color: yellow;'>{s_sent}</mark> "
        else:
            final_text += f"{s_sent} "
            
    return final_text.strip()


# --- UI Rendering Functions ---
def display_header():
    """Displays the header and introduction of the app."""
    st.title("The Neural Intelligence Lab")
    st.write("Ask a question about the fascinating parallels and differences...") # Truncated for brevity
    st.write("*How does memory in an AI compare to a human brain?*")
    st.write("*How is a biological neuron different from an artificial neuron?*")
    st.write("*Why can AI beat humans at chess but struggle with common sense?*")
    st.write("*Could we download our memories into AI?*")

def display_response_and_feedback():
    """Displays the generated answer, sources, and feedback buttons."""
    response_data = st.session_state.response
    
    st.subheader("Answer:")
    st.write(response_data["answer"])
    
    # --- NEW: Placeholder for Related Questions ---
    with st.expander("**See questions related to your topic**"):
        st.write("Here are some related questions you might want to explore:")
        st.button("How is long-term memory consolidation different in brains vs. AI?", disabled=True)
        st.button("What are the ethical implications of advanced AI memory?", disabled=True)
        st.button("Can an AI truly 'forget' like a human does?", disabled=True)
    # --- END NEW SECTION ---

    st.markdown("---")
    st.subheader("The Science Behind It:")

    # Combine all source documents into one text block for processing
    full_source_text = "\n\n".join([doc.page_content for doc in response_data["sources"]])
    
    # Get the highlighted version of the source text
    highlighted_source = highlight_text(full_source_text, response_data["answer"])
    
    # Display the highlighted source text in an expander
    with st.expander("**View the Highlighted Source Material Used**"):
        st.markdown(highlighted_source, unsafe_allow_html=True)
        
    # Display feedback buttons if not already given
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
# NOTE: Need to add this for the sentence tokenizer to work
import nltk
nltk.download('punkt')

st.set_page_config(page_title="The Neural Intelligence Lab")
initialize_state()
display_header()
user_query = st.text_input("Your Question:", placeholder="Type your question here...")

if st.button("Get Answer"):
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        generate_response(user_query)

if st.session_state.response:
    display_response_and_feedback()