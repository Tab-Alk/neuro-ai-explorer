import streamlit as st
from core_engine import query_rag

# --- App UI ---

# Set the page title
st.set_page_config(page_title="Neuro-AI Explorer", page_icon="🤖")

# Display the title and a short introduction
st.title("🧠 Neuro-AI Explorer 🤖")
st.write(
    "Ask a question about the fascinating parallels and differences "
    "between biological brains and artificial intelligence."
)
st.write(
    "Example: *How does memory in an AI compare to a human brain?*"
)


# Get user input
user_query = st.text_input(
    "Your Question:", 
    placeholder="Type your question here..."
)

# Submit button
if st.button("Get Answer"):
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Searching the knowledge base..."):
            # Call the RAG engine to get the context
            context, sources = query_rag(user_query)

            if context:
                # Display the retrieved context
                st.subheader("Retrieved Information:")
                st.info(context)
                
                # Optional: Display sources
                st.markdown("---")
                st.subheader("Sources:")
                for i, source in enumerate(sources):
                    st.write(f"**Source {i+1}:** `{source[0].metadata['source']}`")
                    st.write(f"**Relevance Score:** `{source[1]:.2f}`")

            else:
                st.error("No relevant information found in the knowledge base.")