# The Neural Intelligence Lab

Welcome to the Neural Intelligence Lab! This is a web-based, interactive Q&A application designed to educate users on the parallels and differences between biological brains (neuroscience) and artificial intelligence (AI).

This application is built using a Retrieval-Augmented Generation (RAG) pipeline, which allows it to answer questions based on a curated knowledge base of text documents.

**Live Application URL:** [https://neuro-ai-explorer-pmyhumw8fev2ajhuf2pacl.streamlit.app/)

![Neuro-AI Explorer Screenshot](https://i.imgur.com/8aV4Y1U.png)

## Features

-   **Interactive Q&A:** Ask questions in natural language.
-   **RAG Pipeline:** Retrieves relevant information from a local knowledge base to provide contextually accurate answers.
-   **LLM-Powered Generation:** Uses Groq and Llama 3 to generate smooth, human-like answers based on the retrieved context.
-   **User Feedback:** "Thumbs up/down" buttons to evaluate the quality of the answers.
-   **Cloud Deployed:** Fully deployed and accessible on Streamlit Community Cloud.

## Tech Stack

-   **Python:** Core programming language.
-   **Streamlit:** For building the interactive web application frontend.
-   **LangChain:** To orchestrate the RAG pipeline (document loading, splitting, retrieval, and generation).
-   **Hugging Face:** For the `all-MiniLM-L6-v2` sentence-transformer embedding model.
-   **ChromaDB:** As the local vector store for our knowledge base embeddings.
-   **Groq:** For providing high-speed inference with the Llama 3 language model.

## How It Works

The application follows a complete Retrieval-Augmented Generation (RAG) workflow:

1.  **Load & Chunk:** Text files from the `knowledge_base/` directory are loaded and split into smaller, manageable chunks.
2.  **Embed & Store:** Each chunk is converted into a numerical vector (embedding) using a Hugging Face model and stored in a local ChromaDB vector database. This is a one-time setup process.
3.  **Retrieve:** When a user asks a question, the app embeds the query and uses it to find the most relevant text chunks from the ChromaDB database (semantic search).
4.  **Generate:** The retrieved chunks and the original question are passed to a powerful LLM (Llama 3 via Groq) with a custom prompt. The LLM then synthesizes this information to generate a final, coherent answer.
5.  **Feedback:** The user can provide feedback on the generated answer, which is logged for future evaluation.

## How to Run Locally

To run this project on your own machine, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Tab-Alk/neuro-ai-explorer.git](https://github.com/Tab-Alk/neuro-ai-explorer.git)
    cd neuro-ai-explorer
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Get a Groq API Key:**
    -   Visit [Groq.com](https://console.groq.com/keys) to get a free API key.
    -   Create a file named `.streamlit/secrets.toml` and add your key to it like this:
        ```toml
        GROQ_API_KEY="your_api_key_here"
        ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

---
