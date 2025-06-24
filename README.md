# The Neural Intelligence Lab

Welcome to the Neural Intelligence Lab! This is a web-based, interactive Q&A application designed to be a transparent and helpful knowledge discovery platform. It allows users to explore the fascinating parallels and differences between biological brains and artificial intelligence.

This application is built using a modern Retrieval-Augmented Generation (RAG) pipeline, which allows it to answer questions based on a curated knowledge base and then guide the user toward new, interesting concepts.

**Live Application URL:** [https://neuro-ai-explorer-pmyhumw8fev2ajhuf2pacl.streamlit.app/#sources-with-highlighting ] 
*(Note: Please replace with your final URL if different)*

## Features

-   **Interactive Q&A:** Ask questions in natural language.
-   **AI-Generated Answers:** Uses a powerful Large Language Model (Llama 3 via Groq) to provide clear, synthesized answers.
-   **Explainable AI ("The Glass Box"):**
    -   **Semantic Highlighting:** After an answer is generated, the application visually highlights the specific source sentences that were most semantically similar to the answer, showing exactly where the information came from.
    -   **Full Source Exploration:** Allows users to expand and view the full text of the original source documents.
-   **Knowledge Discovery ("The Guided Tour"):**
    -   **Proactive Suggestions:** The application doesn't just answer—it inspires further learning by using an LLM to dynamically generate relevant follow-up questions based on your current query.
-   **Quantitative Evaluation Framework ("The Report Card"):**
    -   Includes a "gold standard" test set and an evaluation script (`evaluate.py`) that uses the Ragas framework to quantitatively measure RAG performance across key metrics like faithfulness, relevance, and context utilization.

## Tech Stack

-   **Python:** Core programming language.
-   **Streamlit:** For building the interactive web application frontend.
-   **LangChain:** To orchestrate the RAG pipeline (document loading, retrieval, and generation).
-   **Groq:** For providing high-speed inference with Llama 3 language models.
-   **Hugging Face:** For the `all-MiniLM-L6-v2` sentence-transformer embedding model.
-   **ChromaDB:** As the local vector store for knowledge base embeddings.
-   **Ragas:** For the quantitative evaluation of the RAG pipeline.
-   **scikit-learn:** For calculating cosine similarity in the semantic highlighting feature.

## How It Works

The application follows a complete Retrieval-Augmented Generation (RAG) workflow:

1.  **Load & Store:** On first run, knowledge base files (in `.jsonl` format) are loaded, processed into vector embeddings using a Hugging Face model, and stored in a ChromaDB vector database.
2.  **Retrieve:** When a user asks a question, the app embeds the query and uses semantic search to find the most relevant text chunks from the ChromaDB database.
3.  **Generate:** The retrieved chunks and the original question are passed to a powerful LLM (Llama 3 via Groq) with a custom prompt. The LLM then synthesizes this information to generate a final, coherent answer.
4.  **Explain & Suggest:** The app then semantically compares the generated answer to the source text to provide highlighting. It also uses the question/answer context to generate a list of relevant follow-up questions.

## Evaluating Performance

This project includes a robust evaluation suite to measure the quality of the RAG system.

1.  **Set up the Environment:**
    -   Ensure all dependencies from `requirements.txt` are installed in your virtual environment (`pip install -r requirements.txt`).
    -   Create a `.env` file in the root directory and add your `GROQ_API_KEY`.

2.  **Run the Evaluation:**
    ```bash
    python evaluate.py
    ```
    This script will run the questions from `evaluation/gold_standard_test_set.json` against the RAG system and use Ragas to score the results. The final report will be saved to `evaluation/evaluation_results.csv`.

---
