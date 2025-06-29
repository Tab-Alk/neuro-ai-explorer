
# The Neural Intelligence Lab

Welcome to the Neural Intelligence Lab! This is a web-based, interactive knowledge discovery platform designed to accelerate research and analysis at the intersection of neuroscience and artificial intelligence. It allows users to explore the fascinating parallels and differences between biological brains and artificial intelligence, specifically addressing the challenge of efficiently navigating and synthesizing information from vast and diverse scientific literature.

Live Application URL: [https://neuro-ai-explorer-pmyhumw8fev2ajhuf2pacl.streamlit.app/#sources-with-highlighting](https://neuro-ai-explorer-pmyhumw8fev2ajhuf2pacl.streamlit.app/#sources-with-highlighting)

## Features

* **Accelerated Literature Review:** Quickly get precise answers to complex research questions by drawing exclusively from a curated knowledge base that now includes scientific papers in PDF format.
* Interactive Q&A: Ask questions in natural language.
* AI-Generated Answers: Uses a powerful Large Language Model (Llama 3 via Groq) to provide clear, synthesized answers.
* Explainable AI ("The Glass Box"):
    * Semantic Highlighting: After an answer is generated, the application visually highlights the specific source sentences that were most semantically similar to the answer, showing exactly where the information came from.
    * Full Source Exploration: Allows users to expand and view the full text of the original source documents, including excerpts from PDF pages.
* Knowledge Discovery ("The Guided Tour"):
    * Proactive Suggestions: The application doesn't just answer—it inspires further learning by using an LLM to dynamically generate relevant follow-up questions based on your current query, aiding deeper exploration of research topics.
* Quantitative Evaluation Framework ("The Report Card"):
    * Includes a "gold standard" test set and an evaluation script (`evaluate.py`) that uses the Ragas framework to quantitatively measure RAG performance across key metrics like faithfulness, relevance, and context utilization.

## Tech Stack

* Python: Core programming language.
* Streamlit: For building the interactive web application frontend.
* LangChain: To orchestrate the RAG pipeline (document loading, retrieval, and generation).
* LlamaParse: For advanced, intelligent parsing and ingestion of PDF documents, extracting text, tables, and structured metadata.
* Groq: For providing high-speed inference with Llama 3 language models.
* Hugging Face: For the `all-MiniLM-L6-v2` sentence-transformer embedding model.
* ChromaDB: As the local vector store for knowledge base embeddings.
* Ragas: For the quantitative evaluation of the RAG pipeline.
* scikit-learn: For calculating cosine similarity in the semantic highlighting feature.

## How It Works

The application follows a complete Retrieval-Augmented Generation (RAG) workflow:

* **Ingest & Load:** New scientific papers and knowledge artifacts (including PDFs) are processed using LlamaParse for robust content extraction. On first run, or when new PDFs are added, knowledge base files (in `.jsonl` format or parsed from PDF), are processed into vector embeddings using a Hugging Face model, and stored in a ChromaDB vector database.
* Retrieve: When a user asks a question, the app embeds the query and uses semantic search to find the most relevant text chunks from the ChromaDB database. This retrieval step is carefully controlled to ensure the LLM receives the most pertinent context, especially from previously ingested PDF documents.
* Generate: The retrieved chunks and the original question are passed to a powerful LLM (Llama 3 via Groq) with a custom prompt. The LLM then synthesizes this information to generate a final, coherent answer.
* Explain & Suggest: The app then semantically compares the generated answer to the source text to provide highlighting. It also uses the question/answer context to generate a list of relevant follow-up questions.

## Evaluating Performance

This project includes a robust evaluation suite to measure the quality of the RAG system.

### Set up the Environment:

1. Ensure all dependencies from `requirements.txt` are installed in your virtual environment (`pip install -r requirements.txt`).
2. Create a `.env` file in the root directory and add your `GROQ_API_KEY` and `LLAMA_CLOUD_API_KEY` (required for LlamaParse).

### Ingest PDFs (Optional - if you have PDF documents to add to the knowledge base):

1. Place your PDF files into a designated folder (e.g., `data/pdfs`).
2. Run a script or function (e.g., `ingest_pdfs_from_folder("data/pdfs")` in `core_engine.py`) to process and add them to your ChromaDB.

### Run the Evaluation:

```bash
python evaluate.py
```

This script will run the questions from `evaluation/gold_standard_test_set.json` against the RAG system and use Ragas to score the results. The final report will be saved to `evaluation/evaluation_results.csv`.