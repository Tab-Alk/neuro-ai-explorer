# --- PATCH FOR STREAMLIT DEPLOYMENT ---
# This is a hack to use the newer version of sqlite3
# See: https://discuss.streamlit.io/t/issues-with-chromadb-and-sqlite3-on-streamlit-community-cloud/47769
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END PATCH ---

import os
from langchain_community.document_loaders import DirectoryLoader
# ... the rest of your code continues below ...

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define the paths
KNOWLEDGE_BASE_DIR = 'knowledge_base'
DB_DIR = 'db'

# --- Functions for building the RAG pipeline ---

def load_documents():
    """Loads documents from the knowledge base directory."""
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt")
    return loader.load()

def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def get_embedding_function():
    """Initializes the embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def setup_vector_database(documents):
    """Creates and persists the vector database."""
    print("Setting up vector database...")
    embeddings = get_embedding_function()
    db = Chroma.from_documents(
        documents, embeddings, persist_directory=DB_DIR
    )
    print("Vector database setup complete.")
    return db

# --- Main function to be called by the app ---

def query_rag(query_text: str):
    """
    Queries the RAG pipeline.
    If the DB doesn't exist, it builds it first.
    """
    embedding_function = get_embedding_function()
    
    # Check if the database directory exists
    if not os.path.exists(DB_DIR):
        print("Database not found. Building now...")
        # If DB doesn't exist, build it
        documents = load_documents()
        text_chunks = split_documents(documents)
        setup_vector_database(text_chunks)
    
    # Load the persisted database from disk
    db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)

    # Retrieve relevant documents
    results = db.similarity_search_with_score(query_text, k=2)
    
    # Format the context for the response
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    return context_text, results