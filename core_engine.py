# --- PATCH FOR STREAMLIT DEPLOYMENT ---
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END PATCH ---

import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the paths
KNOWLEDGE_BASE_DIR = 'knowledge_base'
DB_DIR = 'db'

# --- Functions for building/loading the RAG pipeline ---

def load_documents():
    """Loads documents from the knowledge base directory."""
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt")
    return loader.load()

def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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

def get_vector_db():
    """Loads the vector database. Builds it if it doesn't exist."""
    embedding_function = get_embedding_function()
    if not os.path.exists(DB_DIR):
        print("Database not found. Building now...")
        documents = load_documents()
        text_chunks = split_documents(documents)
        db = setup_vector_database(text_chunks)
    else:
        db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    return db

# --- Function for the RAG chain ---

def query_rag(query_text: str):
    """
    Queries the RAG pipeline and generates a response.
    """
    vector_db = get_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Define the prompt template
    prompt_template = """
    You are the Neuro-AI Explorer, an expert AI assistant. Your goal is to provide clear, concise, and accurate answers based on the context provided.
    Compare and contrast the concepts from the provided text, focusing on the user's question.
    Do not mention that you are answering from a provided context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Initialize the LLM
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama3-70b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": (lambda x: x)}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Invoke the chain
    response = rag_chain.invoke(query_text)
    
    # Retrieve the source documents
    source_docs = retriever.invoke(query_text)
    
    return response, source_docs