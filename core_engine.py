# --- PATCH FOR STREAMLIT DEPLOYMENT ---
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END PATCH ---

import os
import streamlit as st
import re
from langchain_community.document_loaders import JSONLoader
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

def get_vector_db():
    """Loads the vector database. Builds it if it doesn't exist."""
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(DB_DIR):
        print("Database not found. Building now from .jsonl file...")
        
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, 'neural_lab_kb.jsonl')

        # --- FINAL, CORRECTED JSONLoader CONFIGURATION ---
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.',          # Load the entire JSON object from each line
            content_key="content",  # Use the 'content' field for the main text
            json_lines=True         # Specify that it's a JSON Lines file
        )
        # --- END CORRECTION ---
        
        documents = loader.load()
        text_chunks = documents 

        db = Chroma.from_documents(
            text_chunks, embedding_function, persist_directory=DB_DIR
        )
        print("Vector database setup complete.")
    else:
        db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    return db

# (The rest of the file remains the same)
# --- Function for the RAG chain ---

def query_rag(query_text: str):
    vector_db = get_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    prompt_template = """
    You are an expert AI assistant. Your goal is to provide clear, concise, and accurate answers based on the context provided.
    Compare and contrast the concepts from the provided text, focusing on the user's question.
    Do not mention that you are answering from a provided context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGroq(
        temperature=0.2,
        model_name="llama3-70b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    rag_chain = (
        {"context": retriever, "question": (lambda x: x)}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(query_text)
    source_docs = retriever.invoke(query_text)
    
    return response, source_docs

# --- Function for Related Questions ---

def generate_related_questions(query: str, answer: str):
    prompt_template = """
    Based on the following user query and the provided answer, please generate 3 to 5 follow-up questions that would be logical next steps for a curious user to explore.
    The questions should be distinct from the original query and delve deeper into related topics or explore new, relevant tangents.
    Return ONLY the questions, each on a new line, and starting with a number (e.g., "1. How does..."). Do not include any other text or preamble.

    ORIGINAL QUERY:
    {query}

    GENERATED ANSWER:
    {answer}

    RELATED QUESTIONS:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGroq(
        temperature=0.7,
        model_name="llama3-8b-8192",
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    question_generation_chain = prompt | llm | StrOutputParser()
    response_text = question_generation_chain.invoke({"query": query, "answer": answer})
    questions = re.findall(r'^\d+\.\s*(.*)', response_text, re.MULTILINE)
    
    return questions