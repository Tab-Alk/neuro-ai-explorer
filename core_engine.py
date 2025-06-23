# --- PATCH FOR STREAMLIT DEPLOYMENT ---
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END PATCH ---

import os
import streamlit as st
import re
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Define the paths
KNOWLEDGE_BASE_DIR = 'knowledge_base'
DB_DIR = 'db'

# --- Functions for building/loading the RAG pipeline ---
def get_vector_db():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(DB_DIR):
        print("Database not found. Building now from .jsonl file...")
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, 'neural_lab_kb.jsonl')
        loader = JSONLoader(
            file_path=file_path,
            json_lines=True,
            content_key="content"
        )
        documents = loader.load()
        db = Chroma.from_documents(
            documents, embedding_function, persist_directory=DB_DIR
        )
        print("Vector database setup complete.")
    else:
        db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    return db

# --- CORRECTED RAG CHAIN FUNCTION ---
def get_rag_chain():
    """Creates and returns a RAG chain that returns sources."""
    vector_db = get_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    You are an expert AI assistant. Your goal is to provide a clear and concise answer based only on the context provided.
    If the context does not contain the answer, state that you cannot answer the question with the given information.
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # This is the new, correct chain structure
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # This chain runs in parallel to the main one to retrieve the raw source documents
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "answer": rag_chain}
    )
    
    return rag_chain_with_source

# --- Main query function called by the app ---
def query_rag(query_text: str):
    """
    Invokes the RAG chain and returns the answer and source documents.
    """
    rag_chain = get_rag_chain()
    result = rag_chain.invoke(query_text)
    
    answer = result["answer"]
    sources = result["context"]
    
    return answer, sources

# (The generate_related_questions function remains the same and is not shown here for brevity)
def generate_related_questions(query: str, answer: str):
    # ... (no changes to this function)
    prompt_template = """
    Based on the following user query and the provided answer, please generate 3 to 5 follow-up questions...
    (rest of the function is identical)
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