# (The conditional patch for pysqlite3 at the top remains)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass

import os
import re
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

KNOWLEDGE_BASE_DIR = 'knowledge_base'
DB_DIR = 'db'

def get_embedding_function():
    """Gets the embedding function."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_db():
    embedding_function = get_embedding_function()
    if not os.path.exists(DB_DIR):
        print("Database not found. Building now from .jsonl file...")
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, 'neural_lab_kb.jsonl')
        loader = JSONLoader(file_path=file_path, jq_schema='.', content_key="text", json_lines=True)
        documents = loader.load()
        text_chunks = documents
        db = Chroma.from_documents(text_chunks, embedding_function, persist_directory=DB_DIR)
        print("Vector database setup complete.")
    else:
        db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    return db

def query_rag(query_text: str, api_key: str):
    vector_db = get_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    prompt_template_str = """
    You are an expert AI assistant. Your goal is to provide clear, concise, and accurate answers based on the context provided.
    Compare and contrast the concepts from the provided text, focusing on the user's question.
    Do not mention that you are answering from a provided context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    
    llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", api_key=api_key)
    
    rag_chain = ({"context": retriever, "question": (lambda x: x)} | prompt | llm | StrOutputParser())
    response = rag_chain.invoke(query_text)
    source_docs = retriever.invoke(query_text)
    return response, source_docs

def generate_related_questions(query: str, answer: str, api_key: str):
    prompt_template_str = """
    Based on the following user query and the provided answer, please generate 3 to 5 follow-up questions...
    (rest of template is the same)
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    
    llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192", api_key=api_key)
    
    question_generation_chain = prompt | llm | StrOutputParser()
    response_text = question_generation_chain.invoke({"query": query, "answer": answer})
    questions = re.findall(r'^\d+\.\s*(.*)', response_text, re.MULTILINE)
    return questions