# core engine.py
#  (The conditional patch for pysqlite3 at the top remains)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ModuleNotFoundError:
    pass
from llama_parse import LlamaParse
from langchain_chroma import Chroma

import os
import re
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    
    # Step 1: Retrieve docs
    source_docs = retriever.invoke(query_text)
    
    # Step 2: Combine content from the chunks
    context_text = "\n\n".join([doc.page_content for doc in source_docs])
    
    # Step 3: Define prompt
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
    
    # Step 4: Use direct context in RAG chain
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": context_text, "question": query_text})

    print("\n--- Retrieved Sources ---")
    for i, doc in enumerate(source_docs):
        print(f"[{i}] Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Title: {doc.metadata.get('title', 'No title')}")
        print(f"Excerpt: {doc.page_content[:300]}\n")

    return response, source_docs

# --- THIS IS THE FIXED FUNCTION ---
def generate_related_questions(query: str, answer: str, api_key: str):
    """
    Generates relevant follow-up questions based on the query and answer.
    This version uses a more robust prompt and parsing method.
    """
    # 1. Use a complete and explicit prompt to guide the LLM's output format.
    prompt_template_str = """
    You are a helpful AI assistant. Your task is to generate insightful follow-up questions based on a user's query and the answer they received.

    Analyze the provided query and answer. Brainstorm 3 to 5 relevant questions that would allow a user to explore the topic further. The questions should be distinct from the original query and encourage deeper investigation into related concepts.

    IMPORTANT: Your output MUST be ONLY a numbered list of the questions, with each question on a new line. Do not include any introductory text, concluding remarks, or any other text besides the questions themselves.

    EXAMPLE OUTPUT:
    1. What are the ethical implications of Hebbian learning in autonomous AI?
    2. How is synaptic plasticity modeled in modern neural networks?
    3. Can an AI without Hebbian-style learning ever achieve true generalization?

    ---
    USER QUERY: {query}
    ---
    ANSWER PROVIDED: {answer}
    ---
    
    YOUR GENERATED FOLLOW-UP QUESTIONS (numbered list only):
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    
    llm = ChatGroq(temperature=0.7, model_name="llama3-8b-8192", api_key=api_key)
    
    question_generation_chain = prompt | llm | StrOutputParser()
    response_text = question_generation_chain.invoke({"query": query, "answer": answer})
    
    # 2. Use a more robust parsing method that is less sensitive to formatting errors.
    questions = []
    # Split the response by newlines
    potential_questions = response_text.strip().split('\n')
    for line in potential_questions:
        # Strip leading/trailing whitespace
        clean_line = line.strip()
        # Use regex to remove leading numbers, periods, and spaces (e.g., "1. ", "2. ", etc.)
        question_text = re.sub(r'^\d+\.\s*', '', clean_line)
        # Add to the list only if the line is not empty after cleaning
        if question_text:
            questions.append(question_text)
            
    return questions

# --- LLAMA PARSE PDF INGESTION FUNCTION ---
def parse_and_store_pdf(file_path: str):
    """
    Parses a PDF using LlamaParse, splits it into chunks, and stores the results in Chroma.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    parser = LlamaParse(api_key=api_key)
    docs = parser.load_data(file_path)
    print(f"LlamaParse completed for {file_path}. Number of documents/sections: {len(docs)}")

    for i, doc in enumerate(docs):
        print(f"\n--- LlamaIndex Document {i} Metadata ---")
        print(doc.metadata)

    lc_docs = []
    for i, doc in enumerate(docs):
        metadata = {**doc.metadata, "source": os.path.basename(file_path)}
        if 'page_label' in doc.metadata:
            metadata["page_number"] = doc.metadata['page_label']
        # Try to extract title from the first few lines of the document text
        if i == 0:  # First document section
            lines = doc.text.strip().split("\n")
            for line in lines:
                title_candidate = line.strip()
                if len(title_candidate) > 10 and len(title_candidate.split()) > 3:
                    metadata["title"] = title_candidate
                    break  # Use first reasonable line as title
        lc_docs.append(LangchainDocument(page_content=doc.text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(lc_docs)

    embedding_function = get_embedding_function()
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
        print(f"Loaded existing ChromaDB from {DB_DIR}.")
        db.add_documents(chunks)
        print(f"Added {len(chunks)} chunks from {os.path.basename(file_path)} to existing ChromaDB.")
    else:
        print(f"Creating new ChromaDB at {DB_DIR} for {os.path.basename(file_path)}.")
        db = Chroma.from_documents(chunks, embedding_function=embedding_function, persist_directory=DB_DIR)
        print(f"Created new ChromaDB with {len(chunks)} chunks from {os.path.basename(file_path)}.")


# --- BATCH PDF INGESTION FUNCTION ---
def ingest_pdfs_from_folder(folder_path: str):
    """
    Parses and stores all .pdf files in the specified folder using LlamaParse.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, fname)
            print(f"Processing file: {file_path}")
            parse_and_store_pdf(file_path)