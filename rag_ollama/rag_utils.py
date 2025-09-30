import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def load_document(file_path):
    """
    Loads a document from the given file path.
    
    This function uses PyPDFLoader for PDF files. If you need to support
    other file types like DOCX or TXT, you can add more loaders here
    or use UnstructuredFileLoader, which is more general but requires
    additional system dependencies.
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    # Add other file types as needed, e.g., for .docx, .txt
    # elif file_path.endswith(".docx"):
    #     loader = UnstructuredWordDocumentLoader(file_path)
    # else:
    #     raise ValueError(f"Unsupported file type for {file_path}")
    
    return loader.load()

def chunk_document(docs):
    """
    Splits a list of documents into smaller, manageable chunks.
    
    Adjust chunk_size and chunk_overlap to optimize for your specific
    documents and model context window.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(docs)

def create_faiss_index(chunks, embedding_model_name, index_path="faiss_index"):
    """
    Creates a new FAISS vector store from document chunks and saves it locally.
    
    This function is used for the initial setup or when the source document changes.
    """
    print("Creating FAISS vector store...")
    embedding_model = OllamaEmbeddings(model=embedding_model_name)
    faiss_db = FAISS.from_documents(chunks, embedding_model)
    faiss_db.save_local(index_path)
    print(f"FAISS index created and saved to '{index_path}'.")
    return faiss_db

def load_faiss_index(embedding_model_name, index_path="faiss_index"):
    """
    Loads a pre-existing FAISS vector store from a local directory.
    
    This is faster than recreating the index every time the application runs.
    """
    print(f"Loading FAISS index from '{index_path}'...")
    embedding_model = OllamaEmbeddings(model=embedding_model_name)
    faiss_db = FAISS.load_local(
        index_path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded successfully.")
    return faiss_db

def get_qa_chain(llm_model_name, retriever):
    """
    Creates and returns the Retrieval-Augmented Generation (RAG) chain.
    """
    llm = Ollama(model=llm_model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
