import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from rag_utils import load_faiss_index, get_qa_chain

# --- Phase 1: Load the Saved FAISS Index ---
# This is a new step that replaces the data ingestion and indexing part of the previous script.

# Define the folder where your FAISS index is saved
FAISS_INDEX_PATH = "faiss_index"
embedding_model_name = "mxbai-embed-large:latest"
ollama_model_name = "llama3.2:1b"

# Ensure the embedding model is the same as the one used to create the index

# Load the saved FAISS index from the local directory.
# The `allow_dangerous_deserialization` flag is necessary for LangChain versions >= 0.1.17
# to load a local FAISS index due to security updates.
try:
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    faiss_db = load_faiss_index(embedding_model_name, FAISS_INDEX_PATH)
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Ensure the 'faiss_index' directory exists and contains the index files.")
    exit() # Exit the script if the index cannot be loaded.

# --- Phase 2: Retrieval and Generation (The RAG Chain) ---
# This part of the code remains the same as it uses the loaded vector store.


# Create a retriever from the loaded FAISS vector store.
retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

# Create the Retrieval-Augmented Generation (RAG) chain.
qa_chain = get_qa_chain(ollama_model_name, retriever)

# --- Ask a question and get a response ---

while True:
    query = input("\nEnter a question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    
    print(f"\nUser Query: {query}")
    response = qa_chain.invoke({"query": query})
    print(f"RAG Response: {response['result']}")

print("Exiting RAG application.")