import os
from rag_utils import (
    load_document, 
    chunk_document, 
    create_faiss_index, 
    load_faiss_index, 
    get_qa_chain
)

# Configuration Section
DOCUMENT_PATH = "t3.pdf"
FAISS_INDEX_PATH = f"faiss_index"
EMBEDDING_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "mistral:latest"

if __name__ == "__main__":
    # Check if the FAISS index exists
    if not os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index not found. Creating a new one...")
        
        # Load and chunk the document
        docs = load_document(DOCUMENT_PATH)
        chunks = chunk_document(docs)
        print(len(chunks), "chunks created from the document.")
        
        # Create and save the new FAISS index
        faiss_db = create_faiss_index(chunks, EMBEDDING_MODEL, FAISS_INDEX_PATH)
    else:
        # Load the existing FAISS index
        print("Existing FAISS index found.")
        faiss_db = load_faiss_index(EMBEDDING_MODEL, FAISS_INDEX_PATH)

    # Set up the RAG Chain
    retriever = faiss_db.as_retriever(search_kwargs={"k": 1000})
    
    # Get the RetrievalQA chain
    qa_chain = get_qa_chain(LLM_MODEL, retriever)

    # Start the Query Loop
    while True:
        query = input("\nEnter a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        print(f"\nUser Query: {query}")
        response = qa_chain.invoke({"query": query})
        print(f"RAG Response: {response['result']}")

    print("Exiting RAG application.")
