import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# --- Phase 1: Data Ingestion (Loading and Chunking Documents) ---

# Replace 'your_document.pdf' with your PDF, DOCX, or JSON file.
# For other file types, use the appropriate LangChain Document Loader.
# Example for a PDF file:
loader = PyPDFLoader("t.pdf")
docs = loader.load()

# Split the documents into smaller, manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

# --- Phase 2: Vector Store Creation (Embedding and Indexing) ---

# Specify the Ollama embedding model you're using.
embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")

# Create a FAISS vector store from the document chunks.
# This will generate embeddings and build the FAISS index in-memory.
print("Creating FAISS vector store... This may take a moment.")
faiss_db = FAISS.from_documents(chunks, embedding_model)
print("FAISS vector store created.")

# Optionally, you can save the FAISS index to disk to avoid re-embedding.
faiss_db.save_local("faiss_index")

# To load a saved index, you would use:
# faiss_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# --- Phase 3: Retrieval and Generation (The RAG Chain) ---

# Instantiate the Ollama LLM for text generation.
# Make sure to set the model name to your LLM.
llm = Ollama(model="llama3.2:1b")

# Create a retriever from the FAISS vector store.
# This retriever will find the most relevant chunks when a query is made.
retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

# Create the Retrieval-Augmented Generation (RAG) chain.
# The 'stuff' chain type will combine the retrieved chunks into a single prompt.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# --- Ask a question and get a response ---

query = "What is the main topic of the document?"
print(f"\nUser Query: {query}")
response = qa_chain.invoke({"query": query})
print(f"RAG Response: {response['result']}")

# You can now ask more questions.
query = "Can you summarize the key points?"
print(f"\nUser Query: {query}")
response = qa_chain.invoke({"query": query})
print(f"RAG Response: {response['result']}")