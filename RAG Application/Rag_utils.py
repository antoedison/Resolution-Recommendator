from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd

# Function to load PDF
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)

def load_json(file):
    import json
    with open(file, "r") as f:
        data = json.load(f)
    return json.dumps(data,indent=2)

# Split text into chunks
def split_text(text,x_filename,pdf_hash):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50
    )

    all_chunks = splitter.create_documents([text])
    for chunk in all_chunks:
        chunk.metadata["pdf_name"] = x_filename
        chunk.metadata["Pdf_hash"] = pdf_hash
    return splitter.create_documents([text])


# Create FAISS vectorstore with batching and filtering
def build_vectorstore(chunks, save_path, existing_vectostore=None, batch_size=10):
    all_texts = [doc.page_content.strip() for doc in chunks if doc.page_content.strip()]
    all_metadatas = [doc.metadata for doc in chunks if doc.page_content.strip()]

    filtered_texts = []
    filtered_metadatas = []

    for t, m in zip(all_texts, all_metadatas):
        if 10 < len(t) < 2000:
            filtered_texts.append(t)
            filtered_metadatas.append(m)

    if not filtered_texts:
        raise ValueError("No valid text chunks found for embedding.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    embedded_documents = []

    for i in range(0, len(filtered_texts), batch_size):
        batch_texts = filtered_texts[i:i + batch_size]
        batch_metadatas = filtered_metadatas[i:i + batch_size]

        print(f"🔄 Embedding batch {i // batch_size + 1} with {len(batch_texts)} texts...")
        for j, t in enumerate(batch_texts):
            print(f"   ➤ Chunk {j + 1} preview: {t[:100]}...")

        embedded_documents.extend([
            Document(page_content=text, metadata=meta)
            for text, meta in zip(batch_texts, batch_metadatas)
        ])

    # ✅ Embedding will be done internally here
    vectorstore = FAISS.from_documents(embedded_documents, embeddings)
    if existing_vectostore:
        existing_vectostore.merge_from(vectorstore)
        existing_vectostore.save_local(save_path)
        return existing_vectostore
    else:
        vectorstore.save_local(save_path)
        return vectorstore

# Create RetrievalQA chain
def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, convert_system_message_to_human=True)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs = {"k": 10}),
        return_source_documents=True
    )