from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile, Query
from fastapi.responses import JSONResponse, HTMLResponse
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from typing import List, Optional
import os
import hashlib
from dotenv import load_dotenv
from Rag_utils import load_pdf, split_text, build_vectorstore, create_qa_chain, load_csv, load_json
from fastapi.middleware.cors import CORSMiddleware
from markdown import markdown
from urllib.parse import unquote

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,   # ✅ fixed typo here
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set vector store path
faiss_index_path = "serviceNow_Index"

# Embedding model
embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# Load existing FAISS index if present
qa = None
db = None
if os.path.exists(faiss_index_path):
    db = FAISS.load_local(faiss_index_path, embedding, allow_dangerous_deserialization=True)
    qa = create_qa_chain(db)


# GET method to list all chunks
@app.get("/answers")
def get_chunks():
    try:
        all_docs: List[Document] = db.similarity_search("", k=1000)
        return {"chunks": [doc.page_content for doc in all_docs]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST method to upload PDF file as raw bytes (not multipart)
"""
@app.post("/Upload_files")
async def upload_raw_pdf(request: Request):
    try:
        body = await request.body()
        filename = "uploaded.pdf"  # Static temp file name

        with open(filename, "wb") as f:
            f.write(body)

        text = load_pdf(filename)
        chunks = split_text(text)
        new_db = build_vectorstore(chunks, faiss_index_path, db)

        # Save new or updated DB
        new_db.save_local(faiss_index_path)

        # Cleanup
        os.remove(filename)

        return {"message": "File processed and chunks embedded", "chunks_added": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""

@app.post("/Upload_files")
async def upload_file(request: Request, encoded_filename: str = Header(None), upload: Optional[UploadFile] = File(None)):

    
    global db

    if upload is not None:
        x_filename = upload.filename
        file_bytes = await upload.read()
    else:
        if not encoded_filename:
            raise HTTPException(status_code=400, detail="Missing 'x_filename' header")
        
        x_filename = unquote(encoded_filename)
        file_bytes = await request.body()

    # Save the uploaded file
    with open(x_filename, "wb") as f:
        f.write(file_bytes)

    try:
        extension = os.path.splitext(x_filename)[1].lower()

        if extension == '.pdf':
            text = load_pdf(x_filename)
        elif extension == '.csv':
            text = load_csv(x_filename)
        elif extension == '.json':
            text = load_json(x_filename)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            

        # Extract and process text
        hash_content = hashlib.sha256(text.encode('utf-8')).hexdigest()
        all_chunks = split_text(text, x_filename, hash_content)

        # Rebuild vectorstore
        vectorstore = build_vectorstore(all_chunks, faiss_index_path, db)
        vectorstore.save_local(faiss_index_path)

        # Update global state
        global qa
        db = vectorstore
        qa = create_qa_chain(vectorstore)

        return JSONResponse(content={
            "message": f"File '{x_filename}' received successfully!",
            "chunks_added": len(all_chunks)
        })

    finally:
        # Always clean up file even if there's an error
        if os.path.exists(x_filename):
            os.remove(x_filename)


@app.get("/answer")
async def get_answers(query: str):
    if qa is None:
        raise HTTPException(status_code=400, detail="No qa chain available. Please upload a PDF first")
    
    try:
        raw_result = qa.invoke(query)
        result = markdown(raw_result["result"])
        return HTMLResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))