from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from typing import List, Dict
from agents import (
    ConversationClassifier,
    ChatAgent,
    RetrieverAgent,
    ValidatorAgent,
    DialogAgent,
    SolverAgent
)

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    clarification_attempts: int = 0  # Track clarification attempts

# Initialize components
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
vectorstore = FAISS.load_local("serviceNow_Index", embedding, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=google_api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize all agents
classifier = ConversationClassifier(llm)
chat_agent = ChatAgent(llm, memory)
retriever = RetrieverAgent(vectorstore, memory)
validator = ValidatorAgent(llm, memory)
dialog = DialogAgent(llm, memory)
solver = SolverAgent(llm, memory)

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Classify query type
        query_type = classifier.classify(req.question, req.history)
        memory.chat_memory.add_user_message(req.question)
        
        if query_type == "chat":
            response = chat_agent.respond(req.question)
            memory.chat_memory.add_ai_message(response)
            return {
                "answer": response,
                "type": "chat",
                "follow_up_needed": False,
                "clarification_attempts": 0  # Reset attempts
            }
        
        # Knowledge-based flow
        solutions = retriever.retrieve(req.question)
        is_complete, valid_solutions, missing_details = validator.validate(req.question, solutions)
        
        MAX_ATTEMPTS = 3  # Maximum allowed clarification attempts
        
        if not is_complete and req.clarification_attempts < MAX_ATTEMPTS:
            follow_up = dialog.ask_clarification(missing_details)
            memory.chat_memory.add_ai_message(follow_up)
            return {
                "answer": follow_up,
                "type": "clarification",
                "follow_up_needed": True,
                "partial_solutions": valid_solutions,
                "clarification_attempts": req.clarification_attempts + 1
            }
        elif not is_complete:  # Reached max attempts
            if valid_solutions:
                final_response = solver.solve(req.question, valid_solutions)
                final_response = "After several attempts to clarify, here's the best I can suggest:\n\n" + final_response
            else:
                final_response = "I'm sorry, after several attempts I still don't have enough information to provide a complete solution."
            
            memory.chat_memory.add_ai_message(final_response)
            return {
                "answer": final_response,
                "type": "solution",
                "follow_up_needed": False,
                "clarification_attempts": 0
            }
        
        # If we have complete solutions
        final_response = solver.solve(req.question, valid_solutions)
        memory.chat_memory.add_ai_message(final_response)
        return {
            "answer": final_response,
            "type": "solution",
            "follow_up_needed": False,
            "clarification_attempts": 0
        }

    except Exception as e:
        return {"error": str(e), "type": "error"}
