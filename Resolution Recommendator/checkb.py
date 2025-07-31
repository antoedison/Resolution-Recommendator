from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from typing import List, Dict, Literal, Union
from langgraph.graph import StateGraph, END

from agents import (
    ConversationClassifier,
    ChatAgent,
    RetrieverAgent,
    ValidatorAgent,
    DialogAgent,
    SolverAgent
)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Model
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    clarification_attempts: int = 0

# Embedding, Memory, LLM
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
vectorstore = FAISS.load_local("D:/Virtusa_Internship/GenAi/Final_Project/vector DB/serviceNow_Index", embedding, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=google_api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Agent Instances
classifier = ConversationClassifier(llm)
chat_agent = ChatAgent(llm, memory)
retriever = RetrieverAgent(vectorstore, memory)
validator = ValidatorAgent(llm, memory)
dialog = DialogAgent(llm, memory)
solver = SolverAgent(llm, memory)

MAX_ATTEMPTS = 3

# Shared Graph State
class AgentState(dict):
    question: str
    history: List[Dict[str, str]]
    clarification_attempts: int
    query_type: Union[Literal["chat", "knowledge"], None]
    partial_solutions: Union[List[str], None]
    missing_details: Union[str, None]
    final_response: Union[str, None]
    follow_up_needed: bool
    responder_agent: Union[str, None]

# Define LangGraph Nodes
def classify_node(state: AgentState) -> AgentState:
    state["query_type"] = classifier.classify(state["question"], state["history"])
    return state

def chat_node(state: AgentState) -> AgentState:
    response = chat_agent.respond(state["question"])
    state["final_response"] = response
    state["follow_up_needed"] = False
    state["responder_agent"] = "ChatAgent"
    return state

def retrieve_node(state: AgentState) -> AgentState:
    state["partial_solutions"] = retriever.retrieve(state["question"])
    return state

def validate_node(state: AgentState) -> AgentState:
    is_complete, valid_solutions, missing_details = validator.validate(state["question"], state["partial_solutions"])
    state["is_complete"] = is_complete
    state["partial_solutions"] = valid_solutions
    state["missing_details"] = missing_details
    return state

def clarify_node(state: AgentState) -> AgentState:
    state["clarification_attempts"] += 1
    state["final_response"] = dialog.ask_clarification(state["missing_details"])
    state["follow_up_needed"] = True
    state["responder_agent"] = "DialogAgent"
    return state

def solve_node(state: AgentState) -> AgentState:
    solution = solver.solve(state["question"], state["partial_solutions"])
    if state["clarification_attempts"] >= MAX_ATTEMPTS and not state["partial_solutions"]:
        solution = "I'm sorry, after several attempts I still don't have enough information to provide a complete solution."
    elif state["clarification_attempts"] >= MAX_ATTEMPTS:
        solution = "After several attempts to clarify, here's the best I can suggest:\n\n" + solution
    state["final_response"] = solution
    state["follow_up_needed"] = False
    state["responder_agent"] = "SolverAgent" 
    return state

# Router Logic
def route_by_type(state: AgentState) -> str:
    return "chat" if state["query_type"] == "chat" else "retrieve"

def route_by_completion(state: AgentState) -> str:
    if state["is_complete"]:
        return "solve"
    elif state["clarification_attempts"] < MAX_ATTEMPTS:
        return "clarify"
    else:
        return "solve"

# Build the LangGraph
graph = StateGraph(AgentState)
graph.add_node("classify", classify_node)
graph.add_node("chat", chat_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("validate", validate_node)
graph.add_node("clarify", clarify_node)
graph.add_node("solve", solve_node)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_by_type, {
    "chat": "chat",
    "retrieve": "retrieve"
})

graph.add_edge("chat", END)
graph.add_edge("solve", END)

graph.add_edge("retrieve", "validate")
graph.add_conditional_edges("validate", route_by_completion, {
    "clarify": "clarify",
    "solve": "solve"
})
graph.add_edge("clarify", "validate")

langgraph_executor = graph.compile()

# Endpoint using LangGraph
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        state = {
            "question": req.question,
            "history": req.history,
            "clarification_attempts": req.clarification_attempts
        }
        final_state = langgraph_executor.invoke(state)

        return {
            "answer": final_state["final_response"],
            "type": final_state.get("query_type", "solution"),
            "follow_up_needed": final_state.get("follow_up_needed", False),
            "clarification_attempts": final_state.get("clarification_attempts", 0),
            "responder_agent": final_state.get("responder_agent", "Unknown")
        }
    except Exception as e:
        return {"error": str(e), "type": "error"}
