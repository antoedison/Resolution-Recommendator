from typing import List, Dict, Tuple, Optional
import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class ConversationClassifier:
    def __init__(self, llm):
        self.llm = llm
    
    def classify(self, query: str, history: List[Dict]) -> str:
        prompt = f"""
        Determine if this query requires:
        1. Technical solution from knowledge base (respond with "knowledge")
        2. General conversation (respond with "chat")
        
        Rules:
        - Use "knowledge" for IT problems, error messages, or technical how-to questions
        - Use "chat" for greetings, thanks, or non-technical questions
        
        Examples:
        Query: "How fix SSL error?" → "knowledge"
        Query: "Hello!" → "chat"
        
        Current conversation:
        {history}
        
        Query: "{query}"
        """
        response = self.llm.invoke(prompt).content.strip().lower()
        return "knowledge" if "knowledge" in response else "chat"

class ChatAgent:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.prompt_template = PromptTemplate(
            input_variables=["history", "message"],
            template="""
            You're a friendly IT helpdesk assistant. Respond naturally to this:
            
            History:
            {history}
            
            Message:
            {message}
            
            Guidelines:
            1. Keep responses under 2 sentences
            2. Maintain professional tone
            3. For technical questions, say "Let me check our knowledge base"
            """
        )
    
    def respond(self, message: str) -> str:
        history = self.memory.load_memory_variables({})["chat_history"]
        prompt = self.prompt_template.format(history=history, message=message)
        return self.llm.invoke(prompt).content

class RetrieverAgent:
    def __init__(self, vectorstore, memory):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        self.memory = memory
    
    def retrieve(self, query: str) -> List[str]:
        context = self.memory.load_memory_variables({})["chat_history"]
        enhanced_query = f"{query}\nContext:\n{context}"
        
        docs = self.retriever.get_relevant_documents(enhanced_query)
        if not docs:
            raise ValueError("No relevant documents found in knowledge base")
        return [doc.page_content for doc in docs]

class ValidatorAgent:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
    
    def validate(self, query: str, solutions: List[str]) -> Tuple[bool, List[str], List[str]]:
        prompt = f"""
        Analyze these solutions for: "{query}"
        
        Available solutions:
        {solutions}
        
        Respond in this EXACT format:
        {{
            "complete_solutions": ["list", "of", "complete", "solutions"],
            "incomplete_solutions": ["list", "of", "incomplete", "solutions"],
            "missing_details": ["list", "of", "questions"]
        }}
        """
        try:
            response = self.llm.invoke(prompt).content
            start = response.find("{")
            end = response.rfind("}") + 1
            result = json.loads(response[start:end])
            is_complete = not bool(result["incomplete_solutions"])
            return is_complete, result["complete_solutions"], result["missing_details"]
        except Exception as e:
            return True, solutions, []

class DialogAgent:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
    
    def ask_clarification(self, missing_details: List[str]) -> str:
        prompt = f"""
        Conversation history:
        {self.memory.load_memory_variables({})["chat_history"]}
        
        Ask ONE concise follow-up question to get these details:
        {missing_details}
        
        Phrase it naturally as an IT support specialist would.
        """
        return self.llm.invoke(prompt).content

class SolverAgent:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "history"],
            template="""
            # IT Helpdesk Solution
            ## Knowledge Context:
            {context}
            
            ## Conversation History:
            {history}
            
            ## User Query:
            {question}
            
            ## Requirements:
            - Use ONLY the provided context
            - Format as numbered steps
            - Include code examples where applicable
            - If context is insufficient, state that clearly
            
            ## Solution:
            """
        )
    
    def solve(self, query: str, context: List[str]) -> str:
        history = self.memory.load_memory_variables({})["chat_history"]
        context_str = "\n".join(context)
        prompt = self.prompt_template.format(
            context=context_str,
            question=query,
            history=history
        )
        return self.llm.invoke(prompt).content

