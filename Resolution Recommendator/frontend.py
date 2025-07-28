import streamlit as st
import requests
from typing import Dict, List

# Configuration
FASTAPI_URL = "http://127.0.0.1:8000/chat"  # Update if your API is hosted elsewhere

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False

# UI Setup
st.title("🤖 IT Helpdesk Assistant")
st.caption("Powered by 4-Agent AI System")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Single chat input that adapts to context
if st.session_state.awaiting_clarification:
    input_label = "Provide the requested details..."
else:
    input_label = "Describe your IT issue..."

if prompt := st.chat_input(input_label):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare API request based on context
    if st.session_state.awaiting_clarification:
        # For follow-up responses, combine with original query
        original_query = next(
            (msg["content"] for msg in reversed(st.session_state.messages) 
             if msg["role"] == "user" and "assistant" in [m["role"] for m in st.session_state.messages[st.session_state.messages.index(msg):]]),
            prompt
        )
        payload = {
            "question": f"{original_query}\nAdditional details: {prompt}",
            "history": st.session_state.messages
        }
        st.session_state.awaiting_clarification = False
    else:
        # For new queries
        payload = {
            "question": prompt,
            "history": st.session_state.messages
        }
    
    # Show typing indicator
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("⌛ Processing...")
        
        # Call FastAPI endpoint
        try:
            response = requests.post(
                FASTAPI_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response_data = response.json()
            
            if response.status_code == 200:
                if response_data.get("follow_up_needed", False):
                    # Handle clarification flow
                    st.session_state.awaiting_clarification = True
                    status_placeholder.markdown(response_data["answer"])
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_data["answer"]
                    })
                else:
                    # Display complete response
                    status_placeholder.markdown(response_data["answer"])
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_data["answer"]
                    })
            else:
                status_placeholder.error(f"API Error: {response_data.get('error', 'Unknown error')}")
                
        except Exception as e:
            status_placeholder.error(f"Connection failed: {str(e)}")

