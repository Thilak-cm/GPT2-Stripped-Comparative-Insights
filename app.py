import os
import time

import streamlit as st

from replicate_backend import call_replicate


st.set_page_config(page_title="848K GPT-2 Chat", layout="wide")

# Custom CSS for ChatGPT-like interface
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem 0;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] {
        padding: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for model selection
with st.sidebar:
    st.title("848K GPT-2")
    st.markdown("**GPT-2 Unveiled: Comparative Insights**")
    st.divider()
    
    model_option = st.selectbox(
        "Choose a Positional Encoding:",
        [
            "Kerple",
            "ALIBI",
            "FIRE",
            "Learned PE",
            "RoPE",
            "Sinusoidal",
        ],
        index=0,
    )
    
    st.divider()
    
    api_token_present = bool(os.getenv("REPLICATE_API_TOKEN"))
    if not api_token_present:
        st.error("⚠️ Backend not connected")
    else:
        st.success("✓ Backend connected")
    
    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #666;">
        <b>Authors:</b><br>
        Thilak Mohan | Sumedha Vadlamani | Peeyush Dyavarashetty
        </div>
        """,
        unsafe_allow_html=True
    )

# Ensure chat history is properly initialized
if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []
    st.session_state.first_message = True

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show welcome message only if no chat history
if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        st.markdown(f"Hello! I'm GPT-2 with **{model_option}** positional encoding. How can I help you today?")

# Chat input
prompt = st.chat_input("Message...", disabled=not api_token_present)

if prompt and model_option and api_token_present:
    # Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Clear the welcome message flag
    st.session_state.first_message = False

    # Prepare input for the model
    full_input = "\nUser: " + prompt

    # Show assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner(""):
            try:
                start = time.time()
                response = call_replicate(full_input, model_option, max_new_tokens=50)
                generation_time = time.time() - start
                
                # Display response
                message_placeholder.markdown(response)
                
            except Exception as exc:
                st.error(f"Failed to generate response: {exc}")
                response = None

    # Save assistant message to history
    if response:
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
