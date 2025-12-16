import os
import time
import threading

import streamlit as st

from replicate_backend import call_replicate


def filter_user_message_from_response(response: str, user_prompt: str) -> str:
    """Remove the echoed user message from the LLM response."""
    # Remove common patterns where the model echoes the user's message
    patterns_to_remove = [
        f"User: {user_prompt}",
        f"\nUser: {user_prompt}",
        f"User: {user_prompt}\n",
        f"\nUser: {user_prompt}\n",
    ]
    
    cleaned_response = response
    for pattern in patterns_to_remove:
        if cleaned_response.startswith(pattern):
            cleaned_response = cleaned_response[len(pattern):].strip()
            break
    
    # Also try to find and remove if it appears anywhere in the response
    for pattern in patterns_to_remove:
        cleaned_response = cleaned_response.replace(pattern, "").strip()
    
    return cleaned_response


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

# Check API token (needed for cold start)
api_token_present = bool(os.getenv("REPLICATE_API_TOKEN"))

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
    
    if not api_token_present:
        st.error("⚠️ Backend not connected")
    else:
        st.success("✓ Backend connected")
    
    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #666;">
        <b>Authors:</b><br>
        <a href="https://thilak-cm.github.io/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Thilak Mohan</a> | 
        <a href="https://www.linkedin.com/in/sumedha-vadlamani/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Sumedha Vadlamani</a> | 
        <a href="https://www.linkedin.com/in/peeyush-dyavarashetty/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Peeyush Dyavarashetty</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Disclaimer banner using Streamlit's native component
st.info("⚠️ **Fair Warning:** These models were trained on 4 A100 GPUs for 2 days, so keep your expectations... modest. Built purely for educational purposes—I get to dig deep into PyTorch implementations and learn firsthand. Functionally useless? Maybe. Incredibly valuable for learning? Absolutely.")

# Ensure chat history is properly initialized
if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []
    st.session_state.first_message = True

# Cold start: Warm up the model in the background when app first loads
if "cold_start_done" not in st.session_state:
    st.session_state.cold_start_done = False

def warm_up_model(model_name: str):
    """Background function to warm up the model with a test message."""
    try:
        test_prompt = "\nUser: test message"
        call_replicate(test_prompt, model_name, max_new_tokens=10)
    except Exception:
        # Silently fail - this is just for warming up
        pass

# Trigger cold start once when app loads (if API token is present)
# Note: This runs after sidebar is rendered, so model_option is available
if not st.session_state.cold_start_done and api_token_present:
    st.session_state.cold_start_done = True
    # Run warm-up in background thread to avoid blocking UI
    # Use the currently selected model (defaults to Kerple)
    threading.Thread(target=warm_up_model, args=(model_option,), daemon=True).start()

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
                raw_response = call_replicate(full_input, model_option, max_new_tokens=50)
                generation_time = time.time() - start
                
                # Filter out the user's message from the response
                response = filter_user_message_from_response(raw_response, prompt)
                
                # Display cleaned response
                message_placeholder.markdown(response)
                
            except Exception as exc:
                st.error(f"Failed to generate response: {exc}")
                response = None

    # Save assistant message to history (only the cleaned response)
    if response:
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
