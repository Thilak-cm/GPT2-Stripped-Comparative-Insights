import os
import time

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
    .disclaimer-banner {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: #f0f2f6;
        border-bottom: 1px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 0.875rem;
        color: #333;
        z-index: 999;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .disclaimer-banner strong {
        color: #666;
    }
    .main .block-container {
        padding-top: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Fixed disclaimer banner
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Fair Warning:</strong> These models were trained on 4 A100 GPUs for 2 days, so keep your expectations... modest. Built purely for educational purposes—I get to dig deep into PyTorch implementations and learn firsthand. Functionally useless? Maybe. Incredibly valuable for learning? Absolutely.
    </div>
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
        <a href="https://thilak-cm.github.io/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Thilak Mohan</a> | 
        <a href="https://www.linkedin.com/in/sumedha-vadlamani/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Sumedha Vadlamani</a> | 
        <a href="https://www.linkedin.com/in/peeyush-dyavarashetty/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Peeyush Dyavarashetty</a>
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
        
