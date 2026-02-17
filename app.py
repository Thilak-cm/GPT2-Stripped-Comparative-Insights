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


def sanitize_model_output(text: str) -> str:
    """Clean up raw model output for display."""
    cleaned = text.replace("<|endoftext|>", ".").strip()
    return cleaned if cleaned else "..."


st.set_page_config(page_title="Thilak's LLM", layout="wide")

# Custom CSS for polished interface
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 52rem;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem 0;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] {
        padding: 0.5rem 0;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
    }
    .sidebar-metric {
        background: rgba(79, 139, 249, 0.08);
        border: 1px solid rgba(79, 139, 249, 0.15);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    .sidebar-metric b {
        color: #4F8BF9;
        letter-spacing: 0.02em;
    }
    /* Disclaimer accent */
    .stAlert {
        border-left: 4px solid #f0ad4e;
    }
    /* Typography refinements */
    [data-testid="stSidebar"] label {
        font-size: 0.85rem;
        letter-spacing: 0.03em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Check API token (needed for cold start)
api_token_present = bool(os.getenv("REPLICATE_API_TOKEN"))

# Sidebar — recruiter-facing project card
with st.sidebar:
    st.title("An LLM from Scratch")
    st.caption("CMSC 848K | University of Maryland")
    st.divider()

    st.markdown(
        "**What is this?** A ~128M-parameter LLM following GPT-2 style architecture, built entirely "
        "from scratch — no pre-trained weights, no fine-tuning shortcuts. Custom PyTorch "
        "implementation with distributed data-parallel training across 4 A100 GPUs, "
        "deployed on Replicate with a Streamlit frontend."
    )

    st.divider()
    st.markdown("**Technical Highlights**")

    st.markdown(
        '<div class="sidebar-metric"><b>Architecture:</b> GPT-2 style &middot; 12 layers &middot; 12 heads &middot; 768 dim</div>'
        '<div class="sidebar-metric"><b>Parameters:</b> ~128M (trained from scratch)</div>'
        '<div class="sidebar-metric"><b>Training:</b> 4&times; A100 GPUs &middot; ~2 days</div>'
        '<div class="sidebar-metric"><b>Dataset:</b> FineWeb-Edu (HuggingFace)</div>'
        '<div class="sidebar-metric"><b>Stack:</b> PyTorch &middot; DDP &middot; tiktoken</div>',
        unsafe_allow_html=True
    )

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
        st.error("Backend not connected")
    else:
        st.success("Backend connected")

    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #888;">
        <a href="https://github.com/Thilak-cm/848k-project" target="_blank" style="color: #4F8BF9; text-decoration: none;">GitHub Repo</a><br><br>
        <b>Authors:</b><br>
        <a href="https://thilak-cm.github.io/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Thilak Mohan</a> |
        <a href="https://www.linkedin.com/in/sumedha-vadlamani/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Sumedha Vadlamani</a> |
        <a href="https://www.linkedin.com/in/peeyush-dyavarashetty/" target="_blank" style="color: #4F8BF9; text-decoration: none;">Peeyush Dyavarashetty</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Disclaimer banner
st.warning(
    "These models were trained on 4 A100 GPUs for 2 days, so keep your expectations... modest. "
    "Built purely for educational purposes — I get to dig deep into PyTorch implementations and learn firsthand. "
    "Functionally useless? Maybe. Incredibly valuable for learning? Absolutely.",
    icon="🔬"
)

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
        st.markdown(
            f"You're chatting with a **128M-parameter LLM** using **{model_option}** "
            "positional encoding — built entirely from scratch.\n\n"
            "Try completing a sentence, say hello, or switch positional encodings "
            "in the sidebar to compare outputs."
        )

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
                # Sanitize model output (remove <|endoftext|>, etc.)
                response = sanitize_model_output(response)

                # Display cleaned response
                message_placeholder.markdown(response)

            except Exception as exc:
                st.error(f"Failed to generate response: {exc}")
                response = None

    # Save assistant message to history (only the cleaned response)
    if response:
        st.session_state.chat_history.append({"role": "assistant", "content": response})
