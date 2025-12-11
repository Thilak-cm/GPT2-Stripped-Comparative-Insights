import os
import time

import streamlit as st

from replicate_backend import call_replicate


# Title of the app
st.title("Our 848K project: GPT-2 Unveiled: Comparative Insights")

# Sidebar for model selection
st.sidebar.header("Select a Model")
model_option = st.sidebar.selectbox(
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

# Add authors' names and links at the bottom
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            width: 100%;
            text-align: center;
            padding: 12px 10px;
            background-color: #f9f9f9;
            font-size: 13px;
            font-family: Arial, sans-serif;
            color: #333333;
            box-shadow: 0px -1px 5px rgba(0, 0, 0, 0.1);
        }
        .footer a {
            text-decoration: none;
            margin: -55px;
            margin-left: 10px;
            color: #4F8BF9;
            font-weight: bold;
        }
        .footer i {
            margin-right: 50px;
        }
    </style>
    <div class="footer">
        <b>Authors:</b> 
        Thilak Mohan 
        <a href="https://www.linkedin.com/in/thilak-mohan-687b801b2/" target="_blank">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://github.com/Thilak-cm" target="_blank">
            <i class="fab fa-github"></i>
        </a> |
        Sumedha Vadlamani 
        <a href="https://www.linkedin.com/in/sumedha-vadlamani/" target="_blank">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://github.com/sumedha-24" target="_blank">
            <i class="fab fa-github"></i>
        </a> |
        Peeyush Dyavarashetty 
        <a href="https://www.linkedin.com/in/peeyush-dyavarashetty/" target="_blank">
            <i class="fab fa-linkedin"></i>
        </a>
        <a href="https://github.com/Peeyush4" target="_blank">
            <i class="fab fa-github"></i>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

api_token_present = bool(os.getenv("REPLICATE_API_TOKEN"))
if not api_token_present:
    st.sidebar.error("Missing REPLICATE_API_TOKEN. Add it in Streamlit secrets or environment.")
else:
    st.sidebar.success("Using Replicate for inference.")

st.markdown("### Chat")

# Ensure chat history is properly initialized
if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []

# Render chat transcript in a chat-style layout
for message in st.session_state.chat_history:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["content"])

prompt = st.chat_input("Message Kerple (or choose another PE)...", disabled=not api_token_present)

if prompt and model_option and api_token_present:
    # Show user message immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare full conversation for the model
    full_input = "\n".join(
        ["User: " + m["content"] if m["role"] == "user" else "Assistant: " + m["content"]
         for m in st.session_state.chat_history]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking on Replicate..."):
            try:
                start = time.time()
                response = call_replicate(full_input, model_option, max_new_tokens=50)
                generation_time = time.time() - start
            except Exception as exc:
                st.error(f"Failed to generate response: {exc}")
                st.stop()

            st.markdown(response)
            st.caption(f"Generated in {generation_time:.2f}s via Replicate")

    # Save assistant message separately so we don't echo the user prompt
    st.session_state.chat_history.append({"role": "assistant", "content": response})
        
