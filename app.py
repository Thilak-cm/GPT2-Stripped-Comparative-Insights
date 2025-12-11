import os
import time

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from tiktoken import get_encoding

# Import from other scripts
from chat_with_model import GPTConfig, generate_response
from model_architectures.alibi_arch import alibi_GPT
from model_architectures.fire_arch import fire_GPT
from model_architectures.kerple_arch import kerple_GPT
from model_architectures.learnedPE_arch import learned_pe_GPT
from model_architectures.rope_arch import rope_GPT
from model_architectures.sinusoidal_arch import sinusoidal_GPT
from replicate_backend import call_replicate


repo_id = "thillsss/848k-models"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Title of the app
st.title("Our 848K project: GPT-2 Unveiled: Comparative Insights")

# Sidebar for model selection
st.sidebar.header("Select a Model")
model_option = st.sidebar.selectbox(
    "Choose a Positional Encoding:",
    [
        "ALIBI",
        "FIRE",
        "Kerple",
        "Learned PE",
        "RoPE",
        "Sinusoidal",
    ],
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

model_mapping = {
    "ALIBI": (alibi_GPT, "final_alibi_model.pth"),
    "FIRE": (fire_GPT, "final_fire_model.pth"),
    "Kerple": (kerple_GPT, "final_kerple_model.pth"),
    "Learned PE": (learned_pe_GPT, "final_learned_pe_model.pth"),
    "RoPE": (rope_GPT, "final_rope_model.pth"),
    "Sinusoidal": (sinusoidal_GPT, "final_sinusoidal_model.pth"),
}

# Backend selection
has_replicate = bool(os.getenv("REPLICATE_API_TOKEN"))
backend_options = ["Replicate (remote)"] if has_replicate else []
backend_options.append("Local (torch)")
backend_choice = st.sidebar.radio(
    "Inference backend:", backend_options, index=0 if has_replicate else len(backend_options) - 1
)
use_remote = backend_choice.startswith("Replicate")

if not has_replicate:
    st.sidebar.info("Set REPLICATE_API_TOKEN to enable remote inference.")

# Cache for local models
if "local_model_cache" not in st.session_state:
    st.session_state.local_model_cache = {}


def load_local_model(selected_model):
    if selected_model in st.session_state.local_model_cache:
        return st.session_state.local_model_cache[selected_model]

    model_class, model_path = model_mapping[selected_model]
    tokenizer = get_encoding("gpt2")
    config = GPTConfig(vocab_size=50304)
    model = model_class(config).to(device)
    model_file_path = hf_hub_download(repo_id=repo_id, filename=model_path)

    state_dict = torch.load(model_file_path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", "").replace("module._orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    st.session_state.local_model_cache[selected_model] = (model, tokenizer)
    return model, tokenizer


# Load model on demand when local backend is selected
if model_option:
    st.sidebar.write(f"Selected Model: {model_option}")
    st.sidebar.write(f"Inference: {'Replicate (remote)' if use_remote else 'Local (torch)'}")
    if not use_remote and model_option not in st.session_state.local_model_cache:
        with st.spinner("Loading model locally..."):
            load_local_model(model_option)
        st.success("Model loaded locally.")

# Ensure chat history is properly initialized
if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []

st.subheader("Chat")
user_input = st.text_input("Enter your message:")

if user_input and model_option:
    with st.spinner("Generating response..."):
        # Append user input as a separate entry
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Combine chat history into a clean model input
        full_input = "\n".join(
            [f"User: {m['content']}" if m['role'] == "user" else f"Model: {m['content']}"
             for m in st.session_state.chat_history]
        )

        try:
            if use_remote:
                start = time.time()
                response = call_replicate(full_input, model_option, max_new_tokens=50)
                generation_time = time.time() - start
            else:
                model, tokenizer = load_local_model(model_option)
                response, generation_time = generate_response(model, tokenizer, full_input)
        except Exception as exc:
            st.error(f"Failed to generate response: {exc}")
            st.stop()

        st.write(f"Response generated in {generation_time:.2f} seconds")

        # Append model response as a separate entry
        st.session_state.chat_history.append({"role": "model", "content": response})

        # Display chat history in alternating format
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**Model:** {message['content']}")
        
