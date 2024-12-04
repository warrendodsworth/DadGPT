from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import find_dotenv, load_dotenv
import os
import streamlit as st
import torch

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Smush Date", page_icon="♥️")

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
device = 0 if torch.backends.mps.is_available() else -1


# Load the Hugging Face model for conversational AI (e.g., GPT-2, GPT-3 fine-tuned)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
    )
    nlp_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        token=HUGGINGFACEHUB_API_TOKEN,
        device=device,
    )
    return nlp_pipeline


model_pipeline = load_model()

# Add context for the chatbot to be a good listener
system_message = """You are a relationship expert who listens carefully before providing advice. 
You ask thoughtful, open-ended questions to understand the user's situation better. 
Encourage the user to reflect on their own feelings and experiences before offering guidance. 
You focus on being empathetic and patient, helping the user arrive at their own conclusions."""

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# Function to clear user input
def clear_input():
    st.session_state.user_input = ""


# Chat function
def generate_response(user_input):
    conversation_history = "".join(
        f"User: {chat['user']}\nBot: {chat['bot']}\n"
        for chat in st.session_state.history
    )
    prompt = f"{system_message}\n{conversation_history}User: {user_input}\nBot:"
    response = model_pipeline(prompt, max_new_tokens=150, num_return_sequences=1)[0][
        "generated_text"
    ]
    response = response.replace(prompt, "").strip()  # Clean up the response
    st.session_state.history.append({"user": user_input, "bot": response})


# Sidebar for chat settings
with st.sidebar:
    st.sidebar.header("Chat Settings")
    if st.sidebar.button("Reset Chat"):
        # Clear chat history and user input when the reset button is clicked
        st.session_state.history = []  # Clear the history
        st.session_state.user_input = ""  # Clear user input

# Display chat history
st.title("Smush Date - Relationship Chatbot")
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")

# User input section
st.text_input(
    "Type your message:",
    key="user_input",
    on_change=lambda: (generate_response(st.session_state.user_input), clear_input()),
)
