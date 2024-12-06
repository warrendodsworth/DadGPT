from dotenv import find_dotenv, load_dotenv
import os
import streamlit as st
from huggingface_hub import InferenceClient

# Set the page configuration
st.set_page_config(page_title="SmushGPT", page_icon="♥️")

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Add context for the chatbot to be a good listener
system_message = """You are a relationship expert who listens carefully before providing advice. 
You ask thoughtful, open-ended questions to understand the user's situation better. 
Encourage the user to reflect on their own feelings and experiences before offering guidance. 
You focus on being empathetic and patient, helping the user arrive at their own conclusions.

Smush is a playful and affectionate term for a really tight, cozy hug. 
"""

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []


# Call the API
def model_api(user_input: str, system_message: str):
    client = InferenceClient(api_key=HUGGINGFACEHUB_API_TOKEN)
    messages = [
        {"role": "user", "content": user_input},
        {"role": "system", "content": system_message},
    ]
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct", messages=messages, max_tokens=500
    )
    return completion.choices[0].message.content


# Chat interface using st.chat_message and st.chat_input
st.title("Smush Dating & Relationship Chatbot")

# Display chat history
for chat in st.session_state.history:
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        st.markdown(chat["content"])

# User input section
if user_input := st.chat_input("Talk to me"):
    # Append user message to chat history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Display the user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate bot response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # Reserve space for the bot response
        response_placeholder.markdown("⌛ Thinking...")
        response = model_api(user_input, system_message)

        # Replace placeholder with the actual response
        response_placeholder.markdown(response)

    # Append bot response to chat history
    st.session_state.history.append({"role": "assistant", "content": response})
