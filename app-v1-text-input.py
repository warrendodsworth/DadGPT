from dotenv import find_dotenv, load_dotenv
import os
import streamlit as st
from huggingface_hub import InferenceClient

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Smush Date", page_icon="♥️")

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


# Call the API
def model_api(user_input: str, system_message: str):
    # model = "google/gemma-2-2b-it"
    client = InferenceClient(api_key=HUGGINGFACEHUB_API_TOKEN)
    messages = [
        {"role": "user", "content": user_input},
        {"role": "system", "content": system_message},
    ]
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct", messages=messages, max_tokens=500
    )
    # print(completion.choices[0].message)
    return completion.choices[0].message.content


# Chat function
def generate_response(user_input):
    st.session_state.is_processing = True  # Set processing state to True

    # Simulate loading by showing spinner in place of the text box
    with input_container:
        st.markdown("💬 Thinking...")

    # Call the API
    conversation_history = "".join(
        f"User: {chat['user']}\nBot: {chat['bot']}\n"
        for chat in st.session_state.history
    )
    prompt = f"Conversation History: {conversation_history}. User:{user_input}"
    response = model_api(prompt, system_message=system_message)
    st.session_state.history.append({"user": user_input, "bot": response})
    st.session_state.is_processing = False  # Reset processing state


# Function to clear user input
def clear_input():
    st.session_state.user_input = ""


# Sidebar for chat settings
with st.sidebar:
    st.sidebar.header("Chat Settings")
    if st.sidebar.button("Reset Chat"):
        st.session_state.history = []
        st.session_state.user_input = ""

# Display chat history
st.title("Smush Date - Relationship Chatbot")
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")

# Container for user input and spinner
input_container = st.empty()

if st.session_state.is_processing is False:
    # Display the text box
    with input_container:
        st.text_input(
            "Talk to me",
            key="user_input",
            on_change=lambda: (
                generate_response(st.session_state.user_input),
                clear_input(),
            ),
        )
