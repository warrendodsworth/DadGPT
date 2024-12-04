import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import find_dotenv, load_dotenv
import gradio as gr

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Load the Hugging Face model for conversational AI (e.g., GPT-2, GPT-3 fine-tuned)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2"
    )  # Can replace with another conversational model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    nlp_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return nlp_pipeline


# Add context for the chatbot to be a good listener
system_message = """You are a relationship expert who listens carefully before providing advice. 
You ask thoughtful, open-ended questions to understand the user's situation better. 
Encourage the user to reflect on their own feelings and experiences before offering guidance. 
You focus on being empathetic and patient, helping the user arrive at their own conclusions."""

# Initialize history to store chat conversations
history = []

# Load the model
model_pipeline = load_model()


def chatbot(user_input):
    global history

    # If it's the first message, prepend the system message
    if len(history) == 0:
        full_input = f"{system_message}\nUser: {user_input}\nBot:"
    else:
        conversation_history = ""
        for chat in history:
            conversation_history += f"User: {chat[0]}\nBot: {chat[1]}\n"
        full_input = conversation_history + f"User: {user_input}\nBot:"

    # Generate response from the model
    response = model_pipeline(full_input, max_new_tokens=150, num_return_sequences=1)[
        0
    ]["generated_text"]

    # Remove the user input from the response to avoid repetition
    response = response.replace(user_input, "").strip()

    # Update history with the correct format (list of length 2)
    history.append([user_input, response])
    print("USER:: ", user_input)
    return response


# Define the Gradio interface
def reset_history():
    global history
    history = []  # Reset history to an empty list


with gr.Blocks() as demo:
    # Textbox for user input
    user_input = gr.Textbox(
        label="Ask something", placeholder="How can I get a date with a woman?", lines=1
    )

    # Button to clear history
    clear_button = gr.Button("Clear History")

    # Display chat history
    chat_history = gr.Chatbot()

    # Set up interaction: user input triggers the chatbot function and updates chat history
    user_input.submit(chatbot, inputs=user_input, outputs=chat_history)
    clear_button.click(reset_history, inputs=None, outputs=chat_history)

# Launch the app
demo.launch()
