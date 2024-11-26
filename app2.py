import chainlit as cl
from langchain.llms import Ollama

# Constants
OLLAMA_MODEL = "llama3.2"  # Replace with the specific model name if different

@cl.on_message
async def main(message):
    try:
        query = message.content.strip()

        # Predefined responses for simple inputs
        # PREDEFINED_RESPONSES = {
        #     "hello": "Hi there! How can I assist you?",
        #     "hi": "Hello! How can I help you today?",
        #     "hey": "Hey! What can I do for you?",
        #     "how are you": "I'm just a program, but I'm ready to help! How can I assist you?",
        # }

        # if query.lower() in PREDEFINED_RESPONSES:
        #     await cl.Message(content=PREDEFINED_RESPONSES[query.lower()]).send()
        #     return

        # Directly query the LLM without Vector DB
        llm = Ollama(model=OLLAMA_MODEL)
        answer = llm(prompt=query)  # Pass the query as the prompt

        # Send the LLM response
        await cl.Message(content=f"{answer}").send()

    except Exception as e:
        # Handle exceptions gracefully
        await cl.Message(content=f"An error occurred: {str(e)}", author="system").send()
