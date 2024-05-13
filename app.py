import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Streamlit sidebar for OpenAI API key input
st.sidebar.title("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Check if API key is provided
if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
else:
    # Initialize the ChatOpenAI model
    chat_model = ChatOpenAI(api_key=openai_api_key, model="gpt-4")

    # Streamlit interface for chat
    st.title("Chat with GPT-4")
    st.write("A simple chat interface with GPT-4 using LangChain.")

    # Input text box for user
    user_input = st.text_input("You: ")

    if user_input:
        # Create a prompt template
        prompt_template = PromptTemplate.from_template("You: {input}\nAI:")
        prompt = prompt_template.format(input=user_input)

        # Get the response from the chat model
        response = chat_model(prompt)
        st.write(f"AI: {response}")
