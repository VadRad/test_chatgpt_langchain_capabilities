import streamlit as st
from langchain.memory import ConversationBufferMemory, SummaryMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'

# Initialize the recent conversation memory
recent_memory = ConversationBufferMemory()

# Initialize the summary memory
summary_memory = SummaryMemory()

# Create a conversation chain with both memories
conversation = ConversationChain(
    llm=OpenAI(),
    memory={
        'recent': recent_memory,
        'summary': summary_memory
    }
)

# Streamlit app
st.title("LangChain Chat with Memory")
st.write("Welcome to the LangChain Chat! Type your message below:")

# Chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input from user
user_input = st.text_input("You:", key="input")

# Generate response and update history
if st.button("Send"):
    if user_input:
        response = conversation.predict(input=user_input)
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Bot: {response}")
        st.session_state.input = ""

# Display chat history
if st.session_state.history:
    for chat in st.session_state.history:
        st.write(chat)

if st.button("Clear Chat"):
    st.session_state.history = []
    recent_memory.clear()
    summary_memory.clear()
