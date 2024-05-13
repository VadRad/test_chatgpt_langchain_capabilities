import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Set up Streamlit
st.title("LangChain Chatbot")
st.write("This is a simple chatbot using LangChain and Streamlit.")

# Sidebar for OpenAI API Key
st.sidebar.title("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if openai_api_key:
    # Initialize the chat model
    chat_model = ChatOpenAI(api_key=openai_api_key)

    # Initialize conversation memory
    memory = ConversationBufferMemory()

    # Set up the prompt template
    template = PromptTemplate(
        template="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\n{history}\n\nHuman: {input}\nAI:",
        input_variables=["history", "input"]
    )

    # Initialize the conversation chain
    conversation = ConversationChain(
        llm=chat_model,
        prompt=template,
        memory=memory
    )

    # Chat input
    user_input = st.text_input("You: ", "")

    if user_input:
        response = conversation.predict(input=user_input)
        st.text_area("Chatbot:", value=response, height=200, max_chars=None, key=None)
else:
    st.write("Please enter your OpenAI API key in the sidebar to start the chatbot.")
