import streamlit as st
from langchain import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Set up Streamlit
st.title("LangChain Chatbot")
st.write("This is a simple chatbot using LangChain and Streamlit.")

# Initialize the chat model
openai_api_key = "your-openai-api-key"
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
    st.text_area("Chatbot:", value=response.content, height=200, max_chars=None, key=None)
