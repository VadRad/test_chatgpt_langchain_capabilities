import streamlit as st
import logging
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
)
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app
st.title("LangChain Chat with Memory")
st.write("Welcome to the LangChain Chat! Type your message below:")

# Sidebar for OpenAI API key input
st.sidebar.title("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if openai_api_key:
    try:
        # Initialize the recent conversation memory
        conv_memory = ConversationBufferMemory(
            memory_key="chat_history_lines", input_key="input"
        )

        # Initialize the summary memory
        summary_memory = ConversationSummaryMemory(
            llm=OpenAI(api_key=openai_api_key),
            input_key="input"
        )

        # Combined memory
        memory = CombinedMemory(memories=[conv_memory, summary_memory])

        # Define the prompt template
        _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Summary of conversation:
        {history}
        Current conversation:
        {chat_history_lines}
        Human: {input}
        AI:"""
        
        PROMPT = PromptTemplate(
            input_variables=["history", "input", "chat_history_lines"],
            template=_DEFAULT_TEMPLATE,
        )

        # Create the conversation chain
        llm = OpenAI(api_key=openai_api_key, temperature=0)
        conversation = ConversationChain(llm=llm, verbose=True, memory=memory, prompt=PROMPT)

        # Chat history
        if 'history' not in st.session_state:
            st.session_state.history = []

        # Input from user
        user_input = st.text_input("You:", key="input")

        # Generate response and update history
        if st.button("Send"):
            if user_input:
                try:
                    response = conversation.predict(input=user_input)
                    st.session_state.history.append(f"You: {user_input}")
                    st.session_state.history.append(f"Bot: {response}")
                    st.session_state.input = ""
                    logger.info("Successfully generated response from AI.")
                except Exception as e:
                    if "insufficient_quota" in str(e):
                        st.error("You have exceeded your OpenAI quota. Please check your plan and billing details.")
                    else:
                        st.error("An error occurred while generating the response.")
                    logger.error(f"Error during prediction: {e}")

        # Display chat history
        if st.session_state.history:
            for chat in st.session_state.history:
                st.write(chat)

        if st.button("Clear Chat"):
            st.session_state.history = []
            conv_memory.clear()
            summary_memory.clear()
            logger.info("Chat history and memories cleared.")

    except Exception as e:
        st.error("An error occurred during initialization. Please check your API key and try again.")
        logger.error(f"Initialization error: {e}")
else:
    st.write("Please enter your OpenAI API Key in the sidebar to start chatting.")
