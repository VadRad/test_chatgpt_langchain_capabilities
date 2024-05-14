import os
import json
import random
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# App config
st.set_page_config(page_title="D&D Master Chatbot", page_icon="🧙‍♂️")
st.title("D&D Master Chatbot")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def get_response(user_query, chat_history, players):
    template = """
    You are a Dungeon Master for a freeform role-playing. Respond to the user's queries and actions, considering the history of the conversation, the list of players, and the context of a D&D game. Keep the game engaging, challenging, and fun:

    Chat history: {chat_history}

    Players: {players}

    User action or question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4o")

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "players": players,
        "user_question": user_query,
    })


def generate_initial_prompt(players):
    template = """
    You are a Dungeon Master for a freeform role-playing. The following players have joined the game:
    {players}
    Create an engaging and exciting start to the adventure considering the players' names and their potential roles in the game.
    Let it be short. Greet players and create a hook
    You describe it to a players
    Ask them that the want to do next
    """

    player_names = ", ".join(players)
    prompt = ChatPromptTemplate.from_template(template.format(players=player_names))
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "players": player_names,
    })


# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "players" not in st.session_state:
    st.session_state.players = []
if "game_started" not in st.session_state:
    st.session_state.game_started = False

# Registration form
if not st.session_state.game_started:
    with st.form("registration_form"):
        player_name = st.text_input("Enter your name:")
        submitted = st.form_submit_button("Join the game")
        if submitted and player_name:
            st.session_state.players.append(player_name)
    if len(st.session_state.players) >= 2:  # Minimum 2 players to start the game
        if st.button("Start the Game"):
            st.session_state.game_started = True


# Game conversation
if st.session_state.game_started:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    if not st.session_state.chat_history:
        with st.chat_message("AI"):
            initial_prompt = st.write_stream(generate_initial_prompt(st.session_state.players))
        st.session_state.chat_history.append(AIMessage(content=initial_prompt))
    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = st.write_stream(get_response(user_query, st.session_state.chat_history, st.session_state.players))

        st.session_state.chat_history.append(AIMessage(content=response))


# Display the list of players on the right (sidebar)
st.sidebar.title("Players")
for player in st.session_state.players:
    st.sidebar.write(player)
