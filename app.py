import os
import json
import random
import streamlit as st
import logging
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# App config
st.set_page_config(page_title="Adventure Master Chatbot", page_icon="🧙‍♂️")
st.title("Adventure Master Chatbot")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

logger.info("App started and page configured.")

def get_response(user_query, chat_history, players):
    logger.info("Generating response for user query.")
    template = """
    You are a Dungeon Master for a freeform role-playing. Respond to the user's queries and actions, considering the history of the conversation, the list of players, and the context of a D&amp;D game. Keep the game engaging, challenging, and fun:

    You receive input from each player

    Chat history: {chat_history}

    Players: {players}

    User action or question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm | StrOutputParser()

    response = chain.stream({
        "chat_history": chat_history,
        "players": players,
        "user_question": user_query,
    })

    logger.info("Response generated.")
    return response

def generate_initial_prompt(players):
    logger.info("Generating initial prompt.")
    template = """
    You are a Dungeon Master for a freeform role-playing. The following players have joined the game:
    {players}
    Create an engaging and exciting start to the adventure considering the players' names and their potential roles in the game.
    Let it be short. Greet players and create a hook
    You describe it to a players
    Ask them what they want to do next
    """

    player_names = ", ".join(players)
    prompt = ChatPromptTemplate.from_template(template.format(players=player_names))
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm | StrOutputParser()

    initial_prompt = chain.stream({
        "players": player_names,
    })

    logger.info("Initial prompt generated.")
    return initial_prompt

def increment_turn():
    logger.info("Incrementing turn.")
    current_player_name = st.session_state.players[st.session_state.current_player]
    current_player = st.session_state.current_player
    st.session_state.player_inputs[current_player_name] = st.session_state[f'input_{current_player_name}_{current_player}']
    st.session_state.current_player += 1
    logger.info(f"Turn incremented to {st.session_state.current_player}.")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Initialized chat history.")
if "players" not in st.session_state:
    st.session_state.players = []
    logger.info("Initialized players list.")
if "game_started" not in st.session_state:
    st.session_state.game_started = False
    logger.info("Initialized game_started flag.")
if "current_player" not in st.session_state:
    st.session_state.current_player = 0
    logger.info("Initialized current_player index.")
if "player_inputs" not in st.session_state:
    st.session_state.player_inputs = {}
    logger.info("Initialized player_inputs dictionary.")
if "awaiting_input" not in st.session_state:
    st.session_state.awaiting_input = True
    logger.info("Initialized awaiting_input flag.")

# Registration form
if not st.session_state.game_started:
    with st.form("registration_form"):
        player_name = st.text_input("Enter your name:")
        submitted = st.form_submit_button("Join the game")
        if submitted and player_name:
            st.session_state.players.append(player_name)
            logger.info(f"Player {player_name} joined the game.")
    if len(st.session_state.players) >= 2:  # Minimum 2 players to start the game
        if st.button("Start the Game"):
            st.session_state.game_started = True
            logger.info("Game started with players: " + ", ".join(st.session_state.players))

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
        logger.info("Initial prompt added to chat history.")

    if st.session_state.current_player < len(st.session_state.players):
        # User input
        current_player_name = st.session_state.players[st.session_state.current_player]
        with st.chat_message("AI"):
            st.markdown(f"{current_player_name}, what do you want to do?")

        user_input = st.chat_input("Type your message here...",
                                    key=f"input_{current_player_name}_{st.session_state.current_player}",
                                    on_submit=increment_turn)
        logger.info(f"Waiting for input from player: {current_player_name}")

    # Check if all players have entered their input
    if st.session_state.current_player >= len(st.session_state.players):
        full_content = '\n'.join(f"{key}: {value}" for key, value in st.session_state.player_inputs.items())
        st.session_state.chat_history.append(HumanMessage(content=full_content))

        with st.chat_message("Human"):
            st.markdown(full_content)

        with st.chat_message("AI"):
            response = st.write_stream(get_response(full_content, st.session_state.chat_history, st.session_state.players))
        st.session_state.chat_history.append(AIMessage(content=response))

        # Reset for the next round of inputs
        st.session_state.current_player = 0
        st.session_state.player_inputs = {}
        logger.info("Player inputs reset for next round.")
        st.rerun()

# Display the list of players on the right (sidebar)
st.sidebar.title("Players")
for player in st.session_state.players:
    st.sidebar.write(player)
