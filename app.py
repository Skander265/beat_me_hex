import streamlit as st
from game_menu import show_welcome_screen

st.set_page_config(page_title="Hex Agent", layout="wide")

show_welcome_screen()

import random
import time
import concurrent.futures
from hex_engine.state import GameState
from hex_engine.board import Board
from hex_engine.action import Action
from ai_agent import MyPlayer


st.markdown("""
<style>
    /* Global Cleanup */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    ::-webkit-scrollbar {display: none;}
    .stApp { overflow: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 0rem; }

    /* HEX BOARD BUTTON STYLING */
    /* We only want this 36px width for the board, not the menu! */
    .stButton button {
        width: 36px; height: 36px;
        border-radius: 6px; 
        font-size: 20px; padding: 0;
        line-height: 36px; min-height: 36px;
        background-color: #1f2024; 
        color: #eee;
        border: 1px solid #333;
    }
    
    /* Remove bottom margin for grid alignment */
    div.row-widget.stButton { 
        text-align: center;
        margin-bottom: 0px !important;
        padding-bottom: 2px !important;
    }

    /* LOADING OVERLAY */
    .loading-overlay {
        position: fixed; top: 40%; left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background-color: rgba(30, 30, 30, 0.95);
        border: 1px solid #444; border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        width: 320px; padding: 30px;
        text-align: center; font-size: 20px; color: #ddd;
    }
</style>
""", unsafe_allow_html=True)

class LoadingOverlay:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        
    def info(self, text):
        self.placeholder.markdown("""
            <div class="loading-overlay">
                &nbsp; I am still thinking...
            </div>
        """, unsafe_allow_html=True)

def run_ai_with_loading_screen(agent, state, ui_container):
    def _compute():
        return agent.compute_action(state)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_compute)
        ui_container.info("") 
        while not future.done():
            time.sleep(0.1)
        return future.result()

def log_move(player, pos):
    st.session_state.move_history.append(f"{player} placed at {pos}")

def handle_click(r, c):
    if st.session_state.game_over: return
    state = st.session_state.game_state
    
    if state.get_next_player().get_piece_type() != st.session_state.human_color:
        st.toast("Not your turn!", icon="🚫")
        return

    if (r, c) in state.board.get_env():
        st.toast("Spot taken!", icon="🚫")
        return

    action = Action((r, c), st.session_state.human_color)
    log_move(f"Player ({st.session_state.human_color})", (r, c))
    st.session_state.game_state = state.apply_action(action)

    if st.session_state.game_state.is_done():
        st.session_state.winner = f"Player ({st.session_state.human_color})"
        st.session_state.game_over = True
    else:
        st.session_state.trigger_ai = True

BOARD_SIZE = 11

if 'game_state' not in st.session_state:
    st.session_state.human_color = random.choice(["R", "B"])
    st.session_state.ai_color = "B" if st.session_state.human_color == "R" else "R"
    
    initial_board = Board(BOARD_SIZE)
    st.session_state.game_state = GameState(initial_board, next_player_type="R")
    
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.move_history = []  
    
    agent = MyPlayer(piece_type=st.session_state.ai_color, name="AI_Agent")
    agent.board_size = BOARD_SIZE 
    st.session_state.agent = agent

    st.session_state.trigger_ai = (st.session_state.ai_color == "R")

col1, col2 = st.columns([2, 1])

with col1:
    human = st.session_state.human_color
    st.markdown(f"### {'🟥 You are RED (Top ↕ Bottom)' if human == 'R' else '🟦 You are BLUE (Left ↔ Right)'}")
    
    env = st.session_state.game_state.board.get_env()
    curr = st.session_state.game_state.get_next_player().get_piece_type()
    
    for r in range(BOARD_SIZE):
        left = (0.2 * r) + 0.2
        right = max(0.2, 3 - (0.2 * r))
        cols = st.columns([left] + [1] * BOARD_SIZE + [right])
        
        for c in range(BOARD_SIZE):
            pos = (r, c)
            piece = env.get(pos)
            
            label = "·"
            if piece: label = "🟦" if piece.get_type() == "B" else "🟥"
            
            disabled = st.session_state.game_over or (piece is not None) or (curr != human)
            cols[c+1].button(label, key=f"{r}-{c}", on_click=handle_click, args=(r,c), disabled=disabled)

with col2:
    st.write("## Status")
    curr = st.session_state.game_state.get_next_player().get_piece_type()
    
    if st.session_state.game_over:
        st.success(f"{st.session_state.winner} Wins!")
        if st.button("New Game", type="primary"): 
            st.session_state.clear()
            st.rerun()

    elif st.session_state.trigger_ai:
        st.info("I am still thinking...") 
    elif curr == st.session_state.human_color:
        st.success("Your Turn")
    else:
        st.warning("Waiting...")

    with st.expander("Move History", expanded=True):
        for move in reversed(st.session_state.move_history[-15:]):
            st.text(move)

overlay_placeholder = st.empty()

if st.session_state.get("trigger_ai", False) and not st.session_state.game_over:
    if st.session_state.game_state.get_next_player().get_piece_type() == st.session_state.ai_color:
        
        loader = LoadingOverlay(overlay_placeholder)
        
        ai_action = run_ai_with_loading_screen(
            st.session_state.agent, 
            st.session_state.game_state, 
            loader 
        )
        
        overlay_placeholder.empty()
        
        log_move(f"AI ({st.session_state.ai_color})", ai_action.data["position"])
        st.session_state.game_state = st.session_state.game_state.apply_action(ai_action)
        st.session_state.trigger_ai = False
        
        if st.session_state.game_state.is_done():
            st.session_state.winner = f"AI ({st.session_state.ai_color})"
            st.session_state.game_over = True
        
        st.rerun()