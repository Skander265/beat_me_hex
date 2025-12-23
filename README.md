hex game

in a hex game your goal is to connect the sides assigned to you in a 11x11 board. connect them before your AI opponent does

this is a school project where i had to write an ai agent. wanted to put it on my github so i wrapped it in a streamlit app and added the board logic/state so you can actually play against it.

ai_agent.py: the ai written for the (minimax + alpha beta pruning). It uses iterative deepening, Zobrist hashing for caching, beam search for pruning, and Dijkstra's algorithm to calculate the shortest path to victory.

how to run:

pip install streamlit 
streamlit run app.py

files:

    ai_agent.py: the ai written for the assignment (minimax + alpha beta pruning)

    app.py: the ui

    hex_engine/: game logic needed to run the board
