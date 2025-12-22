import streamlit as st

def show_welcome_screen():

    if 'game_started' not in st.session_state:
        st.session_state.game_started = False

    if st.session_state.game_started:
        return

    st.markdown("""
    <style>
        .menu-box {
            background-color: #1f2024;
            padding: 30px;
            border-radius: 15px;
            border: 1px solid #444;
            text-align: center;
            max-width: 700px;
            margin: auto;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .rules-text {
            text-align: left; 
            margin-top: 20px; 
            font-size: 18px;
            color: #ddd;
        }
        h1, h3 { color: white; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="menu-box">
            <h1>⬡ Welcome to Hex</h1>
            <h3>I challenge you to beat me!</h3>
            <hr style="border-color: #444;">
            <div class="rules-text">
                <b>The Rules (11x11 Board):</b>
                <ul>
                    <li>Players take turns placing pieces on empty spots.</li>
                    <li><span style="color: #ff4b4b;"><b>RED</b></span> connects Top ↕ Bottom.</li>
                    <li><span style="color: #4b88ff;"><b>BLUE</b></span> connects Left ↔ Right.</li>
                    <li>The four corner hexagons belong to both adjacent sides.</li>
                    <li><b>There are no draws in Hex!</b></li>
                </ul>
            </div>
            <br>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") 
        
        if st.button("⚔️ ACCEPT CHALLENGE ⚔️", use_container_width=True, type="primary"):
            st.session_state.game_started = True
            st.rerun()

    st.stop()