import streamlit as st
import pickle
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered"
)

# --- Custom CSS for "Betting App" Theme ---
st.markdown("""
    <style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Neon Accents for inputs */
    div[data-baseweb="select"] > div {
        background-color: #1f2937;
        color: white;
        border: 1px solid #4b5563;
    }
    
    /* Card Styling */
    .css-1r6slb0 {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #00d26a; /* Betting Green */
        color: black;
        font-weight: bold;
        border: none;
        padding: 10px;
        border-radius: 5px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00b058;
        color: white;
    }
    
    /* Probability Text */
    .prob-text {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }
    .win-color { color: #00d26a; }
    .lose-color { color: #ff4b4b; }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
# Ensure 'win_predictor.pkl' is in the same folder
try:
    pipe = pickle.load(open('win_predictor.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'win_predictor.pkl' not found. Please upload it.")
    st.stop()

# --- Constants ---
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# --- UI Layout ---
st.title("🏏 IPL Win Predictor")
st.markdown("### ⚡ Live Match Odds")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('🏏 Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('bowling Team', sorted(teams))

selected_city = st.selectbox('🏟️ Host City', sorted(cities))

target = st.number_input('🎯 Target Score', min_value=0, step=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=10, step=1)

if st.button('🎲 Predict Probability'):
    
    # 1. Calculation Logic
    runs_needed = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_needed * 6) / balls_left if balls_left > 0 else 0

    # 2. Input DataFrame (Must match the Training Columns exactly)
    input_df = pd.DataFrame({
        'batting_squad': [batting_team],
        'bowling_squad': [bowling_team],
        'venue_city': [selected_city],
        'runs_needed': [runs_needed],
        'balls_remaining': [balls_left],
        'wickets_remaining': [wickets_left],
        'target_score': [target],
        'cur_run_rate': [crr],
        'req_run_rate': [rrr]
    })

    # 3. Prediction
    try:
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # 4. Display Results
        st.markdown("---")
        st.subheader("📊 Match Probability")
        
        col_win, col_loss = st.columns(2)
        
        with col_win:
            st.markdown(f"<div class='prob-text win-color'>{batting_team}<br>{round(win*100)}%</div>", unsafe_allow_html=True)
            st.progress(win)
        
        with col_loss:
            st.markdown(f"<div class='prob-text lose-color'>{bowling_team}<br>{round(loss*100)}%</div>", unsafe_allow_html=True)
            st.progress(loss)

    except Exception as e:
        st.error(f"Error in prediction: {e}")