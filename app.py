import streamlit as st
import pickle
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Betting App Predictor - For cricket",
    page_icon="🏏",
    layout="centered"
)

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

# 1. Team Selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('⚾ Bowling Team', sorted(teams))

# 2. Host City
selected_city = st.selectbox('🏟️ Host City', sorted(cities))

# 3. Target Score
target = st.number_input('🎯 Target Score', min_value=0, step=1)

# 4. Match Situation
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score', min_value=0, step=1)

with col4:
    # Separate Overs and Balls to ensure correct cricket logic
    overs = st.number_input('Overs Done', min_value=0, max_value=19, step=1)
    balls = st.number_input('Balls Done (This Over)', min_value=0, max_value=5, step=1)

with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=9, step=1)

# --- Prediction Logic ---
if st.button('🎲 Predict Probability'):
    
    # Validation 1: Same Teams
    if batting_team == bowling_team:
        st.error("❌ Batting and Bowling teams cannot be the same!")
        st.stop()
    
    # Validation 2: Score > Target
    if score >= target:
        st.error(f"❌ Match Over! {batting_team} has already won.")
        st.stop()
        
    # Validation 3: All Out
    if wickets >= 10:
        st.error(f"❌ Match Over! All wickets fell.")
        st.stop()

    # Calculation Logic
    runs_needed = target - score
    
    # Cricket Logic: Total Balls = (Overs * 6) + Balls in current over
    total_balls_bowled = (overs * 6) + balls
    balls_left = 120 - total_balls_bowled
    
    wickets_left = 10 - wickets
    
    # Avoid division by zero
    crr = (score * 6) / total_balls_bowled if total_balls_bowled > 0 else 0
    rrr = (runs_needed * 6) / balls_left if balls_left > 0 else 0

    # Input DataFrame
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

    # Prediction
    try:
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display Results
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