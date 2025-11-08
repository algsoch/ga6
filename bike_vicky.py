import streamlit as st
import pandas as pd
import numpy as np
from bike_vicky import solve_q1, solve_q2, solve_q3, solve_q4, solve_q5, solve_q6, solve_q7, solve_q8, solve_q9

st.title("Bike Sales Analysis - Q&A Viewer")

st.markdown("""
This app demonstrates how each question is calculated based on the bike sales dataset.
Select a question from the sidebar to view its output and calculation details.
""")

# Sidebar selection for question
question_options = {
    "Q1: Pearson correlations for TVS in Gujarat": solve_q1,
    "Q2: Pearson correlations for Yamaha in West Bengal": solve_q2,
    "Q3: Predicted resale price for Tier 2 motorbike": solve_q3,
    "Q4: Forecast resale price for Kawasaki - Versys 650 in Delhi": solve_q4,
    "Q5: Anomalous mileage records (Maharashtra)": solve_q5,
    "Q6: Anomalous mileage records (Uttar Pradesh)": solve_q6,
    "Q7: Distance from Central Command Post to Silver Haven Junction": solve_q7,
    "Q8: Closest community to the Central Command Post": solve_q8,
    "Q9: Evacuation route using nearest neighbor strategy": solve_q9
}

selection = st.sidebar.selectbox("Select Question", list(question_options.keys()))

st.header(selection)

# Run the selected question function and capture the result tuple (Question, Answer)
question_label, answer_text = question_options[selection]()

st.subheader("Calculation Details")
st.text(answer_text)

st.markdown("""
---
The above result is calculated using the functions defined in the code.  
For example, **Q7** calculates the haversine distance between the Central Command Post and Silver Haven Junction.
""")