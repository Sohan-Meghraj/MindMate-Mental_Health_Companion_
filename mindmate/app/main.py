import streamlit as st
from chatbot import predict_emotion  # Make sure chatbot.py is correctly placed

# Streamlit App Title
st.set_page_config(page_title="MindMate - Mental Health Companion", page_icon="🧠")
st.title("💬 MindMate - Mental Health Companion")
st.subheader("Understand your emotions through conversation")

# Input field
user_input = st.text_input("🧠 How are you feeling today?")

# Predict button
if st.button("Analyze"):
    if user_input:
        predicted_emotion = predict_emotion(user_input)
        st.success(f"Predicted Emotion: **{predicted_emotion.upper()}**")
    else:
        st.warning("Please enter how you're feeling to analyze.")
