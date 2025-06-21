import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")
st.write("Enter any news headline or article below to check if it's real or fake.")

# Input text
user_input = st.text_area("News Text", height=200)

# Predict button
if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0][prediction]

        if prediction == 1:
            st.success(f"🟢 Real News! ({probability*100:.2f}% confidence)")
        else:
            st.error(f"🔴 Fake News! ({probability*100:.2f}% confidence)")
