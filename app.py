import streamlit as st
import pickle
import re
import string

# Load trained model
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("Fake News Detection System")

news = st.text_area("Enter News Article")

if st.button("Predict"):
    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        st.success("This is REAL News")
    else:
        st.error("This is FAKE News")
