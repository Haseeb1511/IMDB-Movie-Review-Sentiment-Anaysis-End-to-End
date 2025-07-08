import streamlit as st
import requests


API_URL = "http://fastapi:8000/predict"

st.title("IMDB Sentiment Classifier 🎬")

user_review = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        payload = {"review":user_review}
        response = requests.post(API_URL,json=payload)
             #positive":1,"negative":0
        if response.status_code == 200:
            result = response.json()
            prediction_list = result["prediction"]
            if isinstance(prediction_list,list) and len(prediction_list)==1:
                pred_value = prediction_list[0]
                sentiment = "Postive" if pred_value==1.0 else "Negative"
                st.success(f"Prediction: {sentiment}")
            else:
                st.warning(f"Unexpected prediction format: {prediction_list}")

        else:
            st.error(f"Error: {response.status_code} {response.text}")