import streamlit as st
import joblib

# --------------------------------------------------
# Load trained model and TF-IDF vectorizer
# --------------------------------------------------

model = joblib.load("text_classifier.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")


# --------------------------------------------------
# Streamlit App
# --------------------------------------------------

st.title("Text Classification System")

st.write("Enter a movie review below:")

text = st.text_area(
    "Enter your text",
    placeholder="Example: The movie was excellent and very enjoyable."
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Convert text into TF-IDF features
        text_vector = vectorizer.transform([text])

        # Make prediction
        prediction = model.predict(text_vector)[0]

        # Convert prediction to readable sentiment
        if isinstance(prediction, str):

            prediction_text = prediction.capitalize()

        elif prediction == 1:

            prediction_text = "Positive"

        else:

            prediction_text = "Negative"


        # Display results
        st.write("Input:", text)

        if prediction_text.lower() == "positive":
            st.success("Prediction: Positive")
        else:
            st.error("Prediction: Negative")
