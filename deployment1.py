# -*- coding: utf-8 -*-
"""Deployment1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r-t3SvE1VV2PyLso21NUNNejz3lRqqCO
"""

!pip install streamlit

# Importing necessary libraries
import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model("/content/classification.h5")

# Load the tokenizer
t = Tokenizer(num_words = 500)
with open('/content/token.pkl', 'rb') as handle:
    t = pickle.load(handle)

# Define the function to make prediction
def predict_class_label(text):
    # Convert the input text to sequence of tokens
    text_sequence = t.texts_to_sequences([text])
    # Padding the sequence to a fixed length
    text_sequence = pad_sequences(text_sequence, padding='post', maxlen=250)
    # Making prediction using the loaded model
    prediction = model.predict(text_sequence)
    # Getting the predicted class
    predicted_class = np.argmax(prediction)
    # Returning the predicted class as output
    return predicted_class

# Define the Streamlit app
def app():
    # Set the app title
    st.title("Consumer complaint classification")
    # Set the app description
    st.write("This application redirects the complaint to the appropriate team")
    # Get user input
    user_input = st.text_input("Enter your complaint:")
    # When user clicks the predict button
    if st.button("Predict"):
        # Call the predict_next_word function to make prediction
        prediction = predict_class_label(user_input)
        # Get the predicted class label
        predicted_class_label = ''
        if prediction == 0:
            predicted_class_label = 'Credit card'
        elif prediction == 1:
            predicted_class_label = 'Credit reporting'
        elif prediction == 2:
            predicted_class_label = 'Debt Collection'
        elif prediction == 3:
            predicted_class_label = 'Mortgages and loans'
        else:
            predicted_class_label = 'Retail banking'
        # Show the predicted class label as output
        st.success("The predicted class is {}".format(predicted_class_label))

# Run the app
if __name__ == '__main__':
    app()

