import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import os
import smtplib
from email.mime.text import MIMEText

# Load the iris dataset
iris = load_iris()

# Standardize the features (fit scaler on iris data)
scaler = StandardScaler()
scaler.fit(iris.data)

# Streamlit App title
st.title('Iris Flower Prediction App')

# Sidebar for model selection
st.sidebar.header('Choose a Model')
model_choice = st.sidebar.selectbox("Select Model", ("KNN", "Decision Tree"))

# Load the selected model
if model_choice == "KNN":
    with open('knn_model.pkl', 'rb') as file:
        model = pickle.load(file)
else:
    with open('dt_model.pkl', 'rb') as file:
        model = pickle.load(file)

# Sidebar inputs for the Iris dataset features
st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('Sepal Length (cm)', float(iris.data[:, 0].min()), float(iris.data[:, 0].max()))
sepal_width = st.sidebar.slider('Sepal Width (cm)', float(iris.data[:, 1].min()), float(iris.data[:, 1].max()))
petal_length = st.sidebar.slider('Petal Length (cm)', float(iris.data[:, 2].min()), float(iris.data[:, 2].max()))
petal_width = st.sidebar.slider('Petal Width (cm)', float(iris.data[:, 3].min()), float(iris.data[:, 3].max()))

# Prepare the input data for prediction
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_data_scaled = scaler.transform(input_data)

# Prediction using the loaded model
prediction = model.predict(input_data_scaled)

# Display the prediction
st.subheader('Prediction')
predicted_species = iris.target_names[prediction][0]
st.write(f"The predicted Iris species using {model_choice} is: {predicted_species}")

# Function to save feedback to a text file with type
def save_feedback(feedback, feedback_type):
    with open('feedback.txt', 'a') as f:
        f.write(f"{feedback_type}: {feedback}\n")

# Function to read feedback from a text file
def load_feedback():
    if os.path.isfile('feedback.txt'):
        with open('feedback.txt', 'r') as f:
            return f.readlines()
    return []

# Function to send feedback via email
def send_email(feedback, feedback_type):
    sender_email = "gokulsgn7@gmail.com"  # Your email
    receiver_email = "gokulsgn7@gmail.com"  # Your email (can be different)
    password = "your_password"  # Change to your email password or app password
    
    subject = "New Feedback Received"
    body = f"Feedback Type: {feedback_type}\nFeedback: {feedback}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        st.sidebar.success("Feedback sent to your email!")
    except Exception as e:
        st.sidebar.error("Failed to send email. Please check your email configuration.")

# User Feedback Section
st.sidebar.header('📝 User Feedback')
feedback_type = st.sidebar.selectbox("Select Feedback Type", ["Positive", "Negative", "Suggestion"])
feedback = st.sidebar.text_area("Leave your feedback or suggestions here:")

if st.sidebar.button("Submit Feedback"):
    if feedback.strip():
        save_feedback(feedback, feedback_type)
        send_email(feedback, feedback_type)  # Send feedback via email
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.warning("Please enter some feedback before submitting.")

# Display submitted feedback
st.subheader('👥 Submitted Feedback')
previous_feedback = load_feedback()
if previous_feedback:
    for f in previous_feedback:
        st.write(f)
else:
    st.write("No feedback submitted yet.")
