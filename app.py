# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import google.generativeai as genai

# --- Gemini API Configuration ---
# IMPORTANT: Set this in your Render environment variables
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Load Your Machine Learning Model ---
model = joblib.load('stacking_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
model_columns = joblib.load('model_columns.pkl')

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    # This endpoint remains the same as before
    data = request.get_json()
    df_new = pd.DataFrame([data], columns=model_columns)
    df_new_imputed = imputer.transform(df_new)
    df_new_scaled = scaler.transform(df_new_imputed)
    prediction_encoded = model.predict(df_new_scaled)
    prediction_proba = model.predict_proba(df_new_scaled)
    mapping = {0: "FALSE POSITIVE", 1: "CONFIRMED", 2: "CANDIDATE"}
    prediction_label = mapping[prediction_encoded[0]]
    confidence = np.max(prediction_proba) * 100
    return jsonify({
        'prediction': prediction_label,
        'confidence': f"{confidence:.2f}%"
    })

@app.route('/chat', methods=['POST'])
def chat():
    # This is the new endpoint for the chatbot
    data = request.get_json()
    user_message = data.get('message')
    context = data.get('context')

    if not user_message or not context:
        return jsonify({'error': 'Message and context are required.'}), 400

    # --- Prompt Engineering: The Secret to a Great Chatbot ---
    prompt = f"""
    You are Exo-Chat, a friendly and brilliant NASA astronomer and science communicator. 
    Your goal is to explain complex exoplanet data in an exciting and easy-to-understand way.
    A user has just received a classification for a Kepler Object of Interest and has questions.

    HERE IS THE DATA CONTEXT:
    - Input Data: {context['input_data']}
    - Model Prediction: The object is classified as a "{context['prediction']}" with {context['confidence']} confidence.

    Based on this context, answer the user's question clearly and enthusiastically. 
    If they ask about a specific value, explain what it means (e.g., 'koi_period' is how long it takes to orbit its star).
    Never break character. Be encouraging and spark curiosity about space exploration.

    USER'S QUESTION: "{user_message}"
    """

    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({'reply': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500