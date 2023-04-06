import pickle
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка модели
with open('pickle_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Загрузка векторайзера
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из POST-запроса
    data = request.json
    text = data['text']

    # Векторизация текста
    text_vec = vectorizer.transform([text])

    # Классификация текста
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    # Формирование ответа в формате JSON
    response = {'prediction': prediction, 'probability': probability.tolist()}
    return jsonify(response)

@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
