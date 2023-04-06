import pickle

import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer


# Загрузка модели
with open('pickle_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Загрузка векторайзера
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def form():
    return '''<form action="/predict" method="POST">
                  <label for="text">Введите текст:</label><br>
                  <input type="text" id="text" name="text"><br><br>
                  <input type="submit" value="Отправить">
              </form>'''

@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из POST-запроса
    data = request.form
    text = data['text']

    # Векторизация текста
    text_vec = vectorizer.transform([text])

    # Классификация текста
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    # Формирование ответа в формате JSON
    probability = np.array(probability)
    response = {'prediction': int(prediction), 'probability': probability.tolist()}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
