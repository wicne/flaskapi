from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Modeli yükle
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return "Flask API Çalışıyor!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        features = pd.DataFrame([data['features']], columns=["team1", "team2", "weather", "time"])
        prediction = model.predict(features)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
