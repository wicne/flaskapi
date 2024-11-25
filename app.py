from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Flask uygulamasını başlat
app = Flask(__name__)

# Modeli yükle
model = pickle.load(open('model.pkl', 'rb'))

# Ana sayfa
@app.route('/', methods=['GET'])
def home():
    return "Flask API Çalışıyor! Model tahmin için hazır."

# Tahmin yapmak için bir endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Kullanıcıdan gelen JSON verisini al
    data = request.json
    try:
        # Gelen veriyi işleme
        features = pd.DataFrame([data['features']], columns=["team1", "team2", "weather", "time"])
        
        # Tahmin yap
        prediction = model.predict(features)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)