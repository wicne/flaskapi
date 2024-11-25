from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Flask uygulamasını başlat
app = Flask(__name__)

# Modeli yükle
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Model yüklenemedi: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    return "Flask API Çalışıyor! Model tahmin için hazır."

# Tahmin yapmak için bir endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model yüklenemedi veya mevcut değil'}), 500

    # Kullanıcıdan gelen JSON verisini al
    data = request.get_json(force=True)
    try:
        # JSON verisinin 'features' anahtarını doğrula
        if 'features' not in data:
            return jsonify({'error': "Gönderilen JSON'da 'features' anahtarı eksik."}), 400

        # Gelen veriyi işleme
        features = pd.DataFrame([data['features']], columns=["team1", "team2", "weather", "time"])

        # Tahmin yap
        prediction = model.predict(features)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': f"Veri işleme veya tahmin hatası: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)
