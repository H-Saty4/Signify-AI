from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER_PATH = os.path.join(BASE_DIR, "model_pkl")

try:
    model_file = os.path.join(MODEL_FOLDER_PATH, 'signify_model_rf.pkl')
    encoder_file = os.path.join(MODEL_FOLDER_PATH, 'label_encoder_rf.pkl')
 
    with open(model_file, 'rb') as f: model = pickle.load(f)
    with open(encoder_file, 'rb') as f: encoder = pickle.load(f)
    print(">>> [SUCCESS] Model .pkl dan Encoder berhasil dimuat.")
except Exception as e:
    print(f"\n[FATAL ERROR] Gagal memuat model dari folder '{MODEL_FOLDER_PATH}'.")
    print("  Pastikan file .pkl Anda ada di sana setelah menjalankan train_model.py.")
    print(f"  Detail Error: {e}\n")
    exit()

@app.route('/')
def home():
    """Menyajikan Halaman Beranda."""
    return render_template('home.html')

@app.route('/translator')
def translator():
    """Menyajikan Halaman Penerjemah."""
    return render_template('translator.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data.get('features')
        if not features or len(features) != 63: return jsonify({})

        data_array = np.array([features])
        prediction_idx = model.predict(data_array)
        prediction_proba = model.predict_proba(data_array)
        confidence = float(np.max(prediction_proba))
        
        prediction_result = {'prediction': 'NETRAL', 'confidence': confidence}
        
        if confidence > 0.75:
            label = encoder.inverse_transform(prediction_idx)[0]
            prediction_result['prediction'] = label
            
        return jsonify(prediction_result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(">>> Signify AI Translator (Classic Model) berjalan di http://127.0.0.1:5002 <<<")
    app.run(host='0.0.0.0', port=5002)