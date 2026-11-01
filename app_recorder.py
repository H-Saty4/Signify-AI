
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import csv

app = Flask(__name__)
DATA_PATH = "data_sequences_final"
os.makedirs(DATA_PATH, exist_ok=True)
 
@app.route('/')
def index():
    return render_template('data_recorder.html')

@app.route('/save_sequence', methods=['POST'])
def save_sequence():
    try:
        data = request.json
        label = str(data.get('label', '')).strip().upper()
        features = data.get('features') 
        
        if not label or not features or len(features) != 63:
            return jsonify({'status': 'error', 'message': 'Data tidak valid.'}), 400
        
        csv_path = os.path.join(DATA_PATH, f"{label}.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([f'feature_{i}' for i in range(63)])
            writer.writerow(features)
        
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)