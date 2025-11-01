from flask import Flask, render_template, request, jsonify
import pickle, numpy as np, pandas as pd
from pathlib import Path

app = Flask(__name__)

MODEL = pickle.load(open(Path(__file__).parent / 'model.pkl','rb'))
SCALER = pickle.load(open(Path(__file__).parent / 'scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        arr = np.array([
            float(data['price']),
            float(data['1h']),
            float(data['24h']),
            float(data['7d']),
            float(data['24h_volume']),
            float(data['mkt_cap']),
        ]).reshape(1,-1)
    except Exception as e:
        return jsonify({'error': 'invalid input: '+str(e)}), 400
    arr_scaled = SCALER.transform(arr)
    pred = MODEL.predict(arr_scaled)[0]
    return jsonify({'predicted_liquidity_ratio': float(pred)})

if __name__ == '__main__':
    app.run(debug=True)