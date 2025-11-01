Crypto Liquidity Prediction - simple project
Files:
- train.py : trains a RandomForest model on the provided CSVs and saves model.pkl and scaler.pkl
- app.py : Flask app serving a frontend and /api/predict endpoint
- templates/index.html : simple frontend to input features and call the API
- processed_sample.csv : processed sample used for demo after training

How to run:
1) Move coin_gecko_2022-03-16.csv and coin_gecko_2022-03-17.csv to the parent folder of this project directory.
2) Create a virtualenv and install requirements: pip install -r requirements.txt
3) Run training: python train.py  (this creates model.pkl and scaler.pkl in the project folder)
4) Run API: python app.py
5) Open http://127.0.0.1:5000/ in your browser and try predictions.
