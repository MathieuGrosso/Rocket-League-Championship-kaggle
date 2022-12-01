
from flask import Flask, jsonify, request
from src.app import return_predictions

import pandas as pd
import os


app = Flask(__name__)


model_FILEPATH_A = "/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/model/model_A_0.joblib"
model_FILEPATH_B = "/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/model/model_B_0.joblib"


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"


@app.route('/predict', methods=['POST'])
def get_scores():
    # handle data
    payload = request.json  # get the data
    print(payload)
    input_df = pd.DataFrame(payload)
    print(input_df)
    payload = request.json
    input_df = pd.DataFrame(payload)
    input_df.fillna(-1, inplace=True)

    # load all models

    predictions = return_predictions(
        input_df, model_FILEPATH_A, model_FILEPATH_B)
    scores = [prediction[1] for prediction in predictions]

    return jsonify({'scores': scores})


if __name__ == '__main__':
    app.run()
