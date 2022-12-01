import gc
from pyexpat import model
from joblib import dump, load
import pandas as pd
import os
import pprint
import matplotlib.pyplot as plt
import numpy as np

from src.utils import reduce_mem_usage

# import xgboost as xgb
from lightgbm import LGBMClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate, KFold  # k-fold Cross Validation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# DATA_FILEPATH = os.path.join('app', 'data', 'cs-training.csv')
# MODEL_FILEPATH = os.path.join('app', 'models', 'model.joblib')
# path_1 = "/Users/mathieugrosso/Desktop/Workspace_Git/AI_Advanced/my_model_api/model"


MODEL_PATH = "/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/model/"
DATA_FILEPATH = os.path.join('data', '')


scores = {'A': [], 'B': []}
test_predictions = {'A': [], 'B': []}


def run_model(test, key):
    test_predictions = []
    for dirname, _, filenames in os.walk('../model'):
        for i in filenames:
            if key in i:
                model_path = os.path.join(MODEL_PATH, i)
                model = load(model_path)
                test_predictions.append(model.predict_proba(test)[:, 1])
                print(test_predictions)
    return np.mean(test_predictions)


def run_prediction(input_df, model_A_path, model_B_path):
    for key in test_predictions:
        print(f"Team: {key} ")
        prediction = run_model(input_df, key)
        test_predictions[key].append(prediction)
    return test_predictions


def return_predictions(input_df, model_A_path, model_B_path):
    test_predictions = run_prediction(input_df, model_A_path, model_B_path)

    test_predictions['B'] = test_predictions['B'][0]
    test_predictions['A'] = test_predictions['A'][0]

    test_predictions['team_A_scoring_within_10sec'] = test_predictions['A']
    test_predictions['team_B_scoring_within_10sec'] = test_predictions['B']
    test_predictions['id'] = [i for i in range(
        len(test_predictions['team_A_scoring_within_10sec']))]

    del test_predictions['A']
    del test_predictions['B']
    submission = pd.DataFrame.from_dict(test_predictions)
    submission = submission.to_dict(orient='records')

    return submission


if __name__ == '__main__':
    test_data = [0, 1]
    run_model(test_data, 'A')
