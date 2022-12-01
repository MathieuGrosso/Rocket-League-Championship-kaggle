from pyexpat import model
from joblib import dump, load
import pandas as pd
import os
import pprint
import matplotlib.pyplot as plt
import numpy as np


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
MODEL_FILEPATH_A = os.path.join(
    "/Users/mathieugrosso/Desktop/Workspace_Git/AI_Advanced/my_model_api/model", "model_A_2.joblib")
N_ESTIMATORS = 210
MAX_DEPTH = 8
LEARNING_RATE = 0.1
N_SPLITS = 5
scoring = 'neg_log_loss'
FOLDS = 5

pprint(MODEL_FILEPATH_A)

scores = {'A': [], 'B': []}
test_predictions = {'A': [], 'B': []}
my_seed = 42
param = {}
param['booster'] = 'dart'
param['max_depth'] = MAX_DEPTH
param["learning_rate"] = LEARNING_RATE
param["n_estimators"] = N_ESTIMATORS
param['objective'] = "binary"
param["tree_method"] = 'gpu_hist'
param["subsample"] = 0.1
param["verbosity"] = 0

scores = {'A': [], 'B': []}
test_predictions = {'A': [], 'B': []}

DATA_FILEPATH = os.path.join('data', '')
# def train_and_save_model(model_filepath):
#         train = pd.read_csv(DATA_FILEPATH, index_col=0)
#         train.fillna(-1, inplace=True)
#         X = train.drop('SeriousDlqin2yrs', axis=1)
#         y = train['SeriousDlqin2yrs']
#         gbm = GradientBoostingClassifier()
#         gbm.fit(X, y)
#         dump(gbm, model_filepath)


def train_model():
    for key in test_predictions:
        print(f"Team: {key} ")
        # X, y, test, key
        scores = []
        # test_predictions = []
        cv = KFold(n_splits=N_SPLITS, random_state=my_seed, shuffle=True)
        print(N_SPLITS)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            train_X, val_X = X.iloc[train_idx], X.iloc[test_idx]
            train_y, val_y = y.iloc[train_idx], y.iloc[test_idx]
            path = "./model/saved"
            model_filepath = os.path.join(
                path, "model_"+key+"_"+str(fold)+".joblib")
            # print(model_filepath)
            model = LGBMClassifier()
            model.set_params(**param)

            model.fit(train_X, train_y)

            predictions = model.predict_proba(val_X)[:, 1]
            print(f"predictions:{predictions}")
            score = roc_auc_score(val_y, predictions)
            scores.append(score)
            print(f"Fold {fold + 1} \t\t AUC: {score}")

            fpr, tpr, thresholds = metrics.roc_curve(val_y, predictions)
            roc_auc = metrics.auc(fpr, tpr)
            display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                              estimator_name='example estimator')
            display.plot()
            plt.show()

            dump(model, model_filepath)

            return np.mean(scores)


def predict(model_path, test_data):
    test_predictions = []
    model = load(model_path)
    # test predictions with this model
    test_predictions.append(model.predict_proba(test_data)[:, 1])

    return test_predictions


def run_predictions(model_path, test_data):
    for key in test_predictions:
        print(f"Team: {key} ")
        prediction, score = predict(model_path=model_path, test_data=test_data)
        test_predictions[key].append(prediction)
        scores[key].append(score)

    return (np.mean(scores['A']), np.mean(scores['B']))


def return_predictions():
    # really dirty way to return the predictions
    test_predictions = run_predictions(model_path, test_data)

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
