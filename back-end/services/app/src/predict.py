# from pyexpat import model
# from joblib import dump, load
# import pandas as pd
# import os
# from icecream import ic
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import glob


# # import xgboost as xgb
# # from lightgbm import LGBMClassifier

# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import cross_validate, KFold  # k-fold Cross Validation
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import roc_auc_score
# from sklearn import metrics


# MODEL_PATH = "/app/models"
# DATA_FILEPATH = os.path.join('data', '')
# # model_A_path="/home/ubuntu/Kaggle_Competitions/tabular_playground_series/model_A_0.joblib"
# # model_B_path="/home/ubuntu/Kaggle_Competitions/tabular_playground_series/model_B_0.joblib"
# # liste_model = [model_A_path,model_B_path]

# scores = {'A': [], 'B': []}
# test_predictions = {'A': [], 'B': []}


# def run_model(test, key):
#     test_predictions = []
#     # print(glob.glob(glob.glob('/home/ubuntu/Kaggle_Competitions/tabular_playground_series/')):
#     # import os
#     # ic(os.getcwd())
#     # ic(glob.glob('/app/models/*.joblib'))
#     for filenames in os.listdir('/app/models/'):
#         for i in filenames:
#             path = os.path.basename(filenames)
#             if key in path:
#                 model_path = os.path.join(MODEL_PATH,path)
#                 model = load(model_path)
#                 test_predictions.append(model.predict_proba(test)[:, 1])

#     return np.mean(test_predictions, axis=0)


# def run_prediction(input_df, model_A_path, model_B_path):
#     test_predictions = {'A': [], 'B': []}

#     for key in test_predictions:
#         print(f"Team: {key} ")

#         prediction = run_model(input_df, key)


#         test_predictions[key].append(prediction)
#     return test_predictions


# def return_predictions(input_df, model_A_path, model_B_path):

#     test_predictions = run_prediction(input_df, model_A_path, model_B_path)

#     test_predictions['B'] = test_predictions['B'][0]
#     test_predictions['A'] = test_predictions['A'][0]
#     test_predictions['team_A_scoring_within_10sec'] = test_predictions['A']
#     test_predictions['team_B_scoring_within_10sec'] = test_predictions['B']

#     test_predictions['id'] = [i for i in range(
#         len(test_predictions['team_A_scoring_within_10sec']))]

#     del test_predictions['A']
#     del test_predictions['B']
#     submission = pd.DataFrame.from_dict(test_predictions)
#     submission = submission.to_dict(orient='records')

#     return submission
