
# import pandas as pd
# import numpy as np
# import os
# import gc
# import pprint
# from icecream import ic
# from joblib import dump, load
# from sklearn.model_selection import KFold
# from sklearn.metrics import roc_auc_score
# from sklearn import metrics
# from sklearn.model_selection import cross_validate  # k-fold Cross Validation
# from sklearn.preprocessing import LabelEncoder
# from lightgbm import LGBMClassifier
# import matplotlib.pyplot as plt
# import pprint
# from joblib import dump, load
# from sklearn.model_selection import KFold
# from sklearn.metrics import roc_auc_score
# from sklearn import metrics
# from lightgbm import LGBMClassifier
# import matplotlib.pyplot as plt

# scores = {'A': [], 'B': []}
# test_predictions = {'A': [], 'B': []}

# input_cols = [
#     'ball_pos_x', 'ball_pos_y', 'ball_pos_z', 'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
#     'p0_pos_x', 'p0_pos_y', 'p0_pos_z', 'p0_vel_x', 'p0_vel_y', 'p0_vel_z',
#     'p1_pos_x', 'p1_pos_y', 'p1_pos_z', 'p1_vel_x', 'p1_vel_y', 'p1_vel_z',
#     'p2_pos_x', 'p2_pos_y', 'p2_pos_z', 'p2_vel_x', 'p2_vel_y', 'p2_vel_z',
#     'p3_pos_x', 'p3_pos_y', 'p3_pos_z', 'p3_vel_x', 'p3_vel_y', 'p3_vel_z',
#     'p4_pos_x', 'p4_pos_y', 'p4_pos_z', 'p4_vel_x', 'p4_vel_y', 'p4_vel_z',
#     'p5_pos_x', 'p5_pos_y', 'p5_pos_z', 'p5_vel_x', 'p5_vel_y', 'p5_vel_z',
#     'p0_boost', 'p1_boost',  'p2_boost', 'p3_boost', 'p4_boost', 'p5_boost',
#     'boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer', 'boost4_timer', 'boost5_timer']


# output_cols = ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']

# vel_groups = {
#     f"{el}_vel": [f'{el}_vel_x', f'{el}_vel_y', f'{el}_vel_z']
#     for el in ['ball'] + [f'p{i}' for i in range(6)]
# }
# pos_groups = {
#     f"{el}_pos": [f'{el}_pos_x', f'{el}_pos_y', f'{el}_pos_z']
#     for el in ['ball'] + [f'p{i}' for i in range(6)]
# }
# pos_groups


# def reduce_mem_usage(df, verbose=False):
#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     start_mem = df.memory_usage().sum() / 1024**2
#     for col in df.columns:
#         col_type = df[col].dtypes
#         if col_type in numerics:
#             c_min = df[col].min()
#             c_max = df[col].max()
#             if str(col_type)[:3] == 'int':
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)
#         else:
#             df[col] = df[col].astype('category')
#     end_mem = df.memory_usage().sum() / 1024**2
#     if verbose:
#         print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
#             end_mem, 100 * (start_mem - end_mem) / start_mem))
#     return df


# def read_train_data():
#     files_name = []
#     for dirname, _, filenames in os.walk('../data'):
#         for filename in filenames:
#             print(os.path.join(dirname, filename))
#             files_name.append(os.path.join(dirname, filename))
#     dtypes_df = pd.read_csv(
#         '../data/train_dtypes.csv')

#     dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}

#     cols = list(dtypes.keys())
#     train_df = pd.DataFrame({}, columns=cols)
#     for i in range(10):
#         print(i)
#         df_tmp = pd.read_csv(f'{path_to_data}/train_{i}.csv', dtype=dtypes)
#         if SAMPLE < 1:
#             df_tmp = df_tmp.sample(frac=SAMPLE, random_state=42)

#         train_df = pd.concat([train_df, df_tmp])

#         del df_tmp

#     train_df = reduce_mem_usage(train_df)
#     return train_df


# def read_test_data():
#     dtypes_df = pd.read_csv(
#         '../data/test_dtypes.csv/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/data/test_dtypes.csv')
#     dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}
#     test = pd.read_csv(
#         '../data/test.csv', dtype=dtypes)


# # preprocess data
# def preprocess_data():
#     train0_df = train_df.dropna()
#     test = test.fillna(value=test.mean())
#     return train0_df, test


# def euclidian_dist(x):
#     return np.linalg.norm(x, axis=1)


# def features_engineering(train, test):
#     # absolute speed of the ball :

#     array1 = train['ball_vel_x'].values*train['ball_vel_x'].values + train['ball_vel_y'].values * \
#         train['ball_vel_y'].values + \
#         train['ball_vel_z'].values*train['ball_vel_z'].values
#     array2 = test['ball_vel_x'].values*test['ball_vel_x'].values + test['ball_vel_y'].values * \
#         test['ball_vel_y'].values + \
#         test['ball_vel_z'].values*test['ball_vel_z'].values

#     train["abs_ball_speed"] = [np.sqrt(i) for i in array1]
#     test["abs_ball_speed"] = [np.sqrt(i) for i in array2]
#     train.keys()

#     # grouping by player and ball cat :

#     for col, vec in pos_groups.items():
#         print(vec)
#         train[col + '_dist_ball'] = euclidian_dist(
#             train[vec].values - train[pos_groups["ball_pos"]].values)
#         test[col + '_dist_ball'] = euclidian_dist(
#             test[vec].values - test[pos_groups["ball_pos"]].values)
#     for col, vec in vel_groups.items():
#         train[col] = euclidian_dist(train[vec])
#         test[col] = euclidian_dist(test[vec])

#     dist_cols = ['p0_pos_dist_ball',
#                  'p1_pos_dist_ball', 'p2_pos_dist_ball', 'p3_pos_dist_ball',
#                  'p4_pos_dist_ball', 'p5_pos_dist_ball']

#     train['closest_p_to_ball'] = train[dist_cols].idxmin(axis=1)
#     test['closest_p_to_ball'] = test[dist_cols].idxmin(axis=1)

#     train.replace(['p0_pos_dist_ball',
#                    'p1_pos_dist_ball', 'p2_pos_dist_ball', 'p3_pos_dist_ball',
#                    'p4_pos_dist_ball', 'p5_pos_dist_ball'], [0, 1, 2, 3, 4, 5], inplace=True)

#     test.replace(['p0_pos_dist_ball',
#                   'p1_pos_dist_ball', 'p2_pos_dist_ball', 'p3_pos_dist_ball',
#                   'p4_pos_dist_ball', 'p5_pos_dist_ball'], [0, 1, 2, 3, 4, 5], inplace=True)

#     print(train['closest_p_to_ball'])
#     return train, test


# def preprocess_data(df):
#     df = df.dropna().copy()

#     return ({'A': df['team_A_scoring_within_10sec'], 'B': df['team_B_scoring_within_10sec']},
#             df.drop(['game_num', 'event_id', 'event_time',
#                      'player_scoring_next', 'team_scoring_next',
#                      'team_A_scoring_within_10sec',
#                      'team_B_scoring_within_10sec', "abs_ball_speed", 'ball_pos_dist_ball', ], axis=1))


# scores = {'A': [], 'B': []}
# test_predictions = {'A': [], 'B': []}


# # run model
# # training and cross validation

# N_ESTIMATORS = 210
# MAX_DEPTH = 8
# LEARNING_RATE = 0.1
# N_SPLITS = 5
# my_seed = 42

# param = {}
# param['booster'] = 'dart'
# param['max_depth'] = MAX_DEPTH
# param["learning_rate"] = LEARNING_RATE
# param["n_estimators"] = N_ESTIMATORS
# param['objective'] = "binary"
# param["tree_method"] = 'gpu_hist'
# param["subsample"] = 0.1
# param["verbosity"] = 0

# scoring = 'neg_log_loss'
# FOLDS = 5


# def run_model(X, y, test, key):
#     scores = []
#     test_predictions = []
#     cv = KFold(n_splits=N_SPLITS, random_state=my_seed, shuffle=True)
#     print(N_SPLITS)
#     for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
#         train_X, val_X = X.iloc[train_idx], X.iloc[test_idx]
#         train_y, val_y = y.iloc[train_idx], y.iloc[test_idx]
#         path = "/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/models/"
#         model_filepath = os.path.join(
#             path, "model_" + str(key) + "_" + str(fold)+".joblib")
#         print(model_filepath)
#         # model = xgb.XGBClassifier()
#         model = LGBMClassifier()

#         model.set_params(**param)

#         model.fit(train_X, train_y)

#         dump(model, model_filepath)

#         predictions = model.predict_proba(val_X)[:, 1]
#         # predictions = model.predict_proba(val_X)
#         print(f"predictions:{predictions}")

#         # compute roc auc score
#         ic(val_y[:2])
#         ic(predictions[:2])
#         score = roc_auc_score(val_y, predictions)
#         scores.append(score)
#         print(f"Fold {fold + 1} \t\t AUC: {score}")

#         # display roc auc curve
#         fpr, tpr, thresholds = metrics.roc_curve(val_y, predictions)
#         roc_auc = metrics.auc(fpr, tpr)
#         display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
#                                           estimator_name='example estimator')
#         display.plot()
#         plt.show()

#         # test predictions with this model
#         test_predictions.append(model.predict_proba(test)[:, 1])

#         del model
#         gc.collect()

#     print(f"Overall AUC :", np.mean(scores))
#     return (np.mean(test_predictions, axis=0), np.mean(scores))

# # use a sample of data again :


# if __name__ == '__main__':

#     # load files
#     files_name = []
#     for dirname, _, filenames in os.walk('//Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/data/'):
#         for filename in filenames:
#             print(os.path.join(dirname, filename))
#             files_name.append(os.path.join(dirname, filename))

#     # load train data
#     SAMPLE = 0.2
#     # import tqdm
#     path_to_data = os.path.dirname(files_name[0])
#     print(path_to_data)

#     dtypes_df = pd.read_csv(
#         '//Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/data/train_dtypes.csv')
#     dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}

#     cols = list(dtypes.keys())
#     train_df = pd.DataFrame({}, columns=cols)
#     for i in range(10):
#         print(i)
#         df_tmp = pd.read_csv(f'{path_to_data}/train_{i}.csv', dtype=dtypes)
#         if SAMPLE < 1:
#             df_tmp = df_tmp.sample(frac=SAMPLE, random_state=42)

#         train_df = pd.concat([train_df, df_tmp])

#     del df_tmp
#     gc.collect()

#     # load test data:
#     dtypes_df = pd.read_csv(
#         '/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/data/test_dtypes.csv')
#     dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}
#     test = pd.read_csv(
#         '//Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/data/test.csv', dtype=dtypes)

#     train0_df = train_df.dropna()
#     test = test.fillna(value=test.mean())

#     # features engineering
#     train, test = features_engineering(train0_df, test)

#     train_small = train.sample(frac=0.5)  # for test purposes
#     y_train, X_train = preprocess_data(train)
#     y_train_small, X_train_small = preprocess_data(train_small)

#     test = test.drop(["id"], axis=1)
#     test = test.drop(["ball_pos_dist_ball", "abs_ball_speed"], axis=1)

#     ic(y_train["A"].shape)
#     ic(y_train['B'].shape)
#     ic(X_train.shape)

#     ic(y_train_small["A"].shape)
#     ic(y_train_small['B'].shape)
#     ic(X_train_small.shape)

#     ic(test.shape)
#     for key in test_predictions:
#         print(f"Team: {key} ")
#         prediction, score = run_model(X_train, y_train[key], test, key)
#         test_predictions[key].append(prediction)
#         scores[key].append(score)
#     print('Overall AUC for Team A: ', np.mean(scores['A']))
#     print('Overall AUC for Team B: ', np.mean(scores['B']))
