
import pandas as pd
import numpy as np
import os

# reduce mem usage
# thanks to : ?.ipynb  for this function


def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def read_train_data():
    files_name = []
    for dirname, _, filenames in os.walk('../data'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            files_name.append(os.path.join(dirname, filename))
    dtypes_df = pd.read_csv(
        '../data/train_dtypes.csv')

    dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}

    cols = list(dtypes.keys())
    train_df = pd.DataFrame({}, columns=cols)
    for i in range(10):
        print(i)
        df_tmp = pd.read_csv(f'{path_to_data}/train_{i}.csv', dtype=dtypes)
        if SAMPLE < 1:
            df_tmp = df_tmp.sample(frac=SAMPLE, random_state=42)

        train_df = pd.concat([train_df, df_tmp])

        del df_tmp

    train_df = reduce_mem_usage(train_df)
    return train_df


def read_test_data():
    dtypes_df = pd.read_csv(
        '../data/test_dtypes.csv/Users/mathieugrosso/Desktop/X-HEC-entrepreneurs/IA-advanced/my_model_api/Kaggle_Competitions/tabular_playground_series/data/test_dtypes.csv')
    dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}
    test = pd.read_csv(
        '../data/test.csv', dtype=dtypes)


def train_model(X, y, test, key):
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
