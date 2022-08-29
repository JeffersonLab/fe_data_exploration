import os
import pickle
from typing import Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

cfg = {}


def load_csv(file: str, det_ub=2000, det_lb=-1) -> pd.DataFrame:
    df = pd.read_csv(file, na_values="<undefined>")

    # Manage the date/time columns
    df.insert(loc=0, column='Datetime', value=pd.to_datetime(df.Date, format="%Y-%m-%d_%H:%M:%S"))
    df.Date = pd.to_datetime(df.Datetime.dt.date)

    # This will include the lagged and unlagged columns
    gamma_cols = df.columns[df.columns.str.contains("_gDsRt")].to_list()
    neutron_cols = df.columns[df.columns.str.contains("_nDsRt")].to_list()
    det_cols = gamma_cols + neutron_cols

    # The NDX detector numbers have some bizarrely large numbers.  Need to trim out obvious garbage.  Operational
    # values are in [0, 500], with outliers at 1e16.  Cutoff at [-1, 2000] seems safe.
    orig_len = len(df)
    df_trimmed = df[~((df[det_cols] < det_lb) | (df[det_cols] > det_ub)).any(axis=1)]
    trimmed_len = len(df_trimmed)

    if (trimmed_len / orig_len) < 0.95:
        raise RuntimeError(f"Trimming EXTREME detector outliers removed more than 5% of values.  Orig = {orig_len}. "
                           f"Trimmed = {trimmed_len}")

    return df_trimmed


def main():
    mod_file = 'random_forest_model.pkl'

    print("loading data")
    df = load_csv("data/processed_combined-data-log.txt.csv")

    # Generate some list of column names
    cfg['gmes_cols'] = df.columns[df.columns.str.contains("GMES")].to_list()
    cfg['gamma_cols'] = df.columns[df.columns.str.contains("_gDsRt_lag")].to_list()
    # neutron_cols = df.columns[df.columns.str.contains("_nDsRt_lag")].to_list()
    cfg['neutron_cols'] = ['INX1L22_nDsRt_lag-1', 'INX1L23_nDsRt_lag-1', 'INX1L24_nDsRt_lag-1', 'INX1L25_nDsRt_lag-1',
                    'INX1L26_nDsRt_lag-1', 'INX1L27_nDsRt_lag-1']
    date_cols = ['Datetime', 'Date']



    # Split out interesting sections of data
    X = df[cfg['gmes_cols']]
    y = df[cfg['neutron_cols']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=732)

    if os.path.exists(mod_file):
        print("loading model from file")
        with open(mod_file, 'rb') as f:
            mod = pickle.load(f)
    else:
        print("training model")
        mod = RandomForestRegressor()
        mod.fit(X_train, y_train)

        print("saving model")
        with open(mod_file, 'wb') as f:
            pickle.dump(mod, f)

    print("Showing results")
    y_pred_pure = mod.predict(X_test)
    y_pred = pd.DataFrame(y_pred_pure, columns=y_test.columns)

    test_r_sq = mod.score(X_test, y_test)
    train_r_sq = mod.score(X_train, y_train)

    # Collect various data for reporting
    total_gmes = X_test.sum(axis=1)

    err = pd.DataFrame(y_pred-y_test)
    err['TOTAL_GMES'] = total_gmes
    err_melt = err.melt(id_vars='TOTAL_GMES')
    err_melt['type'] = ['err'] * len(err_melt)

    y_test_pure = y_test.copy()
    y_test["TOTAL_GMES"] = total_gmes
    y_test_melt = y_test.melt(id_vars='TOTAL_GMES')
    y_test_melt['type'] = ['test'] * len(y_test_melt)

    y_pred['TOTAL_GMES'] = total_gmes
    y_pred_melt = y_pred.melt(id_vars='TOTAL_GMES')
    y_pred_melt['type'] = ['pred'] * len(y_pred_melt)

    results = pd.concat((y_test_melt, y_pred_melt, err_melt), ignore_index=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    g = sns.FacetGrid(data=results, col='variable', row='type', margin_titles=True)
    g.map_dataframe(sns.scatterplot, hue='type', x='TOTAL_GMES', y='value')
    # sns.scatterplot(data=err, x='TOTAL_GMES', y='value', hue='variable')
    plt.show()


    print(f"Test Statistics:")
    print(f"Test R^2: {test_r_sq}")
    print(f"RMSE:     {metrics.mean_squared_error(y_test_pure, y_pred_pure)}")

    print(f"Test Statistics:")
    print(f"Train R^2: {train_r_sq}")
    print(f"RMSE:     {metrics.mean_squared_error(y_train, mod.predict(X_train))}")


if __name__ == "__main__":
    main()
