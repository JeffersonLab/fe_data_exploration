import os
import pickle
from typing import Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

# Pandas has some options for how much of a table to display.  Show all columns.
pd.set_option('display.max_columns', None)

# The single cavity measured gradient columns.  Excludes the R1QX PV which is their sum.
# gmes_cols = ["R1Q1GMES", "R1Q2GMES", "R1Q3GMES", "R1Q4GMES", "R1Q5GMES", "R1Q6GMES", "R1Q7GMES", "R1Q8GMES"]
gmes_cols = []
# for z in '2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P'.split(' '):
for z in 'M N O P'.split(' '):
    for c in range(1, 9):
        gmes_cols.append(f"R1{z}{c}GMES")

# The TN has data from four detectors, but most of the action happens at one.  Let's start there.
# rad_cols = ["SLD1L00RAd", "SLD1L01RAd", "SLD1L02RAd", "SLD1L03RAd"]
#rad_cols = ["SLD1L03RAd"]
rad_cols = ["INX1L22_nDsRt_lag-1", "INX1L23_nDsRt_lag-1", "INX1L24_nDsRt_lag-1", "INX1L25_nDsRt_lag-1",
            "INX1L26_nDsRt_lag-1", "INX1L27_nDsRt_lag-1"]

# The FE onsets for 1L26.  Later you may want to include 1L25 in the analysis.  TN lists 0 for cavity one - I'm not sure
# what that actually means.  Always field emits or never?  Let's assume always for now.
# fe_onsets = [0, 15.5, 11.5, 10.1, 14.4, 18.0, 11.4, 7.9]
fe_onsets = pd.read_csv('data/fe_onset-processed-2021-08.tsv', sep='\t', header=None, index_col=0).transpose().values.tolist()[0]


def fit_regression(mod_constructor: callable, df: DataFrame, X: DataFrame, y: DataFrame, X_train: DataFrame,
                   X_test: DataFrame, y_train: DataFrame, y_test: DataFrame) -> None:
    """Perform the standard process of training a model, print it's score, and generating a summary plot."""
    print(f"################\nModel: {mod_constructor.__name__}\nTraining set size: {len(X_train)}\n")

    # This is only useful when working with more than one detector
    # for i in range(len(rad_cols)):
    #     mod = mod_constructor()
    #     mod.fit(X_train, y_train.iloc[:, i])
    #     print(f"{y.columns[i]}: Train R2: {mod.score(X_train, y_train.iloc[:, i])}")
    #     print(f"{y.columns[i]}: Test R2: {mod.score(X_test, y_test.iloc[:, i])}")
    #
    # print("Fitting all at once.")

    mod_file = f"{mod_constructor.__name__}.pkl"
    if os.path.exists(mod_file):
        print("loading model from file")
        with open(mod_file, 'rb') as f:
            model = pickle.load(f)
    else:
        print("training model from scratch")
        model = mod_constructor()
        print("Start Fitting model")
        model.fit(X_train, y_train)
        print("Done Fitting model")

        print("saving model")
        with open(mod_file, 'wb') as f:
            pickle.dump(model, f)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(f"Avg. Train R2: {model.score(X_train, y_train)}")
    print(f"Avg. Test MSE: {mean_squared_error(y_train, y_pred_train)}")
    print(f"Avg. Test MAE: {mean_absolute_error(y_train, y_pred_train)}")

    print(f"Avg. Test R2: {model.score(X_test, y_test)}")
    print(f"Avg. Test MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"Avg. Test MAE: {mean_absolute_error(y_test, y_pred)}")
    print("")

    # The DataFrames that come from model.predict lose the index info.  Get the zone GMES and reset the index so it will
    # match those.
    df['SUMMED_GMES'] = df[gmes_cols].sum(axis=1)
    test_zone_gmes = df.loc[X_test.index.tolist(), 'SUMMED_GMES'].reset_index(drop=True)
    train_zone_gmes = df.loc[X_train.index.tolist(), 'SUMMED_GMES'].reset_index(drop=True)

    # Only show charts for test test data
    # y_pred = pd.DataFrame(model.predict(X_test), columns=y.columns)
    print(f"prediction shape: {y_pred.shape}")
    print(f"y shape: {y.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_train shape: {y_train.shape}")

    y_test1 = y_test.copy().reset_index(drop=True)
    y_test1['SUMMED_GMES'] = test_zone_gmes
    y_test1['type'] = ['Observed'] * len(y_test1)

    y_pred = pd.DataFrame(y_pred, columns=y.columns)
    y_pred['SUMMED_GMES'] = test_zone_gmes
    y_pred['type'] = ["Predicted"] * len(y_pred)

    errors = y_test.reset_index(drop=True) - y_pred
    errors['type'] = ['Error'] * len(errors)
    errors['SUMMED_GMES'] = test_zone_gmes

    out = pd.concat((y_test1, y_pred, errors))
    out_melt = out.melt(id_vars=['SUMMED_GMES', 'type'], var_name="var")

    # y_pred = pd.DataFrame(model.predict(X_test), columns=y.columns)
    # y_pred['SUMMED_GMES'] = test_zone_gmes
    # y_pred['type'] = ["pred_test"] * len(y_pred)
    #
    # y_pred1 = pd.DataFrame(model.predict(X_train), columns=y.columns)
    # y_pred1['SUMMED_GMES'] = train_zone_gmes
    # y_pred1['type'] = ["pred_train"] * len(y_pred1)
    #
    # errors = y_test.reset_index(drop=True) - y_pred
    # errors['type'] = ['test_err'] * len(errors)
    # errors['SUMMED_GMES'] = test_zone_gmes

    # y_act = y_test.copy().reset_index(drop=True)
    # y_act['type'] = ['obs'] * len(y_act)
    # y_act['SUMMED_GMES'] = test_zone_gmes
    #
    # out = pd.concat((y_act, y_pred, y_pred1, errors))
    # out_melt = out.melt(id_vars=['SUMMED_GMES', 'type'], var_name="var")

    plt.figure(figsize=(10, 10))
    data_df = out_melt
    g = sns.FacetGrid(data=data_df, col='var', row='type', hue='type')

    # plt.figure(figsize=(1, 1))
    # data_df = out_melt.query("var == 'INX1L25_nDsRt_lag-1'")
    # g = sns.FacetGrid(data=data_df, col='var', row='type', hue='type', aspect=2.5, height=2)
    g = sns.FacetGrid(data=data_df, col='var', row='type', hue='type', aspect=1.5, height=1.5)
    g.map_dataframe(sns.scatterplot, x='SUMMED_GMES', y='value', alpha=0.2)
    g.set_axis_labels("Total C100 Gradient (MV/m)", "rem/h")
    for (row_key, col_key), ax in g.axes_dict.items():
        ax.set_title(f"\n{col_key[3:7]} {row_key}", loc='center', pad=-20)
        # ax.title.set_position([0.5, 0])
    #g.add_legend()
    # g.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1)
    g.fig.subplots_adjust(top=0.9)
    # g.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.25)
    g.fig.suptitle("Neutron Radiation\nvs Aggregate C100 Gradient")
    # g.fig.suptitle(
    #     f"Sum_i(a_i*ug + b_i * dg + 2**(gmes_i - fe_i, 0) * (c_i + d_i*ug_i + e_i*dg_i)\n"
    #     f"(n={len(X)}, train={len(X_train)})")
    plt.show()


def get_X_y(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Create a set of model inputs and labels from a DataFrame.

    We think that radiation is exponentially related to the source gradient over FE onset and linearly related to the
    gradients in neighboring cavities.  Create columns that roughly reflect those components.

    Arguments:
        df:  A DataFrame containing, at a minimum, the original Legg data.

    Returns:
        X, y: The model inputs (X) and the training labels (y).
    """
    accel_cols = []

    # Calculate the gradient experienced by FE electrons leaving a cavity up and down stream
    # column_names = []
    # columns = []
    for col in gmes_cols:
        ug = np.zeros(shape=(len(df),))
        dg = np.zeros(shape=(len(df),))
        # Make a pass and find the upstream experienced gradient
        for other in gmes_cols:
            if col == other:
                accel_cols.append(f"{col[0:4]}_UGMES")
                df[f"{col[0:4]}_UGMES"] = ug
                # column_names.append(f"{col[0:4]}_UGMES")
                # columns.append(pd.Series(ug, name=f"{col[0:4]}_UGMES"))
                break
            ug += df[other]

        # Make a pass and find the downstream experienced gradient
        past = False
        for other in gmes_cols:
            if not past:
                if col == other:
                    past = True
                continue
            dg += df[other]
        accel_cols.append(f"{col[0:4]}_DGMES")
        df[f"{col[0:4]}_DGMES"] = dg
        # column_names.append(f"{col[0:4]}_DGMES")
        # columns.append(pd.Series(dg, name=f"{col[0:4]}_DGMES"))

    # Collect all of those columns back into one data frame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # df_ud_gmes = pd.concat(columns, axis=1)
    # print(f"df_ud_gmes nas {df_ud_gmes.isna().sum().max()}")
    # print(f"{df_ud_gmes[df_ud_gmes.isna().any(axis=1)]}")

    # Make an X with columns for e**(gmes-fe), ug, dg, and interactions
    X = np.exp2(df[gmes_cols].apply(lambda x: x - fe_onsets, axis=1).applymap(lambda x: max(0, x)))
    X.columns = [f"{col}_trans" for col in gmes_cols]
    X[accel_cols] = df[accel_cols]
    # X[accel_cols] = df_ud_gmes[accel_cols]
    for col in gmes_cols:
        cav = col[0:4]
        X[f"{col}_trans_UG"] = X[f"{col}_trans"] * X[f"{cav}_UGMES"]
        X[f"{col}_trans_DG"] = X[f"{col}_trans"] * X[f"{cav}_DGMES"]

    y = df[rad_cols].copy()
    X = X.copy()

    return X, y


def create_label_columns(df: DataFrame) -> DataFrame:
    """Generate useful derivative columns for training or exploratory analysis.  Returns a modified copy.

    Arguments:
        df: A DataFrame contain at a minimum the original data from legg-data_2016-08-15

    Returns:
        A copy of df with the additional columns.
    """
    is_fe = df[gmes_cols] < fe_onsets
    print(is_fe.iloc[:, 0])
    out = df.copy()
    for i, col in enumerate(gmes_cols):
        out[f"{col[0:4]}_Is_FE"] = is_fe.iloc[:, i]

    return out


def display_summary_plots(df: DataFrame) -> None:
    """Create and display (or maybe save to file) some descriptive summary plots."""
    pass


def print_summary_reports(df: DataFrame) -> None:
    """Print out summary tables or other reports of the data.  Maybe GMES columns or label correlations too."""
    pass


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


def main() -> None:
    """Main routine for reading data, preparing it for training and training some models."""
    # df = pd.read_csv('legg-data_2016-08-15.csv', index_col=False)
    df = load_csv("data/processed_combined-data-log.txt.csv")
    X, y = get_X_y(df)
    is_fe = df[gmes_cols] < fe_onsets

    # 2**6 = 64 (which is what we have for stratification
    # for mod in [LinearRegression, RandomForestRegressor]:
    for mod in [RandomForestRegressor]:
        # You can try this with different sized training sets to see change in performance.
        # for i in [int(len(X) * 0.7), 1000, 100, 64, 32]:
        for i in [0.7]:
            # Only use is_fe[2:8] since we don't have the right mix of data for better. R1Q1 is bypassed, R1Q2 seemed to
            # track closely with another cavity.
            # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i, stratify=is_fe.iloc[:, 2:8])
            print("Splitting data")

            # We want to keep samples from the same set points in either test or train, and not mix them.  Since they
            # are likely similar, it could be polluting the independence of the train and test sets.
            gss = GroupShuffleSplit(train_size=i, random_state=732, n_splits=2)
            train_idx, test_idx = next(gss.split(X, y, groups=df.settle_start))

            X_train = X.iloc[train_idx, :]
            X_test = X.iloc[test_idx, :]
            y_train = y.iloc[train_idx, :]
            y_test = y.iloc[test_idx, :]
            #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i, random_state=732)

            print("Training model")
            fit_regression(mod, df, X, y, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
