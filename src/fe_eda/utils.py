from typing import List, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plots import plot_correlations, strip_plot, timeline_facetgrid


def load_csv(file: str, det_ub=2000, det_lb=-1, post_trim=0.95) -> pd.DataFrame:
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

    if (trimmed_len / orig_len) < post_trim:
        raise RuntimeError(f"Trimming EXTREME detector outliers kept less than {post_trim*100}% of values.  Orig = {orig_len}. "
                           f"Trimmed = {trimmed_len}")

    return df_trimmed


def get_cartesian_correlations(col_df, row_df, fillna=0):
    """Generate a correlation matrix for all combinations of columns between two data frames."""
    corr_df = col_df.apply(lambda col: row_df.corrwith(col).rename(col.name))
    if fillna is not None:
        corr_df.fillna(fillna, inplace=True)

    return corr_df


def trim_dataframe_outliers(df, lbq=0.001, ubq=0.999):
    """Trim the dataframe by removing rows that contain any columns most extreme values.

    A row is excluded if it contains a value that is outside any columns acceptable quantile range [lbq, ubq].

    Args:
        df:  The DataFrame to trim
        lbq: The lower bound quantile.  Applied per column.
        ubq: The upper bound quantile.  Applied per column.

    Returns:
        The trimmed dataframe.
    """
    ub = df.quantile(ubq)
    lb = df.quantile(lbq)
    return df[~((df < lb) | (df > ub)).any(axis=1)]


def summarize_correlations(gmes_df, gamma_df, neutron_df, detector_df, date_cols, pair_plots=False):
    gamma_gmes_corr = get_cartesian_correlations(gmes_df.drop(columns=date_cols), gamma_df.drop(columns=date_cols))
    neutron_gmes_corr = get_cartesian_correlations(gmes_df.drop(columns=date_cols), neutron_df.drop(columns=date_cols))
    detector_corr = detector_df.corr()

    # Plot correlation between gamma and neutron response
    if pair_plots:
        g = sns.pairplot(detector_df,
                         plot_kws={
                             'line_kws': {'color': 'red'},
                             'scatter_kws': {'s': 2, 'alpha': 0.5}}
                         , kind='reg')

        g.fig.suptitle("Gamma vs. Neutron Dose Rate (rem/hr)")
        plt.show()

    plot_correlations(detector_corr, figsize=(10, 10), spa_kws={'left': 0.2, 'right': 1, 'bottom': 0.2},
                      title="Gamma and Neutron Dose Rate (rem/hr) Correlation")

    # Plot correlations between radiation and cavity gradient
    plot_correlations(gamma_gmes_corr, title="Gamma Radiation (rem/hr) correlated with Cavity GMES (MV/m)")
    plot_correlations(neutron_gmes_corr, title="Neutron Radiation (rem/hr) correlated with Cavity GMES (MV/m)")


def summarize_gmes(df, id_cols, split_dates=True):
    """Summarize a DataFrame containing only 'ID' and GMES columns.  Assumes a category column named 'Date' exists."""

    df_melt = df.melt(id_vars=id_cols)

    # Optionally split out individual days of data or keep it all together
    data_sets = []
    if split_dates:
        for date in df_melt.Date.unique():
            data_sets.append(df_melt.query(f"Date=='{date}'"))
    else:
        data_sets.append(df_melt)

    print("About to start plotting")
    # Make a plot for each data set
    for tmp_df in data_sets:
        # Add a strip plot of individual samples.
        strip_plot(tmp_df, title="Observed Cavity GMES MV/m", zorder=1)
        plt.show()


def summarize_detector(df, id_cols, title, timeline_plots=False):
    # Put this into a more seaborn friendly format
    print(df.shape)
    print(df.columns)
    print(id_cols)
    df_melt = df.melt(id_vars=id_cols)

    # Plot out the gamma radiation ranges
    strip_plot(df_melt, title=title, s=2, ylim=None)

    # Gamma radiation over time
    if timeline_plots:
        timeline_facetgrid(df_melt, hue='variable')


def get_cols(df: pd.DataFrame, rad_suffix="") -> Tuple[List[str], List[str], List[str], List[str]]:
    """Get a standard column set for the DataFrame."""
    gmes_cols = df.columns[df.columns.str.contains("GMES")].to_list()
    gamma_cols = df.columns[df.columns.str.contains(f"_gDsRt{rad_suffix}")].to_list()
    neutron_cols = df.columns[df.columns.str.contains(f"_nDsRt{rad_suffix}")].to_list()
    date_cols = ['Datetime', 'Date']
    return gmes_cols, gamma_cols, neutron_cols, date_cols


def get_data_subsets(df: pd.DataFrame, gmes_cols: List[str], gamma_cols: List[str], neutron_cols: List[str],
                     date_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split out interesting sections of data
    gmes_df = df[date_cols + gmes_cols]
    gamma_df = df[date_cols + gamma_cols]
    neutron_df = df[date_cols + neutron_cols]
    detector_df = df[gamma_cols + neutron_cols]
    return gmes_df, gamma_df, neutron_df, detector_df


