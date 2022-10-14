from typing import List, Tuple, Optional
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plots import plot_correlations, strip_plot, timeline_facetgrid
from config import cfg


# TODO: Define columns on which to do trimming

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
        msg = f"Trimming EXTREME detector outliers kept less than {post_trim * 100}% of values.  Orig = {orig_len}. "
              f"Trimmed = {trimmed_len}"
        warnings.warn(msg)
        print(msg)

    print("\n\n##### Data prior to trimming ########\n\n")
    print(df.describe())
    print("\n\n##### Data after trimming #######\n\n")
    print(df_trimmed.describe())

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


def make_pair_plot(df: pd.DataFrame, title: str, filename: str = None):
    print(df.head())
    print(df.describe())
    g = sns.pairplot(df, plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'s': 2, 'alpha': 0.5}}, kind='reg')
    g.fig.suptitle(title)

    # Either show the plot or save it to disk
    if cfg['show_plots']:
        print("HERE")
        plt.show()
    elif filename is not None:
        plt.savefig(f"{cfg['out_dir']}/{filename}")

    return g


def summarize_correlations(gmes_df: pd.DataFrame, gamma_df: pd.DataFrame, neutron_df: pd.DataFrame,
                           detector_df: pd.DataFrame, date_cols: List[str], filename: str = "corr") -> None:
    """Presents a summary of correlations between data.

    Plots are either saved to disk or drawn to screen depending on cfg['show_plots']

    Args:
        gmes_df: A DataFrame containing GMES data to be correlated
        gamma_df: A DataFrame containing Gamma data to be correlated
        neutron_df: A DataFrame containing Neutron data to be correlated
        detector_df: A DataFrame of detector data (both Gamma and Neutron) to be correlated.
        date_cols: A list of date columns contained in the supplied DataFrames
        filename: The base filename used to create these files

    """
    gamma_gmes_corr = get_cartesian_correlations(gmes_df.drop(columns=date_cols), gamma_df.drop(columns=date_cols))
    neutron_gmes_corr = get_cartesian_correlations(gmes_df.drop(columns=date_cols), neutron_df.drop(columns=date_cols))
    detector_corr = detector_df.corr()
    gmes_corr = gmes_df.corr()

    print("\n\n######### Gamma vs GMES Correlations ########")
    print(gamma_gmes_corr)
    print("\n\n######### Neutron vs GMES Correlations ########")
    print(neutron_gmes_corr)
    print("\n\n######### Radiation Correlations ########")
    print(detector_corr)
    print("\n\n######### Radiation Correlations ########")
    print(gmes_corr)

    plot_correlations(detector_corr, figsize=(10, 10), spa_kws={'left': 0.2, 'right': 1, 'bottom': 0.2},
                      title="Gamma and Neutron Dose Rate (rem/hr) Correlation")

    # Plot correlations between radiation and cavity gradient
    plot_correlations(gamma_gmes_corr, title="Gamma Radiation (rem/hr) correlated with Cavity GMES (MV/m)")
    plot_correlations(neutron_gmes_corr, title="Neutron Radiation (rem/hr) correlated with Cavity GMES (MV/m)")


def summarize_gmes(df: pd.DataFrame, id_cols: List[str], split_dates: bool = False,
                   filename: str = "summarize_gmes"):
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
    for idx, tmp_df in enumerate(data_sets):
        print(f"Plotting data set {idx} of {len(data_sets)}")
        print(tmp_df.describe())
        # Add a strip plot of individual samples.
        g = strip_plot(tmp_df, title="Observed Cavity GMES MV/m", zorder=1)
        # Font size on the x-tick labels are usually way to big given how many cavities are in a linac
        xticklabels = [tick.get_text() for tick in plt.xticks()[1]]
        g.set_xticklabels(xticklabels, size=7)

    # Either show the plot or save it to disk
    if cfg['show_plots']:
        plt.show()
    elif filename is not None:
        plt.savefig(f"{cfg['out_dir']}/{filename}")


def summarize_detector(df: pd.DataFrame, id_cols: List[str], title: str, timeline_plots: bool = False,
                       filename: str = None):
    # Put this into a more seaborn friendly format
    print(f"Plotting {title}")
    print(f"id_cols: {id_cols}")
    print(df.describe())

    df_melt = df.melt(id_vars=id_cols)

    # Plot out the gamma radiation ranges
    strip_plot(df_melt, title=title, s=2, ylim=None)

    # Either show the plot or save it to disk
    if cfg['show_plots']:
        plt.show()
    elif filename is not None:
        plt.savefig(f"{cfg['out_dir']}/{filename}")

    # Gamma radiation over time
    if timeline_plots:
        timeline_facetgrid(df_melt, hue='variable')
        # Either show the plot or save it to disk
        if cfg['show_plots']:
            plt.show()
        elif filename is not None:
            plt.savefig(f"{cfg['out_dir']}/{filename}-timeline.png")


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


def get_columns_startswith(df: pd.DataFrame, patterns: List[str]):
    """Returns a list of column names from the given DataFrame that start with at least one of the given patterns."""
    cols = df.columns
    out = []
    for pattern in patterns:
        out += cols[cols.str.startswith(pattern)].to_list()

    # Remove duplicates and return the column names
    return sorted(list(set(out)))


def plot_timeline(df: pd.DataFrame, col_startswith: Optional[List[str]] = None, filename: str = None,
                  subplots_adjust_kw: Optional[dict] = None, **kwargs) -> None:
    if col_startswith is None:
        df.set_index('Datetime').drop(columns=['Date']).plot(linestyle='--', marker='o', **kwargs)

    else:
        fig, axs = plt.subplots(nrows=len(col_startswith), ncols=1, figsize=(25, 2 + 4 * len(col_startswith)),
                                sharex=True, sharey=True)
        for idx, pattern in enumerate(col_startswith):
            ax = axs[idx]
            cols = get_columns_startswith(df, [pattern])
            df.set_index('Datetime').drop(columns=['Date'])[cols].plot(ax=ax, linestyle='--', marker='o',
                                                                       title=pattern, **kwargs)

    plt.subplots_adjust(**subplots_adjust_kw)
    if cfg['show_plots']:
        plt.show()
    elif filename is not None:
        plt.savefig(f"{cfg['out_dir']}/{filename}")
