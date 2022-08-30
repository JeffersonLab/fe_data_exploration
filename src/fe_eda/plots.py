import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.axes


def strip_plot(df, title, x='variable', y='value', alpha=0.25, show=False, palette='colorblind', ylim=(-0.5, 22),
               **kwargs) -> matplotlib.axes.Axes:
    """Create a strip plot.  Generally useful for numerical distributions split on categorical data.

    Args:
        df: A tidy (long) form dataframe.  Typically the product of DataFrame.melt method.
        title: Plot title.  Automatically has trimmed value attached
        x: The df column name used as the x-axis
        y: The df column name used as the y-axis
        alpha: Opacity of the points
        palette: Plot color palette
        ylim: y-axis limits.  None lets the package pick.
        show: Should plt.show() be called after creating the plot
    """
    print(df.head())
    print(df.shape)
    g = sns.stripplot(x=x, y=y, data=df, alpha=alpha, palette=palette, **kwargs)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()

    if show:
        plt.show()

    return g


def timeline_facetgrid(df, x='Datetime', y='value', row="Date", col='variable', sharey=True, sharex='row',
                       margin_titles=True, show=False, **kwargs):
    g = sns.FacetGrid(data=df, row=row, col=col, sharey=sharey, sharex=sharex, margin_titles=margin_titles, **kwargs)
    g.map(sns.lineplot, x, y)
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    xformatter = mdates.DateFormatter("%H:%M")
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(xformatter)
    plt.tight_layout()

    if show:
        plt.show()

    return g


def plot_correlations(corr_df, figsize=(60, 10), cmap=sns.diverging_palette(220, 20, as_cmap=True), title="",
                      corr_decimals=1, show=True, spa_kws=None):
    """Plot a heatmap for correlation data.  Intended for large, asymmetric matrices (10x200)."""

    if spa_kws is None:
        spa_kws = {'left': 0.04, 'right': 1.15}

    # Generate figure.
    plt.figure(figsize=figsize)

    # Supply corr values to a single decimal place as annotation labels
    labels = corr_df.applymap(lambda v: round(v, corr_decimals))

    # Create the customized heatmap
    g = sns.heatmap(corr_df,  # Data
                    annot=labels,  # Cell labels
                    annot_kws={'fontsize': 10},  # Cell label size
                    cmap=cmap,  # Color map used for color bar/cells
                    cbar_kws={'pad': 0.01},  # Padding between color bar and plot in fraction of plot axes
                    linewidth=0.01  # Thickness of lines between cells
                    )
    g.set_title(title)
    plt.subplots_adjust(**spa_kws)

    if show:
        plt.show()

    return g


def plot_cavity_sample_time_series(df, gmes_cols, gamma_cols, neutron_cols,
                                   id_cols=('Datetime', 'Date', 'sample_type', 'settle_start')):
    gmes_melt = df[id_cols + gmes_cols].melt(id_vars=id_cols)
    #    neutron_melt = df[id_cols + neutron_cols].melt(id_vars=id_cols)
    #    gamma_melt = df[id_cols + gamma_cols].melt(id_vars=id_cols)

    sns.lineplot(data=gmes_melt, y='value', hue='settle_start')
