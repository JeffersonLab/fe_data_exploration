import pandas as pd
import utils


def main():
    pd.set_option('display.max_columns', None)
    df = utils.load_csv("data/nl-rf-ndx-trip-data-2021-11-06_2022-02-08.csv")
    # df = df.sample(frac=0.001)
    # df = load_csv("data/normal-running-with-beam-2021-11-06_2022-02-08.csv", post_trim=0.8)
    # df = load_csv("data/processed_combined-data-log.txt.csv")
    # df = load_csv("data/processed_gradient-scan-2021-11-05_113837.646876.txt.csv")
    # df = load_csv("data/test-2021-11-05.csv")
    # df = load_csv("data/test.csv")
    # df = df.head(1000)

    print("Processing data")
    # Get data in useful subsets
    gmes_cols, gamma_cols, neutron_cols, date_cols = utils.get_cols(df)
    gmes_df, gamma_df, neutron_df, detector_df = utils.get_data_subsets(df, gmes_cols=gmes_cols, gamma_cols=gamma_cols,
                                                                        neutron_cols=neutron_cols, date_cols=date_cols)

    # Summarize the gradient and detector responses
    print("Summarizing GMES and Detector in plot form")
    utils.summarize_gmes(gmes_df, date_cols, split_dates=True)
    # utils.summarize_gmes(gmes_df, date_cols, split_dates=False)
    utils.summarize_detector(gamma_df, date_cols, title="Gamma Radiation rem/hr")
    utils.summarize_detector(neutron_df, date_cols, title="Neutron Radiation rem/hr")

    # Generate lots of plots showing correlation information
    print("Summarizing correlations.")
    utils.summarize_correlations(gmes_df, gamma_df, neutron_df, detector_df, date_cols)


if __name__ == '__main__':
    print("Starting")
    main()
