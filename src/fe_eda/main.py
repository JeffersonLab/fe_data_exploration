import argparse
import math
import os
import sys
from datetime import datetime
from typing import List

from matplotlib import pyplot as plt
import pandas as pd
import utils
from config import cfg

plt.style.use('seaborn-whitegrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def main():
    parser = argparse.ArgumentParser(description="Do exploratory data analysis (EDA) related to fe_daq runs.")
    parser.add_argument('-f', '--file', required=True, type=str,
                        help="The input data file to do EDA on.  Recommend it to be in data/")
    parser.add_argument('-p', '--show-plots', default=False, action='store_true',
                        help="Displays plots and output instead of saving the to disk.")
    parser.add_argument('-o', '--operation', type=str, choices=['summary'], default='summary',
                        help='What type of EDA to perform.  "summary" creates a collection of distribution plots and'
                             'reports')
    parser.add_argument('-z', '--rf-zones', type=str, nargs='+', default=['R1M', 'R1N', 'R1O', 'R1P'],
                        help="The EPICSNames for the zones that were being controlled in this data, e.g., R1M")

    args = parser.parse_args()
    cfg['file'] = args.file
    cfg['operation'] = args.operation
    cfg['show_plots'] = args.show_plots
    cfg['rf_zones'] = args.rf_zones

    out_dir = f"out/{os.path.basename(args.file)}-{args.operation}-{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    if cfg['show_plots']:
        out_dir = None
    cfg['out_dir'] = out_dir

    if cfg['out_dir'] is None:
        cfg['report_file'] = None
    else:
        cfg['report_file'] = f"{cfg['out_dir']}/{cfg['operation']}.txt"

    if not os.path.isfile(args.file):
        print(f"Error: input file not found '{args.file}'")
        exit(1)

    if cfg['out_dir'] is not None:
        if os.path.isdir(cfg['out_dir']):
            print(f"Error: output directory already exists '{cfg['out_dir']}'.")
            exit(1)

        # Make the output directory for this job
        os.makedirs(cfg['out_dir'])

    # Do the requested operation
    if cfg['operation'] == 'summary':
        # Redirect standard out to report file if one is given
        if cfg['report_file'] is not None:
            with open(cfg['report_file'], "w") as sys.stdout:
                do_summary(cfg['file'], cfg['rf_zones'])
        else:
            do_summary(cfg['file'], cfg['rf_zones'])



def do_summary(file: str, rf_daq_zones: List[str]) -> None:
    print("\n\n##########################")
    print("Loading data")
    print("##########################")
    df = utils.load_csv(file, det_lb=-math.inf, det_ub=math.inf)

    # Get data in useful subsets
    print("\n\n##########################")
    print("Processing data")
    print("##########################")
    gmes_cols, gamma_cols, neutron_cols, date_cols = utils.get_cols(df, rad_suffix="_lag-1", )
    gmes_df, gamma_df, neutron_df, detector_df = utils.get_data_subsets(df, gmes_cols=gmes_cols, gamma_cols=gamma_cols,
                                                                        neutron_cols=neutron_cols, date_cols=date_cols)
    gmes_study_cols = utils.get_columns_startswith(gmes_df, cfg['rf_zones'])

    # This lets the plots all be plotted at once
    if cfg['show_plots']:
        plt.ion()

    print("\n\n##########################")
    print("Timeline Plots")
    print("##########################")
    utils.plot_timeline(gmes_df, rf_daq_zones,
                        subplots_adjust_kw={'left': 0.05, 'right': 0.95, 'top': 0.95, 'bottom': 0.075, 'hspace': 0.2},
                        ylim=(0, 22), ylabel="MV/m", alpha=0.25, filename='timeline-gmes.png')

    utils.plot_timeline(neutron_df, None,
                        subplots_adjust_kw={'left': 0.1, 'right': 0.95, 'top': 0.95, 'bottom': 0.15},
                        ylabel="rem/h", title="Neutron Timeline", alpha=0.25, filename='timeline-neutron.png')

    utils.plot_timeline(gamma_df, None,
                        subplots_adjust_kw={'left': 0.1, 'right': 0.95, 'top': 0.95, 'bottom': 0.15},
                        ylabel="rem/h", title="Gamma Timeline", alpha=0.25, filename='timeline-gamma.png')

    print("\n\n##########################")
    print("Summarizing GMES")
    print("##########################")
    plt.figure(figsize=(20, 5))
    utils.summarize_gmes(gmes_df, date_cols, split_dates=False, filename="distribution_gmes.png")

    print("\n\n##########################")
    print("Summarizing Gamma Radiation")
    print("##########################")
    plt.figure()
    utils.summarize_detector(gamma_df, date_cols, title="Gamma Radiation rem/hr", filename="distribution_gamma.png")

    print("\n\n##########################")
    print("Summarizing Neutron Radiation")
    print("##########################")
    plt.figure()
    utils.summarize_detector(neutron_df, date_cols, title="Neutron Radiation rem/hr",
                             filename="distribution_neutron.png")

    # Generate lots of plots showing correlation information
    print("\n\n##########################")
    print("Summarizing correlations.")
    print("##########################")

    print("\n\n######### GMES Correlations ########")
    print(gmes_study_cols)
    gmes_corr = gmes_df[gmes_study_cols].reset_index(drop=True).corr()
    print(gmes_corr)
    plt.figure(figsize=(15, 15))
    utils.plot_correlations(gmes_corr, spa_kws={'left': 0.2, 'right': 1, 'bottom': 0.2},
                            title="GMES correlations", filename='corr_gmes.png')

    print("\n\n######### Gamma vs GMES Correlations ########")
    gamma_gmes_corr = utils.get_cartesian_correlations(gmes_df.drop(columns=date_cols),
                                                       gamma_df.drop(columns=date_cols))
    print(gamma_gmes_corr)
    plt.figure(figsize=(60, 10))
    utils.plot_correlations(gamma_gmes_corr, spa_kws={'left': 0.05, 'right': 1.1, 'bottom': 0.2},
                            title="Gamma Radiation (rem/hr) correlated with Cavity GMES (MV/m)",
                            filename='corr_gamma_gmes.png')

    print("\n\n######### Neutron vs GMES Correlations ########")
    neutron_gmes_corr = utils.get_cartesian_correlations(gmes_df.drop(columns=date_cols),
                                                         neutron_df.drop(columns=date_cols))
    print(neutron_gmes_corr)
    plt.figure(figsize=(60, 10))
    utils.plot_correlations(neutron_gmes_corr, spa_kws={'left': 0.05, 'right': 1.1, 'bottom': 0.2},
                            title="Neutron Radiation (rem/hr) correlated with Cavity GMES (MV/m)",
                            filename='corr_neutron_gmes.png')

    print("\n\n######### Detector Correlations ########")
    neutron_corr = neutron_df.corr()
    print("Neutron-Neutron Correlation", neutron_corr)
    plt.figure(figsize=(7, 7))
    utils.plot_correlations(neutron_corr, spa_kws={'left': 0.275, 'right': 1, 'bottom': 0.275},
                            title="Neutron Dose Rate (rem/hr) Correlation", filename='corr_neutron.png')

    gamma_corr = gamma_df.corr()
    print("Gamma-Gamma Correlation", gamma_corr)
    plt.figure(figsize=(7, 7))
    utils.plot_correlations(gamma_corr, spa_kws={'left': 0.275, 'right': 1, 'bottom': 0.275},
                            title="Gamma Dose Rate (rem/hr) Correlation", filename='corr_gamma.png')

    n_vs_g_corr = utils.get_cartesian_correlations(neutron_df.drop(columns=date_cols), gamma_df.drop(columns=date_cols))
    print("Neutron-Gamma Correlation", n_vs_g_corr)
    plt.figure(figsize=(7, 7))
    utils.plot_correlations(n_vs_g_corr, spa_kws={'left': 0.275, 'right': 1, 'bottom': 0.275},
                            title="Neutron vs Gamma Dose Rate (rem/hr) Correlation",
                            filename='corr_neutron_vs_gamma.png')

    detector_corr = detector_df.corr()
    print(detector_corr)
    plt.figure(figsize=(15, 15))
    utils.plot_correlations(detector_corr, spa_kws={'left': 0.2, 'right': 1, 'bottom': 0.2},
                            title="Gamma and Neutron Dose Rate (rem/hr) Correlation", filename='corr_detector.png')

    # Free the process on user input
    if cfg['show_plots']:
        plt.show()
        response = input("Press Enter to exit program and close all plots.")


if __name__ == '__main__':
    main()
