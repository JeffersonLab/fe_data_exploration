import argparse
import math
import os
import sys
from datetime import datetime

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

    args = parser.parse_args()
    cfg['file'] = args.file
    cfg['operation'] = args.operation
    cfg['show_plots'] = args.show_plots

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
                do_summary(cfg['file'])
        else:
            do_summary(cfg['file'])


def do_summary(file: str) -> None:

    print("\n\n##########################")
    print("Loading data")
    print("##########################")
    df = utils.load_csv(file, det_lb=-math.inf, det_ub=math.inf)

    # Get data in useful subsets
    print("\n\n##########################")
    print("Processing data")
    print("##########################")
    gmes_cols, gamma_cols, neutron_cols, date_cols = utils.get_cols(df, rad_suffix="_lag-1",)
    gmes_df, gamma_df, neutron_df, detector_df = utils.get_data_subsets(df, gmes_cols=gmes_cols, gamma_cols=gamma_cols,
                                                                        neutron_cols=neutron_cols, date_cols=date_cols)

    # Summarize the gradient and detector responses
    if cfg['show_plots']:
        plt.ion()

    print("\n\n##########################")
    print("Summarizing GMES")
    print("##########################")
    plt.figure(figsize=(20, 5))
    utils.summarize_gmes(gmes_df, date_cols, split_dates=False)

    print("\n\n##########################")
    print("Summarizing Gamma Radiation")
    print("##########################")
    plt.figure()
    utils.summarize_detector(gamma_df, date_cols, title="Gamma Radiation rem/hr", filename="gamma_summary.png")

    print("\n\n##########################")
    print("Summarizing Neutron Radiation")
    print("##########################")
    plt.figure()
    if cfg['show_plots']:
        plt.ioff()
    utils.summarize_detector(neutron_df, date_cols, title="Neutron Radiation rem/hr", filename="neutron_summary.png")

    return

    # Generate lots of plots showing correlation information
    print("\n\n##########################")
    print("Summarizing correlations.")
    print("##########################")
    utils.summarize_correlations(gmes_df, gamma_df, neutron_df, detector_df, date_cols)


if __name__ == '__main__':
    main()
