import argparse
from os import mkdir
from os.path import exists, join

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def mkdir_if_needed(path, verbose=True):
    if not exists(path):
        mkdir(path)
    else:
        if verbose:
            print(f"'{path}' already exists.")


def str2bool(v):
    """https://stackoverflow.com/a/43357954"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_idata_results(idata, label, output_dir):
    """Saves InferenceData, summary and traceplot.

    Args:
        idata (arviz.InferenceData): InferenceData object
        label (str): Label to use for filenames
        output_dir (stsr): Output directory
    """
    ## Summary

    summary = az.summary(idata, hdi_prob=0.95)
    for var in summary.index.values:
        if var.endswith("]"):  # skip individual subject parameters
            continue
        summary.loc[var, "P>0"] = np.mean(idata.posterior[var].values > 0)
    summary.to_csv(join(output_dir, f"{label}_summary.csv"))

    ## Traceplot
    az.plot_trace(idata, compact=True)
    plt.savefig(join(output_dir, f"{label}_traceplot.png"))
    plt.close()
    ## InferenceData
    idata.to_netcdf(join(output_dir, f"{label}_idata.nc"))
