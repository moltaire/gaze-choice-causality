#!/usr/bin/python
import argparse
from os.path import join

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import myfuncs as my
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from src.utilities import mkdir_if_needed, save_idata_results


def run_ddm_analyses():

    # Load estimates from model-fitting split by presentation format
    estimates = pd.read_csv(args.estimates_file)

    # recover subject variable
    estimates["presentation"] = estimates["subject_id"].str.split("-").str[1]
    estimates["subject_id"] = estimates["subject_id"].str.split("-").str[0].astype(int)

    # 1. Comparison of relative fit between presentation formats
    models = ["TwoStageWithin", "TwoStageBetween"]
    bic = (
        estimates.loc[estimates["model"].isin(models)][
            ["subject_id", "presentation", "model", "loss"]
        ]
        .pivot_table(
            values="loss", index=["subject_id", "presentation"], columns="model"
        )
        .reset_index()
    )
    bic["diff"] = bic[models[0]] - bic[models[1]]
    bic.groupby("presentation")[[models[0], models[1]]].describe().T.round(2).to_csv(
        join(args.output_dir, f"bic-summary_by-presentation.csv")
    )
    # Note that negative differences indicate that the within-alternative model fits better in a given setting, and positive differences indicate that the between-alternatives model fits better

    # Compute difference scores between presentation formats
    differences = bic.pivot_table(
        values="diff", columns="presentation", index="subject_id"
    )
    differences["diff"] = differences["alternatives"] - differences["attributes"]
    # Now, negative differences (of differences) indicate that the within-alternative model fit better in alternative-wise presentation than in attribute-wise presentation, consistent with the hypothesis. So the prediction is that these differences are smaller than 0.

    ## 1.1 Run BF t-Tests
    bf = my.stats.ttestbf.one_sample_ttestbf(differences["diff"])
    bf.to_csv(join(args.output_dir, f"ttestbf_relative-fit.csv"))
    bf_directed = my.stats.ttestbf.BayesFactor.extractBF(
        my.stats.ttestbf.BayesFactor.ttestBF(
            x=differences["diff"], nullInterval=[-np.inf, 0]
        )
    )
    bf_directed.to_csv(join(args.output_dir, f"ttestbf-directed_relative-fit.csv"))

    ## 1.2 Run BEST
    differences.to_csv(join(args.output_dir, "best_relative-fit_data.csv"))
    best_idata = my.stats.best.one_sample_best(
        differences["diff"],
        sigma_low=0.001,
        sigma_high=100,
        sample_kwargs=dict(cores=1),
    )
    save_idata_results(
        idata=best_idata,
        label=f"best_relative-fit",
        output_dir=args.output_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", default=0, type=int, help="Set verbosity (0, 1, >1)."
    )
    parser.add_argument(
        "--estimates-file",
        type=str,
        default="models/pilot2/ddm_fitting_by-presentation.csv",
        help="Relative path to estimates file from presentation-wise DDM fitting.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/ddm_analyses",
        help="Where to save results.",
    )

    parser.add_argument("--seed", type=int, default=2021, help="Random number seed.")

    args = parser.parse_args()

    mkdir_if_needed(args.output_dir)

    np.random.seed(args.seed)

    run_ddm_analyses()
