#!/usr/bin/python
import argparse
import pickle
from os.path import join

import myfuncs as my
import numpy as np
import pandas as pd
from src.utilities import mkdir_if_needed


def run_model_comparison():

    # Load estimates file
    estimates = pd.read_csv(args.estimates_file)

    # Handle potentially different column names
    if not "bic" in estimates.columns:
        estimates["bic"] = estimates["loss"]
    if not "subject" in estimates.columns:
        estimates["subject"] = estimates["subject_id"]
    
    # Summarise BIC
    bics = (
        estimates.pivot_table(values="bic", columns="model", index="subject")
        .describe()
        .T.sort_values("mean")
        .T
    )
    bics.to_csv(join(args.output_dir, "bic_summary.csv"))

    # Make a list of models sorted by mean BIC (low to high)
    models_sorted = bics.columns

    # Count individually best fitting models
    best = estimates.pivot_table(values="bic", columns="model", index="subject").idxmin(
        axis=1
    )
    best_model_counts = best.value_counts().reindex(models_sorted, fill_value=0)
    best_model_counts.name = "count"
    best_model_counts.to_csv(join(args.output_dir, "best_model_counts.csv"))

    # Run BMS
    L = (
        -0.5
        * estimates.pivot_table(values="bic", columns="model", index="subject").T.loc[
            models_sorted
        ]
    )
    bms_result = my.stats.bms(L.values, cores=1, chains=2, draws=5000, tune=3000)
    bms_result["models"] = models_sorted
    with open(join(args.output_dir, "bms_result.pkl"), "wb") as output_file:
        pickle.dump(bms_result, output_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", default=0, type=int, help="Set verbosity (0, 1, >1)."
    )
    parser.add_argument(
        "--estimates-file",
        type=str,
        default="data/processed/estimates.csv",
        help="Relative path to preprocessed estimates file from fit_models.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/model_comparison",
        help="Where to save results.",
    )

    parser.add_argument("--seed", type=int, default=2021, help="Random number seed.")

    args = parser.parse_args()

    mkdir_if_needed(args.output_dir)

    np.random.seed(args.seed)

    run_model_comparison()
