# -*- coding: utf-8 -*-
import argparse
import pickle
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import myfuncs as my
import numpy as np
import pandas as pd
from src.utilities import mkdir_if_needed

matplotlib = my.utilities.set_mpl_defaults(matplotlib)


def main():
    # Load parameter estimates file, best fitting model counts, bms-results
    estimates = pd.read_csv(args.estimates_file)

    # Handle potentially different column names
    if not "bic" in estimates.columns:
        estimates["bic"] = estimates["loss"]
    if not "subject" in estimates.columns:
        estimates["subject"] = estimates["subject_id"]

    models_sorted = (
        estimates.pivot_table(values="bic", columns="model", index="subject")
        .describe()
        .T.sort_values("mean")
        .T.columns
    )

    best_model_counts = pd.read_csv(args.best_model_counts_file, index_col=0)["count"]

    with (open(args.bms_result_file, "rb")) as file:
        bms_result = pickle.load(file)

    # Figure parameters
    model_colors = {model: f"C{m}" for m, model in enumerate(models_sorted)}

    width = 9
    height = 6

    fig, axs = plt.subplots(
        3,
        1,
        figsize=my.utilities.cm2inch(width, 3 * height),
    )

    # Bar plot of best fitting models
    ax = axs[0]
    x = np.arange(len(best_model_counts))
    ax.bar(
        x,
        best_model_counts[models_sorted],
        color=[model_colors[model] for model in models_sorted],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        models_sorted,
        rotation=90,
        ha="center",
    )
    ax.set_ylabel("N participants")
    ax.set_title("Ind. best models")

    # Exceedance probabilities
    ax = axs[1]
    x = np.arange(len(models_sorted))
    ax.bar(
        x,
        bms_result["xp"],
        color=[model_colors[model] for model in models_sorted],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        models_sorted,
        rotation=90,
        ha="center",
    )
    ax.set_ylim(0, 1)
    ax.set_title("Exceedance probability")
    ax.set_ylabel("xp")

    # Violin plot of BICs
    ax = my.plots.violin(
        estimates.pivot_table(index="subject", columns="model", values="bic")[
            models_sorted
        ],
        ax=axs[2],
    )
    ax.set_xlabel(None)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
    )
    ax.set_ylabel("BIC")
    ax.set_title("BIC distributions")
    fig.tight_layout()
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(args.output_dir, f"{args.filename}{args.label}.{extension}"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimates-file", type=str)
    parser.add_argument("--best-model-counts-file", type=str)
    parser.add_argument("--bms-result-file", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--filename", type=str, default="model_comparison")
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()
    if args.label != "":
        args.label = "_" + args.label
    mkdir_if_needed(args.output_dir)

    main()
