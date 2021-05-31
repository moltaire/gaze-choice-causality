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
    # Load parameter estimates file
    est_rec = pd.read_csv(args.estimates_file_recovery, index_col=0)

    # Load `ddm_fitting` results, to obtain *generating* parameteres
    parameters = [
        "alpha",
        "wp",
        "eta",
        "theta",
        "b_last",
        "v",
        "noise",
    ]
    est_gen = (
        pd.read_csv(
            args.estimates_file_fitting,
            index_col=0,
        )
        .rename({parameter: "gen_" + parameter for parameter in parameters}, axis=1)
        .rename({"model": "gen_model"}, axis=1)
        .drop(["loss", "lossfun"], axis=1)
    )
    est_gen["subject_id"] = est_gen.apply(
        lambda x: f"{x['subject_id']}-{x['gen_model']}", axis=1
    )
    # Combine dataframes
    est = est_rec.merge(est_gen, on=["subject_id"], how="left")[
        ["subject_id", "model", "gen_model", "loss", "lossfun"]
        + parameters
        + ["gen_" + parameter for parameter in parameters]
    ]

    # %% Plot parameter recovery
    bounds = dict(
        alpha=[0, 5],
        wp=[0, 1],
        eta=[0, 1],
        theta=[0, 1],
        b_last=[-0.1, 0.1],
        w_between_altwise=[0, 1],
        w_between_attwise=[0, 1],
        v=[0, 30],
        noise=[0, 5],
    )
    models = ["TwoStageWithin", "TwoStageBetween"]  # est["gen_model"].unique()
    model_labels = models
    df = est
    df["rec_model"] = df["model"]

    width = 4
    height = 4
    fig, axs = plt.subplots(
        len(models),
        len(parameters),
        figsize=my.utilities.cm2inch(len(parameters) * width, len(models) * height),
    )

    for m, (model, model_label) in enumerate(zip(models, model_labels)):

        for p, parameter in enumerate(parameters):
            ax = axs[m, p]

            df_m = df.loc[(df["gen_model"] == model) & (df["rec_model"] == model)]

            # Labels
            if p == 0:
                ax.set_ylabel(f"{model_label}\n\nRecovered")
            if m == (len(models) - 1):
                ax.set_xlabel("Generating")
            if m == 0:
                ax.set_title(parameter)

            # Linear model plot
            gen_values = df_m[f"gen_{parameter}"].values
            rec_values = df_m[f"{parameter}"].values

            if np.any(np.isnan(gen_values)):
                ax.annotate(
                    "N/A",
                    (0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white", edgecolor="black", pad=0.5, boxstyle="round"
                    ),
                )
            else:
                output = my.plots.lm(
                    gen_values,
                    rec_values,
                    ax=ax,
                    run_correlation=True,
                    scatter_kws={"clip_on": True},
                    xrange=bounds[parameter],
                    family="student",
                    sample_kwargs={"cores": 1},
                )

            # Plot diagonal
            ax.plot(
                bounds[parameter],
                bounds[parameter],
                color="black",
                alpha=0.5,
                zorder=-1,
                lw=0.75,
            )

            # Limits
            ax.set_xlim(*bounds[parameter])
            ax.set_ylim(*bounds[parameter])

    plt.tight_layout()
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(args.output_dir, f"ddm_recovery_parameters.{extension}"),
            bbox_inches="tight",
        )

    # %% Plot model recovery
    # Run BMS for each generating model
    mpps = []
    xps = []
    for gen in models:
        L = -0.5 * (
            est.loc[(est["gen_model"] == gen)]
            .pivot_table(index="subject_id", values="loss", columns="rec_model")
            .T.loc[models]
        )

        bms_result = my.stats.bms(L=L.values, cores=1, draws=5000, tune=10000)
        bms_result["models"] = models
        mpps.append(bms_result["r"])
        xps.append(bms_result["xp"])
    mpps = np.vstack(mpps)
    xps = np.vstack(xps)

    # Make a plot
    fig, axs = plt.subplots(figsize=my.utilities.cm2inch(width, height))
    ax = my.plots.model_recovery(mpp=mpps, xp=xps, model_labels=model_labels)
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(args.output_dir, f"ddm_recovery_models.{extension}"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimates-file-recovery", type=str)
    parser.add_argument("--estimates-file-fitting", type=str)
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()
    mkdir_if_needed(args.output_dir)

    main()
