# -*- coding: utf-8 -*-
import argparse
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import myfuncs as my
import numpy as np
import pandas as pd
from src.utilities import mkdir_if_needed

matplotlib = my.utilities.set_mpl_defaults(matplotlib)


def plot_psychometric(
    choices, ax=None, bins=None, bin_labels=None, alpha=1, **plot_kwargs
):
    if ax is None:
        ax = plt.gca()
    if bins is None:
        bins = np.linspace(choices["delta_ev"].min(), choices["delta_ev"].max(), 6)

    # Bin data
    choices["delta_ev_binned"] = pd.cut(choices["delta_ev"], bins=bins)

    # Compute mean ± SEM values
    data = (
        choices.groupby(["subject_id", "delta_ev_binned"])["choose_higher_p"]
        .mean()
        .reset_index()
        .groupby("delta_ev_binned")["choose_higher_p"]
        .agg(["mean", "sem"])
    )
    if bin_labels is None:
        bin_labels = data.index

    # Plot
    x = np.arange(len(data.index))
    ax.plot(x, data["mean"], "--o", clip_on=False, alpha=alpha, **plot_kwargs)
    ax.vlines(
        x,
        data["mean"].values - data["sem"].values,
        data["mean"].values + data["sem"].values,
        color="black",
        alpha=alpha,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")
    ax.set_xlabel(r"EV$_{higher~p}$ - EV$_{higher~m}$")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(Choose higher p)")
    return ax


def make_psychometric_figure(choices, bins):
    bin_centers = np.array(0.5 * (bins[:-1] + bins[1:]))

    fig, axs = plt.subplots(1, 2, figsize=my.utilities.cm2inch(9, 5), sharey=True)

    # Duration manipulation
    axs[0] = plot_psychometric(
        choices, bins=bins, bin_labels=bin_centers, ax=axs[0], alpha=0.5
    )
    for duration_favours in ["higher_p", "higher_m"]:
        axs[0] = plot_psychometric(
            choices.loc[choices["duration_favours_str"] == duration_favours].copy(),
            bins=bins,
            bin_labels=bin_centers,
            ax=axs[0],
            alpha=0.5,
            label=duration_favours.replace("_", " "),
        )
    axs[0].legend(title="Favoured")
    axs[0].set_title("Duration manipulation")

    # Sequence manipulation
    axs[1] = plot_psychometric(
        choices, bins=bins, bin_labels=bin_centers, ax=axs[1], alpha=0.5
    )
    for last_stage_favours in ["higher_p", "higher_m"]:
        axs[1] = plot_psychometric(
            choices.loc[choices["last_stage_favours_str"] == last_stage_favours].copy(),
            bins=bins,
            bin_labels=bin_centers,
            ax=axs[1],
            alpha=0.5,
            label=last_stage_favours.replace("_", " "),
        )
    axs[1].set_ylabel(None)
    axs[1].yaxis.set_tick_params(labelbottom=True)
    axs[1].legend(title="Last stage\nfavoured")
    axs[1].set_title("Sequence manipulation")

    # Panel labels
    fig.tight_layout(w_pad=4)
    my.utilities.label_axes(fig, fontweight="bold", fontsize=8, loc=(-0.3, 0.975))

    return fig, axs


def plot_individual_changes(data, columns=["higher_m", "higher_p"], ax=None):
    if ax is None:
        ax = plt.gca()

    x = np.arange(len(columns))

    for index, data_s in data.iterrows():
        y = [data_s.loc[column] for column in columns]

        # Plot lines
        ax.plot(
            x,
            y,
            "-k",
            alpha=0.1,
            zorder=0,
        )
        # Plot some white dots without edge and with full alpha to hide lines overlapping with transparent dots
        ax.plot(
            x,
            y,
            "ow",
            alpha=1,
            markersize=6,
            zorder=1,
            linewidth=0,
            markeredgecolor="none",
            clip_on=False,
        )
        # Plot dots
        ax.plot(
            x,
            y,
            "ok",
            alpha=0.1,
            zorder=2,
        )

    # Plot means
    means = data[columns].mean(axis=0)
    sems = data[columns].std(axis=0) / np.sqrt(len(data[columns]))
    ax.plot(x, [means.loc[column] for column in columns], "-o")
    ax.vlines(
        x,
        [means.loc[column] - sems.loc[column] for column in columns],
        [means.loc[column] + sems.loc[column] for column in columns],
        color="black",
    )

    # Ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(columns)

    return ax


def make_individual_change_figure(args, columns):
    fig, axs = plt.subplots(2, 2, figsize=my.utilities.cm2inch(7, 9), sharey=True)

    for c, condition in enumerate(["duration", "sequence"]):
        for p, presentation in enumerate(["alternatives", "attributes"]):

            ax = axs[c, p]

            # Load data that was used for the corresponding statistical analysis:
            data = pd.read_csv(
                join(
                    args.choice_analyses_dir,
                    f"best_{condition}_{presentation}_data.csv",
                ),
                index_col=0,
            )

            # Plot
            plot_individual_changes(data, columns=columns, ax=ax)
            ax.set_xlim(-0.25, 1.25)
            ax.set_ylim(0, 1)
            ax.set_xticklabels([column.replace("_", " ") for column in columns])
            if c == 0:
                ax.set_title(f"{presentation[:-1].capitalize()}-wise")
            else:
                ax.set_xlabel("Favoured")

            # Read BEST result
            summary = pd.read_csv(
                join(
                    args.choice_analyses_dir,
                    f"best_{condition}_{presentation}_summary.csv",
                ),
                index_col=0,
            )
            annotation = (
                f"$d$ = {summary.loc['d', 'mean']:.2f} "
                + f"[{summary.loc['d', 'hdi_2.5%']:.2f}, "
                + f"{summary.loc['d', 'hdi_97.5%']:.2f}]"
            )

            my.utilities.horizontal_text_line(
                annotation, 0, 1, 0.95, lineTextGap=0.01, ax=ax, fontsize=4
            )

            # Ticks
            if p == 1:
                ax.yaxis.set_tick_params(labelbottom=True)
            else:
                ax.set_ylabel(f"{condition.capitalize()}\n\nP(Choose higher p)")

    plt.tight_layout(h_pad=3, w_pad=3)
    my.utilities.label_axes(fig, fontweight="bold", fontsize=8, loc=(-0.475, 0.975))
    return fig, axs


def plot_regression_weights(
    summary,
    predictors,
    predictor_labels=None,
    includecolor="slategray",
    excludecolor="mediumaquamarine",
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    if predictor_labels is None:
        predictor_labels = {predictor: predictor for predictor in predictors}

    means = summary.loc[predictors, "mean"].values
    hdi_lower = summary.loc[predictors, "hdi_2.5%"].values
    hdi_upper = summary.loc[predictors, "hdi_97.5%"].values
    hdi_excludes_zero = (hdi_lower > 0) | (hdi_upper < 0)
    color = np.array([includecolor, excludecolor])[hdi_excludes_zero.astype(int)]

    # HDI
    x = np.arange(len(predictors))
    ax.vlines(x, hdi_lower, hdi_upper, color=color, zorder=1)

    # Mean
    ax.scatter(
        x,
        means,
        marker="o",
        edgecolor="black",
        linewidth=0.5,
        color=color,
        clip_on=False,
        zorder=2,
    )

    # Zero line
    ax.axhline(0, color="black", linewidth=0.75, zorder=-3, alpha=0.75)

    # Labels, ticks, etc
    ax.set_xticks(x)
    ax.set_xticklabels(
        [predictor_labels[predictor] for predictor in predictors],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Weight (a.u.)")
    ax.set_title("GLM weights")

    return ax


def main():

    # Load choice data
    choices = pd.read_csv(args.data_file, index_col=0)

    # Drop catch trials
    choices = choices.loc[choices["condition"].str.startswith("exp")]

    # 1. Make Psychometrics Figure
    ev_bins = np.arange(-2.5, 2.6, 1.).round(2)
    make_psychometric_figure(choices, bins=ev_bins)
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(
                args.output_dir,
                f"choice_analyses_psychometrics{args.label}.{extension}",
            ),
            bbox_inches="tight",
        )

    # 2. Make Individual Change Figure
    columns = ["higher_m", "higher_p"]
    make_individual_change_figure(args=args, columns=columns)
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(
                args.output_dir,
                f"choice_analyses_individual_changes{args.label}.{extension}",
            ),
            bbox_inches="tight",
        )

    # 3. Make GLM weight figure
    # Load GLM results
    glm_summary = pd.read_csv(
        join(args.choice_analyses_dir, "glm_summary.csv"),
        index_col=0,
    )

    predictors = [
        "delta_ev_z",
        "duration_favours_fx",
        "last_stage_favours_fx",
        "by_attribute_fx",
        "duration_favours_fx:by_attribute_fx",
        "last_stage_favours_fx:by_attribute_fx",
    ]

    predictor_labels = {
        "delta_ev_z": "ΔEV",
        "duration_favours_fx": "Duration (+ = longer)",
        "last_stage_favours_fx": "Sequence (+ = last)",
        "by_attribute_fx": "Presentation (+ = by att.)",
        "duration_favours_fx:by_attribute_fx": "Duration x Presentation",
        "last_stage_favours_fx:by_attribute_fx": "Sequence x Presentation",
    }

    fig, ax = plt.subplots(figsize=my.utilities.cm2inch(4.5, 4.5))
    plot_regression_weights(glm_summary, predictors, predictor_labels, ax=ax)
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(
                args.output_dir,
                f"choice_analyses_glm_weights{args.label}.{extension}",
            ),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--choice-analyses-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()
    if args.label != "":
        args.label = "_" + args.label
    mkdir_if_needed(args.output_dir)

    main()
