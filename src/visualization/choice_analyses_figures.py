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


def plot_individual_changes(
    y1, y2, axs=None, markersize_ind=4, markersize_mean=5, alpha=0.02, bins=11
):
    if axs is None:
        fig, axs = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [1, 2]},
            figsize=my.utilities.cm2inch(3.25, 4.5),
            dpi=300,
        )

    # Bar handles
    ax = axs[1]
    x = [0, 1]
    for y1i, y2i in zip(y1, y2):
        # Plot lines
        ax.plot(
            x,
            [y1i, y2i],
            "-k",
            alpha=alpha,
            zorder=0,
        )
        # Plot some white dots without edge and with full alpha to hide lines overlapping with transparent dots
        ax.plot(
            x,
            [y1i, y2i],
            "ow",
            alpha=1,
            markersize=markersize_ind,
            zorder=1,
            linewidth=0,
            markeredgecolor="none",
            clip_on=False,
        )
        # Plot dots
        ax.plot(
            x,
            [y1i, y2i],
            "ok",
            alpha=alpha,
            markersize=markersize_ind,
            markeredgewidth=0,
            zorder=2,
            clip_on=False,
        )

    # Plot mean change
    ax.plot(x, [np.mean(y1), np.mean(y2)], "-o", markersize=markersize_mean)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(x)

    # Histogram of individual changes
    ax = axs[0]
    ax.hist(y2 - y1, color="black", edgecolor="white", linewidth=0.5, bins=bins)

    return axs


def make_individual_change_figure(args):
    fig, axs = plt.subplots(
        2,
        4,
        figsize=my.utilities.cm2inch(3 * 3.4, 6),
        sharey="row",
        gridspec_kw={"height_ratios": [1, 2.5]},
    )

    i = 0
    for c, condition in enumerate(["duration", "sequence"]):
        for p, presentation in enumerate(["alternatives", "attributes"]):

            axs_i = axs[:, i]

            # Load data that was used for the corresponding statistical analysis:
            data = pd.read_csv(
                join(
                    args.choice_analyses_dir,
                    f"best_{condition}_{presentation}_data.csv",
                ),
                index_col=0,
            )

            # Plot
            axs_i = plot_individual_changes(
                y1=data["higher_m"],
                y2=data["higher_p"],
                axs=axs_i,
                bins=np.arange(-0.4, 0.41, 0.05),
            )

            # Read BEST result
            summary = pd.read_csv(
                join(
                    args.choice_analyses_dir,
                    f"best_{condition}_{presentation}_summary.csv",
                ),
                index_col=0,
            )

            # Read BF t-test result
            bft = pd.read_csv(
                join(
                    args.choice_analyses_dir,
                    f"ttestbf-directed_{condition}_{presentation}.csv",
                ),
                index_col=0,
            )

            # Make stats annotation
            axs_i[1].text(
                0.5,
                1.3,
                f'$d$ = {summary.loc["d", "mean"]:.2f} [{summary.loc["d", "hdi_2.5%"]:.2f}, {summary.loc["d", "hdi_97.5%"]:.2f}]\n'
                + "BF$_{10}$ = "
                + f'{bft["bf"].values[0]:.2f}',
                ha="center",
                va="center",
                transform=axs_i[1].transAxes,
                fontsize=4,
                bbox=dict(boxstyle="round,pad=.4", fc="none"),
            )

            # Labels
            if condition == "duration":
                xlabel = "Shown longer"
            elif condition == "sequence":
                xlabel = "Shown last"
            if presentation == "alternatives":
                xticklabels = ["$Hm$", "$Hp$"]
            elif presentation == "attributes":
                xticklabels = ["$m$", "$p$"]
            axs_i[1].set_xticks([0, 1])
            axs_i[1].set_xticklabels(xticklabels)
            axs_i[1].set_xlabel(xlabel)
            axs_i[0].set_title(f"{presentation[:-1].capitalize()}-\nwise")
            axs_i[0].set_xlabel("$\Delta$P(Choose $Hp$)")
            i += 1
    axs[0, 0].set_ylabel("N")
    axs[1, 0].set_ylabel("P(Choose $Hp$)")
    for ax in axs[1, :]:
        ax.set_ylim(0, 1)
    for ax in axs[0, :]:
        xticks = np.arange(-0.4, 0.41, 0.2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([np.round(val, 2) if val != 0 else "0" for val in xticks])
        ax.set_xlim(-0.35, 0.35)
    fig.align_ylabels(axs)
    fig.tight_layout(h_pad=0, w_pad=4)
    fig.text(
        np.mean(
            [
                fig.axes[i].get_position().get_points()[0][0]
                + 0.5 * fig.axes[i].get_position().width
                for i in [0, 1]
            ]
        ),
        1,
        "Duration",
        va="bottom",
        ha="center",
    )
    fig.text(
        np.mean(
            [
                fig.axes[i].get_position().get_points()[0][0]
                + 0.5 * fig.axes[i].get_position().width
                for i in [2, 3]
            ]
        ),
        1,
        "Sequence",
        va="bottom",
        ha="center",
    )
    my.utilities.label_axes(
        fig,
        loc=2 * ([(-0.5, 1.05)] + 3 * [(-0.2, 1.05)]),
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="center",
    )

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
    ev_bins = np.arange(-2.5, 2.6, 1.0).round(2)
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
    make_individual_change_figure(args=args)
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
