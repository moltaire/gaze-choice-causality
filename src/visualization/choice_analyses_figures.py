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

    # Order manipulation
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
    axs[1].set_title("Order manipulation")

    # Panel labels
    fig.tight_layout(w_pad=4)
    my.utilities.label_axes(fig, fontweight="bold", fontsize=8, loc=(-0.3, 0.975))

    return fig, axs


def plot_individual_changes(
    y1,
    y2,
    axs=None,
    markersize_ind=4,
    alpha=0.02,
    bins=11,
    mean_kwargs={"markersize": 5},
    cmap=plt.cm.viridis,
    norm=plt.cm.colors.Normalize(vmin=0, vmax=1),
):
    if axs is None:
        fig, axs = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [2, 1]},
            figsize=my.utilities.cm2inch(3.25, 4.5),
            dpi=300,
        )

    # Bar handles
    ax = axs[0]
    x = [0, 1]
    for y1i, y2i in zip(y1, y2):
        if cmap is not None:
            if norm is None:
                norm = plt.cm.colors.NoNorm
            color = cmap(norm(y2i - y1i))
        else:
            color = "black"

        # Plot lines
        ax.plot(
            x,
            [y1i, y2i],
            "-",
            color=color,
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
            "o",
            color=color,
            alpha=alpha,
            markersize=markersize_ind,
            markeredgewidth=0,
            zorder=2,
            clip_on=False,
        )

    # Plot mean change
    if cmap is not None:
        if norm is None:
            norm = plt.cm.colors.NoNorm
        color = cmap(norm(np.mean(y2) - np.mean(y1)))
    else:
        color = mean_kwargs.get(color, "C0")
    mean_kwargs.pop("color", None)
    ax.plot(
        x,
        [np.mean(y1), np.mean(y2)],
        "-o",
        color="black",
        markerfacecolor=color,
        **mean_kwargs,
    )
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(x)

    # Histogram of individual changes
    ax = axs[1]
    my.plots.hist(y2 - y1, norm=norm, cm=cmap, bins=bins, ax=ax)

    return axs


def make_individual_change_figure(args):
    fig, axs = plt.subplots(
        2,
        4,
        figsize=my.utilities.cm2inch(12, 7),
        sharey="row",
        gridspec_kw={"height_ratios": [2, 1]},
        dpi=300,
    )

    i = 0
    for c, condition in enumerate(["duration", "sequence"]):
        for p, presentation in enumerate(["alternatives", "attributes"]):
            if condition == "duration":
                if presentation == "alternatives":
                    h = "H1a"
                else:
                    h = "H1b"
            else:
                if presentation == "alternatives":
                    h = "H2a"
                else:
                    h = "H2b"

            axs_i = axs[:, i]

            # Load data that was used for the corresponding statistical analysis:
            data = pd.read_csv(
                join(
                    args.choice_analyses_dir,
                    f"best_{condition}_{presentation}_data.csv",
                ),
                index_col=0,
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

            ## Determine statistical credibility of effect larger than 0
            ### Positive difference
            if (summary.loc["d", "hdi_2.5%"] > 0) and (
                summary.loc["d", "hdi_97.5%"] > 0
            ):
                color = plt.cm.coolwarm(1.0)
                lw = 1.0
            ### Negative difference
            elif (summary.loc["d", "hdi_2.5%"] < 0) and (
                summary.loc["d", "hdi_97.5%"] < 0
            ):
                color = plt.cm.coolwarm(0.0)
                lw = 0.75
            else:
                color = plt.cm.coolwarm(0.5)
                lw = 0.75

            # Plot
            axs_i = plot_individual_changes(
                y1=data["higher_m"],
                y2=data["higher_p"],
                axs=axs_i,
                bins=np.arange(-0.425, 0.426, 0.05),
                cmap=plt.cm.coolwarm,
                norm=plt.cm.colors.TwoSlopeNorm(vmin=-0.15, vcenter=0, vmax=0.15),
                alpha=0.1,
                mean_kwargs={"lw": lw, "markeredgewidth": lw},
            )
            axs_i[1].set_ylim(0, 50)

            ## Plot mean difference and 95% HDI
            # Vertical dashed line at 0
            axs_i[1].axvline(
                0, color="black", ls="--", lw=0.75, alpha=0.75, zorder=1, dashes=(3, 3)
            )

            axs_i[1].scatter(
                summary.loc["difference", "mean"],
                55,
                marker=".",
                s=8,
                color=color,
                edgecolor=None,
                linewidth=0,
                clip_on=False,
            )
            axs_i[1].hlines(
                55,
                summary.loc["difference", "hdi_2.5%"],
                summary.loc["difference", "hdi_97.5%"],
                lw=0.75,
                color=color,
                clip_on=False,
            )

            # Make stats annotation
            bf_10 = bft["bf"].values[0]
            if bf_10 >= 3:
                bf = bf_10
                bf_label = "BF$_{{+}0}$"
                bold_str_pre = r"$\bf{"
                bold_str_post = "}$"
            elif (1 / bf_10) >= 3:
                bf_label = "BF$_{0{+}}$"
                bf = 1 / bf_10
                bold_str_pre = r"$\bf{"
                bold_str_post = "}$"
            else:
                bf = bf_10
                bf_label = "BF$_{{+}0}$"
                bold_str_pre = ""
                bold_str_post = ""

            axs_i[0].text(
                0.5,
                1.2,
                f'$d$ = {summary.loc["d", "mean"]:.2f} [{summary.loc["d", "hdi_2.5%"]:.2f}, {summary.loc["d", "hdi_97.5%"]:.2f}]\n'
                + bf_label
                + " = "
                + bold_str_pre
                + f"{bf:.2f}"
                + bold_str_post,
                ha="center",
                va="center",
                transform=axs_i[0].transAxes,
                fontsize=4,
                bbox=dict(boxstyle="round,pad=.6", fc="none", ec=color, lw=lw),
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
            axs_i[0].set_xticks([0, 1])
            axs_i[0].set_xticklabels(xticklabels)
            axs_i[0].set_xlabel(xlabel, labelpad=1)
            axs_i[0].set_title(
                r"$\bf{" + h + "}$" + f"\n\n{presentation[:-1].capitalize()}-\nwise",
                y=1.3,
            )
            axs_i[1].set_xlabel("$\Delta$P(Choose $Hp$)")
            i += 1
    axs[1, 0].set_ylabel("N")
    axs[0, 0].set_ylabel("P(Choose $Hp$)")
    for ax in axs[0, :]:
        ax.set_ylim(0, 1)
    for ax in axs[1, :]:
        xticks = np.arange(-0.4, 0.41, 0.2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([np.round(val, 2) if val != 0 else "0" for val in xticks])
        ax.set_xlim(-0.3, 0.3)
    fig.align_ylabels(axs)
    fig.tight_layout(h_pad=2, w_pad=4)
    fig.text(
        np.mean(
            [
                fig.axes[i].get_position().get_points()[0][0]
                + 0.5 * fig.axes[i].get_position().width
                for i in [0, 1]
            ]
        ),
        0.95,
        "Duration",
        va="bottom",
        ha="center",
        fontweight="bold",
    )
    fig.text(
        np.mean(
            [
                fig.axes[i].get_position().get_points()[0][0]
                + 0.5 * fig.axes[i].get_position().width
                for i in [2, 3]
            ]
        ),
        0.95,
        "Order",
        va="bottom",
        ha="center",
        fontweight="bold",
    )
    my.utilities.label_axes(
        fig,
        loc=2 * ([(-0.4, 1.0)] + 3 * [(-0.15, 1.0)]),
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
        "Intercept",
        "delta_ev_z",
        "duration_favours_fx",
        "last_stage_favours_fx",
        "by_attribute_fx",
        "duration_favours_fx:by_attribute_fx",
        "last_stage_favours_fx:by_attribute_fx",
    ]

    predictor_labels = {
        "Intercept": "Intercept",
        "delta_ev_z": "ΔEV ($Hp - Hm$)",
        "duration_favours_fx": "Duration ($Hp$ or $p$ longer)",
        "last_stage_favours_fx": "Order ($Hp$ or $p$ last)",
        "by_attribute_fx": "Presentation (by alternative)",
        "duration_favours_fx:by_attribute_fx": "Duration x Presentation",
        "last_stage_favours_fx:by_attribute_fx": "Order x Presentation",
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
