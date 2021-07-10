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


def plot_choice_problems(
    problems,
    marker="-o",
    color_by="implied_alpha",
    color=None,
    cb_title=r"$\alpha$",
    color_max_val=3,
    cm=None,
    norm=None,
    ax=None,
    add_colorbar=True,
    alpha=0.5,
):
    if ax is None:
        ax = plt.gca()
    if cm is None:
        cm = plt.cm.get_cmap("viridis")

    for i, problem in problems.iterrows():

        if color_by is None:
            color_problem = color
        else:
            color_problem = cm(problem[color_by] / color_max_val)

        ax.plot(
            np.array([problem["p0"], problem["p1"]]),
            np.array([problem["m0"], problem["m1"]]),
            marker,
            alpha=alpha,
            color=color_problem,
            zorder=2,
            clip_on=False,
        )

    if add_colorbar:
        cb = ax.scatter(
            problem["p0"],
            problem["p1"],
            c=problem[color_by],
            s=0,
            cmap=cm,
            norm=norm,
        )
        clb = plt.colorbar(cb)
        clb.ax.set_title(cb_title)

    ax.set_xlabel("$p$")
    ax.set_ylabel("$m$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def make_trial_figure(
    core,
    catch,
    alpha_core=0.5,
    alpha_catch=0.3,
    bins=np.arange(-0.25, 2.76, 0.5),
    implied_alpha_max=3,
):

    cm = plt.cm.get_cmap("cividis")
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=1, vmin=0, vmax=implied_alpha_max)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, axs = plt.subplots(1, 3, figsize=my.utilities.cm2inch(12, 4))

    # Core problems
    axs[0].set_title("Core choice problems")
    axs[0] = plot_choice_problems(
        core,
        marker="-o",
        color_by="implied_alpha",
        color_max_val=3,
        cm=cm,
        norm=norm,
        ax=axs[0],
        add_colorbar=False,
        alpha=alpha_core,
    )

    # Histogram of implied alphas
    axs[1] = my.plots.hist(
        core["implied_alpha"], bins=bins, cm=cm, norm=norm, ax=axs[1]
    )
    axs[1].set_title("Implied " + r"$\alpha$" + "\n if indifferent")
    axs[1].set_xlabel(r"$\alpha$")
    axs[1].yaxis.get_major_locator().set_params(integer=True)  # force integer y-ticks
    axs[1].set_ylabel("Frequency")
    axs[1].set_ylim(0, 2)

    # Catch problems
    axs[2].set_title("Catch trials")
    axs[2] = plot_choice_problems(
        catch,
        marker="-^",
        color_by=None,
        color="black",
        ax=axs[2],
        add_colorbar=False,
        alpha=alpha_catch,
    )

    plt.tight_layout()
    return fig, axs


def main():

    conditions = pd.read_csv(args.conditions_file, index_col=0)
    conditions[["m0", "m1"]] /= 10
    core = (
        conditions.loc[conditions["condition"].str.startswith("exp")]
        .groupby("core_id")
        .head(1)
    )
    catch = conditions.loc[conditions["condition"].str.startswith("catch")]
    fig, axs = make_trial_figure(
        core, catch, alpha_core=0.5, alpha_catch=0.3, bins=np.arange(-0.1, 3.11, 0.2)
    )
    axs[1].set_xticks(np.arange(0, 3.1, 1))
    axs[1].set_xticklabels(np.arange(0, 3.1, 1).astype(int))
    plt.tight_layout(w_pad=3)
    my.utilities.label_axes(
        fig, fontweight="bold", fontsize=8, loc=[(-0.45, 1), (-0.35, 1), (-0.45, 1)]
    )
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(
                args.output_dir,
                f"conditions.{extension}",
            ),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions-file", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    mkdir_if_needed(args.output_dir)

    main()
