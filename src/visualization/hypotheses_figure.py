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


def make_hypotheses_figure():
    fig, axs = plt.subplots(2, 2, figsize=my.utilities.cm2inch(4.5, 6), sharey=True)

    for p, (presentation, xticklabels, hletter) in enumerate(
        zip(
            ["Alternative-wise", "Attribute-wise"],
            [("$Hm$", "$Hp$"), ("$m$", "$p$")],
            ["a", "b"],
        )
    ):
        for i, (ivar, xlabel, hnumber) in enumerate(
            zip(["Duration", "Order"], ["Shown longer", "Shown last"], ["1", "2"])
        ):
            ax = axs[i, p]

            ax.plot([0, 1], [0.3, 0.7], "-o")

            ax.set_xticks([0, 1])
            ax.set_xlim(-0.5, 1.5)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel(xlabel)

            ax.set_yticks([])
            ax.set_ylim(0, 1)
            if p == 0:
                ax.set_ylabel("P(Choose $Hp$)")

            ax.text(
                -0.4, 1, "H" + hnumber + hletter, va="top", ha="left", fontweight="bold"
            )

    fig.tight_layout(h_pad=4, w_pad=3)
    fig.text(
        x=np.mean(
            [
                fig.axes[i].get_position().get_points()[0][0]
                + 0.5 * fig.axes[i].get_position().width
                for i in [0, 1]
            ]
        ),
        y=fig.axes[0].get_position().get_points()[1][1] + 0.05,
        s="Duration",
        fontweight="bold",
        ha="center",
    )
    fig.text(
        x=np.mean(
            [
                fig.axes[i].get_position().get_points()[0][0]
                + 0.5 * fig.axes[i].get_position().width
                for i in [0, 1]
            ]
        ),
        y=fig.axes[-1].get_position().get_points()[1][1] + 0.05,
        s="Order",
        fontweight="bold",
        ha="center",
    )


def main():

    make_hypotheses_figure()
    
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(
                args.output_dir,
                f"hypotheses.{extension}",
            ),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    mkdir_if_needed(args.output_dir)

    main()
