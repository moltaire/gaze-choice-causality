# -*- coding: utf-8 -*-
import argparse
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import myfuncs as my
from src.utilities import mkdir_if_needed

matplotlib = my.utilities.set_mpl_defaults(matplotlib)


def main():

    # Make the model prediction figure :)
    fig, axs = plt.subplots(1, 2, figsize=my.utilities.cm2inch(7, 4), sharey=True)

    x = [0, 1]

    # Alternative-wise
    ax = axs[0]
    ax.plot(x, [0.7, 0.3], "--o", label="Within")
    ax.plot(x, [0.5, 0.5], "--o", label="Between")

    ax.set_ylim(0, 1)
    ax.set_ylabel("P(Choose High $p$)")
    ax.set_yticks([])
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["High $p$", "High $m$"])
    ax.set_xlabel("Longer / last shown\nalternative")
    ax.set_title("Alternative-wise\npresentation")

    # Attribute-wise
    ax = axs[1]
    ax.plot(x, [0.5, 0.5], "--o", label="Within")
    ax.plot(x, [0.7, 0.3], "--o", label="Between")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["$p$", "$m$"])
    ax.set_xlabel("Longer / last shown\nattribute")
    ax.set_title("Attribute-wise\npresentation")
    ax.yaxis.set_tick_params(labelbottom=True)

    ax.legend(title="Model", bbox_to_anchor=(1, 0.5), loc="center left")
    fig.tight_layout(w_pad=3)

    my.utilities.label_axes(fig, loc=(-0.16, 1.125), fontweight="bold")
    for extension in ["pdf", "png"]:
        plt.savefig(
            join(args.output_dir, f"ddm_predictions.{extension}"),
            bbox_inches="tight",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()
    mkdir_if_needed(args.output_dir)

    main()
