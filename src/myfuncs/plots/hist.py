# /usr/bin/python

import matplotlib.pyplot as plt


def hist(x, ax=None, cm=None, norm=None, **kwargs):
    """Make a custom histogram.
    Supports color-mapping (see https://stackoverflow.com/a/23062183)

    Args:
        x (array like): x values
        ax (matplotlib.axis, optional): Axis to plot on. Defaults to current axis.
        **kwargs: Keyword arguments passed onto plt.hist function.

    Returns:
        matplotlib.axis: Axis with the histogram.
    """
    if ax is None:
        ax = plt.gca()

    if norm is None:
        norm = plt.cm.colors.NoNorm()

    n, bins, patches = ax.hist(x, linewidth=0.75, edgecolor="white", **kwargs)

    # Color bars if a cmap is given
    if cm is not None:
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for value, p in zip(bin_centers, patches):
        plt.setp(p, "facecolor", cm(norm(value)))

    return ax
