# /usr/bin/python

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from myfuncs.plots.scatter import scatter
from myfuncs.stats.bayescorr import bayesian_correlation


def lm(
    x,
    y,
    hdi_prob=0.95,
    stats_annotation=True,
    run_correlation=True,
    ax=None,
    bandalpha=0.6,
    scatter_kws={},
    scatter_color=None,
    line_color=None,
    xrange=None,
    sample_kwargs={},
    family="normal",
    **kwargs,
):
    """Make a custom linear model plot with confidence bands.

    Args:
        x (array like): x values
        y (array like): y values
        ax (matplotlib.axis, optional): Axis to plot on. Defaults to current axis.
        bandalpha (float, optional): Opacity level of confidence band.
        scatter_kws (dict, optional): Dictionary of keyword arguments passed onto `scatter`.
        **kwargs: Keyword arguments passed onto plot of regression line.

    Returns:
        tuple
            matplotlib.axis: Axis with the linear model plot.
            arviz.InferenceData: GLM
            arviz.InferenceData: Correlation, optional, if `run_correlation`
    """
    if ax is None:
        ax = plt.gca()

    # Determine color (this is necessary so that the scatter and the line have the same color)
    if scatter_color is None and line_color is None:
        next_color = next(ax._get_lines.prop_cycler)["color"]
        scatter_color = next_color
        line_color = next_color
    elif scatter_color is None:
        scatter_color = line_color
    elif line_color is None:
        line_color = scatter_color

    # Scatter
    ax = scatter(x, y, color=scatter_color, ax=ax, **scatter_kws)

    # Run GLM in PyMC3
    df = pd.DataFrame(dict(x=x, y=y))
    with pm.Model() as glm:
        pm.GLM.from_formula("y ~ x", data=df, family=family)
        idata_glm = pm.sample(return_inferencedata=True, **sample_kwargs)

    summary_glm = az.summary(idata_glm, hdi_prob=hdi_prob)

    # Plot MAP regression line
    if xrange is None:
        xs = np.linspace(np.min(x), np.max(x), 100)
    else:
        xs = np.linspace(*xrange, 100)
    intercept = summary_glm.loc["Intercept", "mean"]
    beta = summary_glm.loc["x", "mean"]
    ax.plot(xs, intercept + beta * xs, color=line_color, zorder=0, **kwargs)

    # Plot posterior predictive credible region band
    intercept_samples = idata_glm.posterior["Intercept"].data.ravel()
    beta_samples = idata_glm.posterior["x"].data.ravel()
    ypred = intercept_samples + beta_samples * xs[:, None]
    ypred_lower = np.quantile(ypred, (1 - hdi_prob) / 2, axis=1)
    ypred_upper = np.quantile(ypred, 1 - (1 - hdi_prob) / 2, axis=1)
    ax.fill_between(
        xs,
        ypred_lower,
        ypred_upper,
        color=line_color,
        zorder=1,
        alpha=bandalpha,
        linewidth=0,
    )

    # Bayesian correlation analysis
    if run_correlation:
        idata_corr = bayesian_correlation(x, y, sample_kwargs=sample_kwargs)
        summary_corr = az.summary(idata_corr, hdi_prob=hdi_prob)

    # Add stats annotation
    if stats_annotation:
        # Determine HDI column names in az.summary, based on hdi_prob
        if ((100 * hdi_prob) % 2) == 0:
            digits = 0
        else:
            digits = 1
        hdi_lower = f"hdi_{100 * (1 - hdi_prob) / 2:.{digits}f}%"
        hdi_upper = f"hdi_{100 * (1 - (1 - hdi_prob) / 2):.{digits}f}%"

        stat_str = (
            f"Intercept = {summary_glm.loc['Intercept', 'mean']:.2f} [{summary_glm.loc['Intercept', hdi_lower]:.2f}, {summary_glm.loc['Intercept', hdi_upper]:.2f}]"
            + "\n"
            + f"Slope = {summary_glm.loc['x', 'mean']:.2f} [{summary_glm.loc['x', hdi_lower]:.2f}, {summary_glm.loc['x', hdi_upper]:.2f}]"
        )

        if run_correlation:
            stat_str_corr = (
                f"r = {summary_corr.loc['r', 'mean']:.2f} [{summary_corr.loc['r', hdi_lower]:.2f}, {summary_corr.loc['r', hdi_upper]:.2f}]"
                + "\n"
            )
            stat_str = stat_str_corr + stat_str

        ax.annotate(
            stat_str,
            [1, 0.05],
            xycoords="axes fraction",
            ma="right",
            ha="right",
            va="bottom",
            fontsize=4,
        )

    if run_correlation:
        return ax, idata_glm, idata_corr
    else:
        return ax, idata_glm