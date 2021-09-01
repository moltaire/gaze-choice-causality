#!usr/bin/python

# This module contains code to run Bayes Factor t-tests
# using the R "BayesFactor" library

# One sample t-test
# Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G. (2009). Bayesian t tests for accepting and rejecting the null hypothesis. Psychonomic Bulletin & Review, 16(2), 225–237. https://doi.org/10.3758/PBR.16.2.225

# Code is adapted from:
# https://statsthinking21.github.io/statsthinking21-python/10-BayesianStatistics.html

import pandas as pd
import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()

# import the BayesFactor package
BayesFactor = importr("BayesFactor")


def one_sample_ttestbf(y, mu=0):
    """Performs the JZS t-test described in Rouder et al. (2009)

    Args:
        y ([type]): [description]
        mu (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    # Import the data into the R workspace
    robjects.globalenv["y"] = y - mu

    # Run the test and compute the Bayes factor
    r("bf = ttestBF(x=y)")
    # Extract result
    bf = r("extractBF(bf)")

    return bf


def two_sample_ttestbf(y1, y2, paired=False):

    # Import the data into the R workspace
    robjects.globalenv["df"] = pd.DataFrame(dict(y1=y1, y2=y2))

    # Run the test and compute the Bayes factor
    if paired:
        r("bf = ttestBF(x=df$y1, y=df$y2, paired=TRUE)")
    else:
        r("bf = ttestBF(x=df$y1, y=df$y2)")
    # Extract result
    bf = r("extractBF(bf)")

    return bf