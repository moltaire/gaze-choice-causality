import arviz as az
import numpy as np
import pymc3 as pm
import theano.tensor as tt


def bms(L, hdi_prob=0.95, **sample_kwargs):
    """This function computes the exceedance probabilities (xp)
    and expected relative frequencies (r) from an array of log-evidences.

    Args:
        L (numpy.ndarray): Array of model log-evidences (higher is better fit).
            Array shape should be (K models; N subjects)

        **sample_kwargs: Additional arguments to the pymc.sample function.
            Currently `cores=1` seems to be necessary.

    Returns:
        dict: Dictionary with values xp and r.

    Reference:
        Stephan, K. E., Penny, W. D., Daunizeau, J., Moran, R. J., & Friston, K. J. (2009). Bayesian model selection for group studies. Neuroimage, 46(4), 1004-1017.
    """

    K, N = L.shape

    with pm.Model() as bms:

        def lookup_L(L, N):
            """This function looks up the log-evidences for all N subjects,
            given the current model labels m.
            """
            return L[tt.cast(m, dtype="int32"), tt.cast(tt.arange(N), dtype="int32")]

        # Priors
        alpha = pm.Uniform("alpha", 0, N, shape=K, testval=np.ones(K))

        # Model
        r = pm.Dirichlet("r", a=alpha, testval=np.ones(K) / K)
        m = pm.Categorical("m", p=r, shape=N, testval=0)

        # Look up log evidence
        ll = pm.DensityDist("ll", logp=lookup_L, observed=dict(L=L, N=N))

        # Sample
        inferencedata = pm.sample(return_inferencedata=True, **sample_kwargs)

    # Build results
    result = {}
    result["summary"] = az.summary(
        inferencedata, hdi_prob=hdi_prob, var_names=["alpha", "r"]
    )
    result["xp"] = np.array(
        [
            np.mean(
                inferencedata.posterior["r"].data[:, :, k]
                == inferencedata.posterior["r"].data.max(axis=-1)
            )
            for k in range(K)
        ]
    )
    r_unscaled = np.array(
        [np.mean(inferencedata.posterior["r"].data[:, :, k]) for k in range(K)]
    )
    result["r"] = r_unscaled / r_unscaled.sum()

    return result
