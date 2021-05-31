#!usr/bin/python
import json

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from src.models.utils import choose, egreedy, softmax


class ChoiceModel(object):
    """Base class for probabilistic choice models

    Contains methods shared across models to
        1) simulate choices (`simulate_choices`)
        2) compute negative log-likelihood (`compute_nll`)
        3) perform parameter estimation (`fit`)
    """

    def __init__(self):
        super(ChoiceModel, self).__init__()

    def simulate_choices(self, parameters):
        """For given parameters, predict choice probabilities and generate choices from them."""
        choices = choose(self.predict_choiceprobs(parameters))
        return choices

    def recover(self, parameters_gen=None, **fit_kwargs):
        if parameters_gen is None:
            print(
                "No `parameters_gen` for simulation given. Trying to estimate some from attached data..."
            )
            parameters_gen, nll = self.fit(**fit_kwargs)

        # Simulate synthetic data
        self.choices = self.simulate_choices(parameters=parameters_gen)

        # Re-estimate parameters
        parameters_rec, nll = self.fit(**fit_kwargs)

        # Format result
        recovery_df = pd.DataFrame(
            {
                parameter_name + "_gen": parameter_gen
                for parameter_name, parameter_gen in zip(
                    self.parameter_names, parameters_gen
                )
            },
            index=[0],
        )
        for parameter_name, parameter_rec in zip(self.parameter_names, parameters_rec):
            recovery_df[parameter_name + "_rec"] = parameter_rec
        recovery_df["nll"] = nll

        return recovery_df

    def compute_nll(self, parameters, verbose=False, nonzeroconst=1e-6):
        """Compute negative log-likelihood of the data, given parameters."""
        choiceprobs = self.predict_choiceprobs(parameters)
        chosenprobs = choiceprobs[
            np.arange(choiceprobs.shape[0]).astype(int), self.choices.astype(int)
        ]
        nll = -np.sum(np.log(chosenprobs + nonzeroconst))
        if verbose > 1:
            print(
                "\t",
                "Subject",
                self.subject,
                "\t",
                *np.round(parameters, 2),
                "\tNLL",
                np.round(nll, 2),
                end="\r"
            )
        return nll

    def fit(
        self, method="minimize", n_runs=1, seed=None, verbose=False, **method_kwargs
    ):
        """Estimate best fitting parameters using maximum log-likelihood.

        Parameters:
        -----------
        method : str, optional
            Optimization method to use. Must be one of ['minimize', 'differential_evolution'], defaults to 'minimize'.
        n_runs : int, optional
            Number of optimization runs. Should probably be more than 1 if method='minimize'. Defaults to 1.
        seed : int, optional
            Random seed. Defaults to no seed.
        verbose : int, optional
            Verbosity toggle. Prints some stuff if > 0. Prints more stuff if > 1... Defaults to 0.
        **method_kwargs : optional
            Additional keyword arguments to be passed on to the optimizer.

        Returns:
        -------
        tuple
            (maximum-likelihood estimates, minimum negative log-likelihood)
        """
        best_nll = np.inf
        best_x = np.zeros(self.n_parameters) * np.nan
        for run in range(n_runs):
            if verbose > 0:
                print(
                    "{}\tSubject {}\tRun {} of {} ({:.0f}%)".format(
                        self.label,
                        self.subject,
                        run + 1,
                        n_runs,
                        100 * (run + 1) / n_runs,
                    ),
                    # end="\r",
                )
            if seed is not None:
                if isinstance(self.subject, str):
                    subject_for_seed = int(self.subject.split("-")[0])
                else:
                    subject_for_seed = self.subject
                np.random.seed(seed * subject_for_seed + seed * run)

            if method == "minimize":
                x0 = [
                    np.random.uniform(*self.parameter_bounds[p])
                    for p in range(self.n_parameters)
                ]
                result = minimize(
                    self.compute_nll,
                    x0=x0,
                    bounds=self.parameter_bounds,
                    **method_kwargs
                )
            elif method == "differential_evolution":
                result = differential_evolution(
                    self.compute_nll, bounds=self.parameter_bounds, **method_kwargs
                )
            else:
                raise ValueError(
                    'Unknown method "{}". Use "minimize" or "differential_evolution".'.format(
                        method
                    )
                )
            if result.fun < best_nll:
                best_nll = result.fun
                best_x = result.x
        return best_x, best_nll


class ExpectedUtility(ChoiceModel):
    """Expected Utility model

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        label="EU",
        parameter_names=["alpha", "beta"],
        parameter_bounds=[(0, 5), (0, 50)],
    ):
        super(ExpectedUtility, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, beta = parameters
        utilities = self.probabilities * self.outcomes ** alpha
        choiceprobs = softmax(beta * utilities)
        return choiceprobs


class WeightedAdditiveDN(ChoiceModel):
    """Weighted additive attribute model with divisive normalization

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        label="WA_dn",
        parameter_names=["wp", "beta"],
        parameter_bounds=[(0, 1), (0, 50)],
    ):
        super(WeightedAdditiveDN, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta = parameters
        p_n = self.probabilities / self.probabilities.sum(axis=1, keepdims=True)
        m_n = self.outcomes / self.outcomes.sum(axis=1, keepdims=True)
        utilities = wp * p_n + (1 - wp) * m_n
        choiceprobs = softmax(beta * utilities)
        return choiceprobs


class WeightedAdditiveDNBLast(ChoiceModel):
    """Weighted additive attribute model with divisive normalization

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        label="WA_dn_b-Last",
        parameter_names=["wp", "beta", "b_last"],
        parameter_bounds=[(0, 1), (0, 200), (-1, 1)],
    ):
        super(WeightedAdditiveDNBLast, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta, b_last = parameters

        # Biases
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        p_n = self.probabilities / self.probabilities.sum(axis=1, keepdims=True)
        m_n = self.outcomes / self.outcomes.sum(axis=1, keepdims=True)
        utilities = wp * p_n + (1 - wp) * m_n
        choiceprobs = softmax(beta * (utilities + last_bias))
        return choiceprobs


class WeightedAdditiveDNBLastBLonger(ChoiceModel):
    """Weighted additive attribute model with divisive normalization

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="WA_dn_b-Last_b-Longer",
        parameter_names=["wp", "beta", "b_last", "b_longer"],
        parameter_bounds=[(0, 1), (0, 200), (-1, 1), (-1, 1)],
    ):
        super(WeightedAdditiveDNBLastBLonger, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta, b_last, b_longer = parameters

        # Biases
        longer_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        longer_bias[np.arange(len(longer_bias)), self.longer] = b_longer
        longer_bias = longer_bias[:, :2]

        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        p_n = self.probabilities / self.probabilities.sum(axis=1, keepdims=True)
        m_n = self.outcomes / self.outcomes.sum(axis=1, keepdims=True)
        utilities = wp * p_n + (1 - wp) * m_n
        choiceprobs = softmax(beta * (utilities + last_bias + longer_bias))
        return choiceprobs


class WeightedAdditive(ChoiceModel):
    """Weighted additive attribute model with divisive normalization

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        label="WA",
        parameter_names=["wp", "beta"],
        parameter_bounds=[(0, 1), (0, 200)],
    ):
        super(WeightedAdditive, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta = parameters
        utilities = wp * self.probabilities + (1 - wp) * self.outcomes
        choiceprobs = softmax(beta * utilities)
        return choiceprobs


class WeightedAdditiveBLastBLonger(ChoiceModel):
    """Weighted additive attribute model with divisive normalization

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="WA_b-Last_b-Longer",
        parameter_names=["wp", "beta", "b_last", "b_longer"],
        parameter_bounds=[(0, 1), (0, 200), (-1, 1), (-1, 1)],
    ):
        super(WeightedAdditiveBLastBLonger, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta, b_last, b_longer = parameters

        # Biases
        longer_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        longer_bias[np.arange(len(longer_bias)), self.longer] = b_longer
        longer_bias = longer_bias[:, :2]

        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        utilities = wp * self.probabilities + (1 - wp) * self.outcomes
        choiceprobs = softmax(beta * (utilities + last_bias + longer_bias))
        return choiceprobs


class WeightedAdditiveBLast(ChoiceModel):
    """Weighted additive attribute model with divisive normalization

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        label="WA_b-Last",
        parameter_names=["wp", "beta", "b_last"],
        parameter_bounds=[(0, 1), (0, 200), (-1, 1)],
    ):
        super(WeightedAdditiveBLast, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta, b_last = parameters

        # Biases
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        utilities = wp * self.probabilities + (1 - wp) * self.outcomes
        choiceprobs = softmax(beta * (utilities + last_bias))
        return choiceprobs


class ExpectedUtilityDurWeighted(ChoiceModel):
    """Presentation duration weighted Expected Utility model

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        dur_cols=["g0", "g1"],
        label="EU_dur-weighted",
        parameter_names=["alpha", "beta", "theta"],
        parameter_bounds=[(0, 5), (0, 200), (0, 1)],
    ):
        super(ExpectedUtilityDurWeighted, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.durations = data[dur_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, beta, theta = parameters
        utilities = self.probabilities * self.outcomes ** alpha
        biased_u = self.durations * utilities + (1 - self.durations) * theta * utilities
        choiceprobs = softmax(beta * biased_u)
        return choiceprobs


class OutcomeCutoff(ChoiceModel):
    """A heuristic model where the higher probability option is chosen unless one outcome reaches a threshold.
    Uses epsilon greedy choice rule.

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        label="mCutoff",
        parameter_names=["m_min", "epsilon"],
        parameter_bounds=[(0, 10), (0, 0.5)],
    ):
        super(OutcomeCutoff, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        m_min, epsilon = parameters
        m_threshold_crossed = np.any(self.outcomes > m_min, axis=1)
        choiceprobs = np.where(
            m_threshold_crossed[:, None],
            egreedy(self.outcomes, epsilon),
            egreedy(self.probabilities, epsilon),
        )
        return choiceprobs


class ExpectedUtilityBLastBLonger(ChoiceModel):
    """Expected Utility model

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="EU_b-Last_b-Longer",
        parameter_names=["alpha", "beta", "b_longer", "b_last"],
        parameter_bounds=[(0, 5), (0, 200), (-1, 1), (-1, 1)],
    ):
        super(ExpectedUtilityBLastBLonger, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, beta, b_longer, b_last = parameters
        utilities = self.probabilities * self.outcomes ** alpha

        # Biases
        longer_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        longer_bias[np.arange(len(longer_bias)), self.longer] = b_longer
        longer_bias = longer_bias[:, :2]

        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (utilities + longer_bias + last_bias))
        return choiceprobs


class ExpectedUtilityBLast(ChoiceModel):
    """Expected Utility model

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        label="EU_b-Last",
        parameter_names=["alpha", "beta", "b_last"],
        parameter_bounds=[(0, 5), (0, 200), (-1, 1)],
    ):
        super(ExpectedUtilityBLast, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, beta, b_last = parameters
        utilities = self.probabilities * self.outcomes ** alpha

        # Bias
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (utilities + last_bias))
        return choiceprobs


class OutcomeCutoffBLastBLonger(ChoiceModel):
    """A heuristic model where the higher probability option is chosen unless one outcome reaches a threshold.
    Uses epsilon greedy choice rule.

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="mCutoff_b-Last_b-Longer",
        parameter_names=["m_min", "epsilon", "b_longer", "b_last"],
        parameter_bounds=[(0, 10), (0, 0.5), (-1, 1), (-1, 1)],
    ):
        super(OutcomeCutoffBLastBLonger, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        m_min, epsilon, b_longer, b_last = parameters
        m_threshold_crossed = np.any(self.outcomes > m_min, axis=1)

        # Biases
        longer_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        longer_bias[np.arange(len(longer_bias)), self.longer] = b_longer
        longer_bias = longer_bias[:, :2]

        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = np.where(
            m_threshold_crossed[:, None],
            egreedy(self.outcomes + longer_bias + last_bias, epsilon),
            egreedy(self.probabilities + longer_bias + last_bias, epsilon),
        )
        return choiceprobs


class OutcomeCutoffBLast(ChoiceModel):
    """A heuristic model where the higher probability option is chosen unless one outcome reaches a threshold.
    Uses epsilon greedy choice rule.

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        label="mCutoff_b-Last",
        parameter_names=["m_min", "epsilon", "b_last"],
        parameter_bounds=[(0, 10), (0, 0.5), (-1, 1)],
    ):
        super(OutcomeCutoffBLast, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        m_min, epsilon, b_last = parameters
        m_threshold_crossed = np.any(self.outcomes > m_min, axis=1)

        # Biases
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = np.where(
            m_threshold_crossed[:, None],
            egreedy(self.outcomes + last_bias, epsilon),
            egreedy(self.probabilities + last_bias, epsilon),
        )
        return choiceprobs


class ProspectTheory(ChoiceModel):
    """Prospect Theory model.
    Assumes that objective probabilities are transformed into decision weights (using weighting function with parameter $\gamma$), and outcome utilities are computed with a power-function with parameter $\alpha$. Choice probabilities are derived from subjective expected utilities via a softmax function with inverse temperature parameter $\beta$.

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        label="PT",
        parameter_names=["alpha", "gamma", "beta"],
        parameter_bounds=[(0, 5), (0.28, 1), (0, 200)],
    ):
        super(ProspectTheory, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, gamma, beta = parameters
        p = self.probabilities
        w = p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        SU = w * self.outcomes ** alpha
        choiceprobs = softmax(beta * SU)
        return choiceprobs


class ProspectTheoryBLast(ChoiceModel):
    """Prospect Theory model.
    Assumes that objective probabilities are transformed into decision weights (using weighting function with parameter $\gamma$), and outcome utilities are computed with a power-function with parameter $\alpha$. Choice probabilities are derived from subjective expected utilities via a softmax function with inverse temperature parameter $\beta$.

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="PT_b-Last",
        parameter_names=["alpha", "gamma", "beta", "b_last"],
        parameter_bounds=[(0, 5), (0.28, 1), (0, 200), (-1, 1)],
    ):
        super(ProspectTheoryBLast, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, gamma, beta, b_last = parameters
        p = self.probabilities
        w = p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        SU = w * self.outcomes ** alpha

        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (SU + last_bias))

        return choiceprobs


class ProspectTheoryBLastBLonger(ChoiceModel):
    """Prospect Theory model.
    Assumes that objective probabilities are transformed into decision weights (using weighting function with parameter $\gamma$), and outcome utilities are computed with a power-function with parameter $\alpha$. Choice probabilities are derived from subjective expected utilities via a softmax function with inverse temperature parameter $\beta$.

    Attributes:
        choices (np.ndarray): Array of choices of type int
        label (str, optional): Model label
        outcomes (np.ndarray): Array (n_trials x n_alternatives) of option outcomes
        probabilities (np.ndarray): Array (n_trials x n_alternatives) of outcome probabilities
        parameter_bounds (list): List of tuples of parameter boundaries [(alpha_low, alpha_up), (beta_low, beta_up)]
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="PT_b-Last_b-Longer",
        parameter_names=["alpha", "gamma", "beta", "b_last", "b_longer"],
        parameter_bounds=[(0, 5), (0.28, 1), (0, 200), (-1, 1), (-1, 1)],
    ):
        super(ProspectTheoryBLastBLonger, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, gamma, beta, b_last, b_longer = parameters
        p = self.probabilities
        w = p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        SU = w * self.outcomes ** alpha

        # Biases
        longer_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        longer_bias[np.arange(len(longer_bias)), self.longer] = b_longer
        longer_bias = longer_bias[:, :2]

        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (SU + longer_bias + last_bias))

        return choiceprobs


class Glickman1Layer(ChoiceModel):
    """Three alternative adaptation from the winning model from Glickman et al., 2019
    Assumes that in each fixation, gaze-biased subjective utilities (see PT) are accumulated and all accumulators (irrespective of fixation) are subject to leak over individual fixations.

    Parameters
    ----------
    alpha (alpha > 0)
        Utility function parameter
    gamma (0.28 < gamma < 1)
        Probability weighting parameter
    beta (beta > 0)
        Inverse temperature parameter
    lambda (0 < lambda < 1)
        Leak parameter (0 = perfect memory, 1 = full leak)
    theta (0 < theta < 1)
        Gaze bias parameter
    """

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        label="glickman1layer",
        parameter_names=["alpha", "gamma", "beta", "lam", "theta"],
        parameter_bounds=[(0, 5), (0.2, 1), (0, 200), (0, 1), (0, 1)],
    ):
        super(Glickman1Layer, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.fixated_alternatives = data["fixated_alternatives"].values
        self.fixated_attributes = data["fixated_attributes"].values  # 0 = p, 1 = m
        self.fixation_durations = data["fixation_durations"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)
        self.n_items = len(probability_cols)

    def predict_choiceprobs(self, parameters):

        alpha, gamma, beta, lam, theta = parameters
        p = self.probabilities
        w = p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        SU = w * self.outcomes ** alpha
        Y = np.zeros((self.n_trials, self.n_items))

        for trial in range(self.n_trials):

            # If fixation data present
            if isinstance(self.fixation_durations[trial], np.ndarray):
                for dur, alt, att in zip(
                    self.fixation_durations[trial],
                    self.fixated_alternatives[trial],
                    self.fixated_attributes[trial],
                ):

                    # Option wise gaze discount
                    theta_vector = np.ones(self.n_items) * theta
                    theta_vector[alt] = 1.0

                    Y[trial, :] = (1 - lam) * Y[trial, :] + theta_vector * SU[trial, :]

        choiceprobs = softmax(beta * Y)
        return choiceprobs


class Glickman2Layer(ChoiceModel):
    """Three alternative adaption from 2-layer model from Glickman et al., 2019
    Also assumes that over fixations, subjective utilities (see PT) are accumulated. However, in contrast to the 1-layer model, here, the subjective stimulus attributes (decision weights and subjective utilities) also accumulate across fixations. The gaze-bias acts on the input to these lower-level accumulators (decision weights and subjective utilities), which are then combined *after the gaze bias was applied* in the next level.
    Accumulators on both levels are subject to leak.

    For a reference, see Glickman et al., 2019 (Fig. 6A)

    Parameters
    ----------
    alpha (alpha > 0)
        Utility function parameter
    gamma (0.28 < gamma < 1)
        Probability weighting parameter
    beta (beta > 0)
        Inverse temperature parameter
    lambda (0 < lambda < 1)
        Leak parameter (0 = perfect memory, 1 = full leak)
    theta (0 < theta < 1)
        Gaze bias parameter
    """

    def __init__(
        self,
        data,
        probability_cols=["pA", "pB", "pC"],
        outcome_cols=["mA", "mB", "mC"],
        label="Glickman2Layer",
        parameter_names=["alpha", "gamma", "beta", "lam", "theta"],
        parameter_bounds=[(0, 5), (0.2, 1), (0, 50), (0, 1), (0, 1)],
    ):
        super(Glickman2Layer, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.choices = data["choice"].values
        self.fixated_alternatives = data["fixated_alternatives"].values  # 0 = p, 1 = m
        self.fixated_attributes = data["fixated_attributes"].values
        self.fixation_durations = data["fixation_durations"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)
        self.n_items = len(probability_cols)
        self.n_attributes = 2

    def predict_choiceprobs(self, parameters):

        alpha, gamma, beta, lam, theta = parameters
        p = self.probabilities
        w = p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
        m = self.outcomes ** alpha
        L1w = np.zeros((self.n_trials, self.n_items))
        L1m = np.zeros((self.n_trials, self.n_items))
        L2 = np.zeros((self.n_trials, self.n_items))

        for trial in range(self.n_trials):

            # If fixation data present
            if isinstance(self.fixation_durations[trial], np.ndarray):
                for dur, alt, att in zip(
                    self.fixation_durations[trial],
                    self.fixated_alternatives[trial],
                    self.fixated_attributes[trial],
                ):

                    # AOI wise gaze discount
                    theta_vector = np.ones((self.n_items, self.n_attributes)) * theta
                    theta_vector[alt, att] = 1.0

                    L1w[trial, :] = (1 - lam) * L1w[trial, :] + theta_vector[:, 0] * w[
                        trial, :
                    ]
                    L1m[trial, :] = (1 - lam) * L1m[trial, :] + theta_vector[:, 1] * m[
                        trial, :
                    ]
                    L2[trial, :] = (1 - lam) * L2[trial, :] + L1w[trial, :] * L1m[
                        trial, :
                    ]

        choiceprobs = softmax(beta * L2)
        return choiceprobs


class DiffOfDiffBLastBLonger(ChoiceModel):
    """"""

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="DiffOfDiff_b-Last_b-Longer",
        parameter_names=["wp", "beta", "b_last", "b_longer"],
        parameter_bounds=[(0, 1), (0, 200), (-1, 1), (-1, 1)],
    ):
        super(DiffOfDiffBLastBLonger, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta, b_last, b_longer = parameters

        # Attribute Differences
        d_p = np.vstack(
            [
                self.probabilities[:, 0] - self.probabilities[:, 1],
                self.probabilities[:, 1] - self.probabilities[:, 0],
            ]
        ).T

        d_m = np.vstack(
            [
                self.outcomes[:, 0] - self.outcomes[:, 1],
                self.outcomes[:, 1] - self.outcomes[:, 0],
            ]
        ).T

        # Difference of differences
        D = wp * d_p + (1 - wp) * d_m

        # Biases
        longer_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        longer_bias[np.arange(len(longer_bias)), self.longer] = b_longer
        longer_bias = longer_bias[:, :2]

        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (D + last_bias + longer_bias))
        return choiceprobs


class DiffOfDiffBLast(ChoiceModel):
    """"""

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="DiffOfDiff_b-Last",
        parameter_names=["wp", "beta", "b_last"],
        parameter_bounds=[(0, 1), (0, 200), (-1, 1)],
    ):
        super(DiffOfDiffBLast, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta, b_last = parameters

        # Attribute Differences
        d_p = np.vstack(
            [
                self.probabilities[:, 0] - self.probabilities[:, 1],
                self.probabilities[:, 1] - self.probabilities[:, 0],
            ]
        ).T

        d_m = np.vstack(
            [
                self.outcomes[:, 0] - self.outcomes[:, 1],
                self.outcomes[:, 1] - self.outcomes[:, 0],
            ]
        ).T

        # Difference of differences
        D = wp * d_p + (1 - wp) * d_m

        # Biases
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (D + last_bias))
        return choiceprobs


class DiffOfDiff(ChoiceModel):
    """"""

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="DiffOfDiff",
        parameter_names=["wp", "beta"],
        parameter_bounds=[(0, 1), (0, 200)],
    ):
        super(DiffOfDiff, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.longer = data[duration_favours_col].fillna(2).astype(int).values.ravel()
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta = parameters

        # Attribute Differences
        d_p = np.vstack(
            [
                self.probabilities[:, 0] - self.probabilities[:, 1],
                self.probabilities[:, 1] - self.probabilities[:, 0],
            ]
        ).T

        d_m = np.vstack(
            [
                self.outcomes[:, 0] - self.outcomes[:, 1],
                self.outcomes[:, 1] - self.outcomes[:, 0],
            ]
        ).T

        # Difference of differences
        D = wp * d_p + (1 - wp) * d_m

        choiceprobs = softmax(beta * D)
        return choiceprobs


class TwoStageWithin(ChoiceModel):
    """"""

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        alt_gaze_cols=["g0", "g1"],
        att_gaze_cols=["gp", "gm"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="TwoStageWithin",
        parameter_names=["alpha", "theta", "beta", "b_last"],
        parameter_bounds=[(0, 3), (0, 1), (0, 100), (-0.1, 0.1)],
    ):
        super(TwoStageWithin, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values / 10
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.alt_gaze = data[alt_gaze_cols].values
        self.att_gaze = data[att_gaze_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, theta, beta, b_last = parameters

        eu = self.probabilities * self.outcomes ** alpha

        # # Set theta to 1 in attribute-wise presentation trials
        # trial_thetas = (
        #     (1 - self.data["presentation01"]) * theta + (self.data["presentation01"])
        # ).values[:, None]
        trial_thetas = theta

        X = self.alt_gaze * eu + (1 - self.alt_gaze) * trial_thetas * eu

        # Biases
        # Set last-favours to 2 (favouring neither alt 1 or alt 2) in attribute-wise presentation
        last_favours = np.where(self.data["by_attribute"], 2, self.last_favours)
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (X + last_bias))
        return choiceprobs


class TwoStageBetween(ChoiceModel):
    """"""

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        alt_gaze_cols=["g0", "g1"],
        att_gaze_cols=["gp", "gm"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="TwoStageBetween",
        parameter_names=["wp", "eta", "beta", "b_last"],
        parameter_bounds=[(0, 1), (0, 1), (0, 100), (-0.1, 0.1)],
    ):
        super(TwoStageBetween, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.alt_gaze = data[alt_gaze_cols].values
        self.att_gaze = data[att_gaze_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, eta, beta, b_last = parameters

        pn = self.probabilities / self.probabilities.sum(axis=1, keepdims=True)
        mn = self.outcomes / self.outcomes.sum(axis=1, keepdims=True)

        # Set theta to 1 in attribute-wise presentation trials
        # trial_etas = (
        #     (1 - self.data["presentation01"]) + (self.data["presentation01"] * eta)
        # ).values[:, None]
        trial_etas = eta

        X = self.att_gaze[:, 0][:, None] * (
            wp * pn + trial_etas * (1 - wp) * mn
        ) + self.att_gaze[:, 1][:, None] * (trial_etas * wp * pn + (1 - wp) * mn)

        # Biases
        # Set last-favours to 2 (favouring neither alt 1 or alt 2) in alternative-wise presentation
        last_favours = np.where(~self.data["by_attribute"], 2, self.last_favours)
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (X + last_bias))
        return choiceprobs


class TwoStageMixture(ChoiceModel):
    """"""

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        alt_gaze_cols=["g0", "g1"],
        att_gaze_cols=["gp", "gm"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="TwoStageMixture",
        parameter_names=[
            "alpha",
            "theta",
            "wp",
            "eta",
            "s_between",
            "w_between_attwise",
            "w_between_altwise",
            "beta",
            "b_last",
        ],
        parameter_bounds=[
            (0, 3),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 40),
            (0, 1),
            (0, 1),
            (0, 200),
            (-1, 1),
        ],
    ):
        super(TwoStageMixture, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.alt_gaze = data[alt_gaze_cols].values
        self.att_gaze = data[att_gaze_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        (
            alpha,
            theta,
            wp,
            eta,
            s_between,
            w_between_attwise,
            w_between_altwise,
            beta,
            b_last,
        ) = parameters

        # Between alternatives component
        pn = self.probabilities / self.probabilities.sum(axis=1, keepdims=True)
        mn = self.outcomes / self.outcomes.sum(axis=1, keepdims=True)

        # Set theta to 1 in attribute-wise presentation trials
        # trial_etas = (
        #     (1 - self.data["presentation01"]) + (self.data["presentation01"] * eta)
        # ).values[:, None]
        trial_etas = eta

        X_between = self.att_gaze[:, 0][:, None] * (
            wp * pn + trial_etas * (1 - wp) * mn
        ) + self.att_gaze[:, 1][:, None] * (trial_etas * wp * pn + (1 - wp) * mn)

        # Within alternative component
        eu = self.probabilities * self.outcomes ** alpha

        # Set theta to 1 in attribute-wise presentation trials
        # trial_thetas = (
        #     (1 - self.data["presentation01"]) * theta + (self.data["presentation01"])
        # ).values[:, None]
        trial_thetas = theta

        X_within = self.alt_gaze * eu + (1 - self.alt_gaze) * trial_thetas * eu

        # Weighted combination
        w_between = (
            self.data["presentation01"] * w_between_attwise
            + (1 - self.data["presentation01"]) * w_between_altwise
        ).values[:, None]

        X = (1 - w_between) * X_within + w_between * s_between * X_between

        # Biases
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (X + last_bias))
        return choiceprobs


class TwoStageMixtureNoScaling(ChoiceModel):
    """"""

    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        alt_gaze_cols=["g0", "g1"],
        att_gaze_cols=["gp", "gm"],
        last_stage_favours_col=["last_stage_favours"],
        duration_favours_col=["duration_favours"],
        label="TwoStageMixtureNoScaling",
        parameter_names=[
            "alpha",
            "theta",
            "wp",
            "eta",
            "w_between_attwise",
            "w_between_altwise",
            "beta",
            "b_last",
        ],
        parameter_bounds=[
            (0, 3),  # alpha
            (0, 1),  # theta
            (0, 1),  # wp
            (0, 1),  # eta
            (0, 1),  # w_between_attwise
            (0, 1),  # w_between_altwise
            (0, 100),  # beta
            (-0.1, 0.1),  # b_last
        ],
    ):
        super(TwoStageMixtureNoScaling, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values / 10
        self.last_favours = (
            data[last_stage_favours_col].fillna(2).astype(int).values.ravel()
        )
        self.alt_gaze = data[alt_gaze_cols].values
        self.att_gaze = data[att_gaze_cols].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        (
            alpha,
            theta,
            wp,
            eta,
            w_between_attwise,
            w_between_altwise,
            beta,
            b_last,
        ) = parameters

        # Between alternatives component
        pn = self.probabilities / self.probabilities.sum(axis=1, keepdims=True)
        mn = self.outcomes / self.outcomes.sum(axis=1, keepdims=True)

        # Set theta to 1 in attribute-wise presentation trials
        # trial_etas = (
        #     (1 - self.data["presentation01"]) + (self.data["presentation01"] * eta)
        # ).values[:, None]
        trial_etas = eta

        X_between = self.att_gaze[:, 0][:, None] * (
            wp * pn + trial_etas * (1 - wp) * mn
        ) + self.att_gaze[:, 1][:, None] * (trial_etas * wp * pn + (1 - wp) * mn)

        # Within alternative component
        eu = self.probabilities * self.outcomes ** alpha

        # Set theta to 1 in attribute-wise presentation trials
        # trial_thetas = (
        #     (1 - self.data["presentation01"]) * theta + (self.data["presentation01"])
        # ).values[:, None]
        trial_thetas = theta

        X_within = self.alt_gaze * eu + (1 - self.alt_gaze) * trial_thetas * eu

        # Weighted combination
        w_between = (
            self.data["presentation01"] * w_between_attwise
            + (1 - self.data["presentation01"]) * w_between_altwise
        ).values[:, None]

        X = (1 - w_between) * X_within + w_between * X_between

        # Biases
        last_bias = np.zeros((self.outcomes.shape[0], self.outcomes.shape[1] + 1))
        last_bias[np.arange(len(last_bias)), self.last_favours] = b_last
        last_bias = last_bias[:, :2]

        choiceprobs = softmax(beta * (X + last_bias))
        return choiceprobs


class LeakyAltwiseDiscount(ChoiceModel):
    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        sequence_col="sequence",
        presentation_col="presentation",
        label="LeakyAltwiseDiscount",
        parameter_names=["alpha", "beta", "omega", "theta"],
        parameter_bounds=[(0, 5), (0, 100), (0, 1), (0, 1)],
    ):
        super(LeakyAltwiseDiscount, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values
        self.sequences = data[sequence_col].values
        self.presentations = data[presentation_col].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        alpha, beta, omega, theta = parameters
        s = self.probabilities * self.outcomes ** alpha

        V = np.zeros_like(s)

        for trial in range(self.n_trials):
            sequence_trial = json.loads(self.sequences[trial])
            fixation_durations = np.array(sequence_trial["durations"]) / 1000
            fixated_alternatives = np.array(sequence_trial["alternatives"])
            for i, (dur, target) in enumerate(
                zip(fixation_durations, fixated_alternatives)
            ):
                if target in [0, 1]:
                    theta_vec = np.ones(s.shape[1]) * theta
                    theta_vec[target] = 1
                else:  # target is "all", both alternatives visible
                    theta_vec = np.ones(s.shape[1])
                V[trial, :] = omega ** dur * V[trial, :] + theta_vec * s[trial, :] * dur

        choiceprobs = softmax(beta * V)
        return choiceprobs


class LeakyAttwiseDiscount(ChoiceModel):
    def __init__(
        self,
        data,
        probability_cols=["p0", "p1"],
        outcome_cols=["m0", "m1"],
        sequence_col="sequence",
        presentation_col="presentation",
        label="LeakyAttwiseDiscount",
        parameter_names=["wp", "beta", "omega", "eta"],
        parameter_bounds=[(0, 1), (0, 200), (0, 1), (0, 1)],
    ):
        super().__init__()  # super(LeakyAttwiseDiscount, self).__init__()
        self.data = data
        self.probability_cols = probability_cols
        self.probabilities = data[probability_cols].values
        self.outcomes = data[outcome_cols].values / 10
        self.sequences = data[sequence_col].values
        self.presentations = data[presentation_col].values
        self.choices = data["choice"].values
        self.label = label
        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds
        self.n_parameters = len(parameter_names)
        self.n_trials = len(data)

    def predict_choiceprobs(self, parameters):
        wp, beta, omega, eta = parameters
        n_attributes = 2

        dp = np.vstack(
            [
                self.probabilities[:, 0] - self.probabilities[:, 1],
                self.probabilities[:, 1] - self.probabilities[:, 0],
            ]
        ).T
        dm = np.vstack(
            [
                self.outcomes[:, 0] - self.outcomes[:, 1],
                self.outcomes[:, 1] - self.outcomes[:, 0],
            ]
        ).T

        V = np.zeros_like(self.outcomes)

        for trial in range(self.n_trials):
            sequence_trial = json.loads(self.sequences[trial])
            fixation_durations = np.array(sequence_trial["durations"]) / 1000
            fixated_attributes = np.array(sequence_trial["attributes"])
            for i, (dur, target) in enumerate(
                zip(fixation_durations, fixated_attributes)
            ):
                if target in ["p", "m"]:
                    if target == "p":
                        target_int = 0
                    else:
                        target_int = 1
                    eta_vec = np.ones(n_attributes) * eta
                    eta_vec[target_int] = 1
                else:  # target is "all", both alternatives visible
                    eta_vec = np.ones(n_attributes)
                s = eta_vec[0] * wp * dp[trial, :] + (1 - wp) * eta_vec[1] * dm[trial]
                V[trial, :] = omega ** dur * V[trial, :] + s * dur

        choiceprobs = softmax(beta * V)
        return choiceprobs