import numpy as np
from ddm import Sample, fit_adjust_model
from ddm.models import LossRobustBIC

from src.models.ddms.data import TrialData
from src.models.ddms.fitting import make_result_df
from src.models.ddms.agent import Agent


class TwoStagePyDDM(Agent):
    def __init__(self, dt=0.01, agentVars=None):
        super().__init__(agentVars=agentVars)
        self.dt = dt

    def make_stage_two_choice(self, trial, maxSteps, seed=None):
        """Run the stage 2 evidence accumulation process.

        Args:
            trial ([type]): [description]
            maxSteps (int): Maximum number of accumulation steps.
            seed (int, optional): Numpy random seed. Defaults to None.

        Returns:
            TrialData: Response, response time, trajectory, drift, and trial information.
        """
        if seed is not None:
            np.random.seed(seed)

        # Construct drift coefficient from trial information
        drift = self.compute_stage_two_drift(trial)

        # Draw trial parameters
        t0 = np.random.normal(loc=self.agentVars.t0, scale=self.agentVars.st0)
        t0steps = int(np.round(t0 / self.dt))
        v = np.random.normal(loc=self.agentVars.v, scale=self.agentVars.sv)
        z = np.random.normal(loc=self.agentVars.z, scale=self.agentVars.sz)
        a = self.agentVars.a
        xInit = z * a

        # Initialize evidence vector and noise
        X = np.zeros(t0steps + maxSteps + 1) * np.nan
        X[: (t0steps + 1)] = xInit
        e = np.random.normal(
            loc=0, scale=self.agentVars.s / np.sqrt(1 / self.dt), size=X.size
        )

        # Accumulation
        t = t0steps
        while t < (t0steps + maxSteps):
            t += 1
            X[t] = X[t - 1] + self.dt * v * drift + e[t]

            if (X[t] >= a) or (X[t] <= 0):
                rt = t * self.dt
                if X[t] >= a:
                    response = 0
                else:
                    response = 1
                return TrialData(
                    response=response,
                    rt=rt,
                    X=X,
                    drift=drift,
                    status="threshold crossed",
                    trial=trial,
                )
        return TrialData(
            response="undecided",
            rt=maxSteps * self.dt,
            X=X,
            drift=drift,
            status="reached maxSteps",
            trial=trial,
        )

    def predict_choice_rt(
        self, data, maxSteps, choice_col="pred_choice", rt_col="pred_rt"
    ):
        """Predicts choice and RT data for each trial row contained in data.

        Args:
            data (pandas.DataFrame): DataFrame containing one trial per row.
            maxSteps (int): Maximum number of steps in simulation.

        Returns:
            pandas.DataFrame: `data` with added columns for predicted choices and RT.
        """
        predictions = data.apply(
            lambda x: self.make_stage_two_choice(x, maxSteps=maxSteps), axis=1
        ).tolist()
        data[choice_col] = [pred.response for pred in predictions]
        data[rt_col] = [pred.rt for pred in predictions]
        return data

    def fit_model(
        self,
        data,
        choice_column_name,
        rt_column_name,
        n_runs=1,
        lossfunction=LossRobustBIC,
        verbose=True,
        upper_alternative=1,
    ):
        # Build model if it's not there already
        try:
            model = self.model
        except AttributeError:
            if verbose > 0:
                print(
                    f"Building pyDDM model with dx={self.dx}, dt={self.dt}, and T_dur={self.T_dur}..."
                )
            self.model = self.build_model()
            model = self.model

        # Remove undecided trials and trials with RT out of range
        df = data.loc[
            (data[choice_column_name].isin([0, 1]))
            & (data[rt_column_name] < self.T_dur)
        ].copy()
        n_removed = len(data) - len(df)
        if n_removed > 0:
            if verbose > 0:
                print(f"Removed {n_removed} trials with missing or slow responses...")

        # Format choice column:
        # pyddm thinks in 'correct' (True, upper bound) and 'error' (False, lower bound)
        # So if we coded the `upper_alternative` as 1 in our choice column, all is good. If we coded the `upper_alternative` as 0, however, we need to reformat this.
        if upper_alternative == 1:
            df[f"{choice_column_name}_formatted"] = df[choice_column_name]
        elif upper_alternative == 0:
            if verbose > 0:
                print(
                    "Recoding choice column so that upper boundary corresponds to alternative 1, as assumed by pyDDM."
                )
            df[f"{choice_column_name}_formatted"] = 1 - df[choice_column_name]

        # Read data into pyDDM sample object
        sample = Sample.from_pandas_dataframe(
            df[
                [f"{choice_column_name}_formatted", rt_column_name]
                + self.required_conditions
            ],
            rt_column_name=rt_column_name,
            correct_column_name=f"{choice_column_name}_formatted",
        )

        # Fit model
        loss = np.inf
        best_fit = None
        for run in range(n_runs):
            if verbose > 0:
                print(f"  Fitting model (run {run + 1} of {n_runs})...")
            fit = fit_adjust_model(
                sample=sample,
                model=model,
                verbose=(verbose > 1),
                lossfunction=lossfunction,
            )
            if fit.fitresult.value() < loss:
                if verbose > 0:
                    print("    Best value:", fit.fitresult.value())
                loss = fit.fitresult.value()
                best_fit = fit
        self.fit = best_fit

        # Format results and save to .csv after each subject
        fit_df = make_result_df([best_fit])
        fit_df["model"] = self.label
        self.fit_df = fit_df

        return best_fit
