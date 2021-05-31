import pandas as pd
import numpy as np
from os.path import join, exists
from src.utilities import mkdir_if_needed
from tqdm import tqdm
from copy import deepcopy
from ddm import Sample, fit_adjust_model
from ddm.models import LossRobustBIC


def fit_predict_individual_model(
    df,
    subject,
    model,
    required_conditions,
    output_dir,
    label,
    overwrite=False,
    seed=0,
    n_runs=1,
    n_reps=1,
    choice_column_name="choice",
    rt_column_name="rt",
    subject_column_name="subject_id",
    condition_column_name="condition",
    verbose=True,
    upper_alternative=1,
):
    """
    This function
    - fits a model to data in a DataFrame individually
    - saves the estimates as a DataFrame
    - saves fitted model strings to be read later
    - predicts mean RT and accuracy for all conditions
    - saves predictions as a DataFrame
    - creates a synthetic predicted dataset of the same size as the original data
    """
    print(f"Fitting {label} for subject {subject}...")
    mkdir_if_needed(join(output_dir, label), verbose=False)
    for subfolder in ["fitted_models", "estimates", "predictions", "synthetic"]:
        mkdir_if_needed(join(output_dir, label, subfolder), verbose=False)

    estimates_file = join(
        output_dir, label, "estimates", f"{label}_{subject}_estimates.csv"
    )
    prediction_file = join(
        output_dir, label, "predictions", f"{label}_{subject}_predictions.csv"
    )
    synthetic_file = join(
        output_dir, label, "synthetic", f"{label}_{subject}_synthetic.csv"
    )
    fitted_models_path = join(output_dir, label, "fitted_models")

    if overwrite or (not (exists(estimates_file) and exists(prediction_file))):
        if exists(estimates_file) and exists(prediction_file):
            print(f"Overwriting existing results at '{estimates_file}'.")
        predictions = []
        estimates = []
        synthetic = []

        model = deepcopy(model)

        # Process data: Subset to subject, drop trials with missing values on any of the required conditions
        df_s = df.loc[
            df[subject_column_name] == subject,
            [
                subject_column_name,
                condition_column_name,
                choice_column_name,
                rt_column_name,
            ]
            + required_conditions,
        ]
        n_removed = len(df_s) - len(df_s.dropna(axis=0))
        df_s = df_s.dropna(axis=0).reset_index(drop=True)
        print(f"    Data includes {len(df_s)} trials.")
        if verbose > 0:
            if n_removed > 0:
                print(
                    f"  Subject {subject}\t{label}\tRemoved {n_removed} trials with missing responses."
                )
        # Format choice column:
        # pyddm thinks in 'correct' (True, upper bound) and 'error' (False, lower bound)
        # So if we coded the `upper_alternative` as 1 in our choice column, all is good. If we coded the `upper_alternative` as 0, however, we need to reformat this.
        if upper_alternative == 1:
            df_s[f"{choice_column_name}_formatted"] = df_s[choice_column_name]
        elif upper_alternative == 0:
            if verbose > 0:
                print(
                    "Recoding choice column so that upper boundary corresponds to alternative 1, as assumed by pyDDM."
                )
            df_s[f"{choice_column_name}_formatted"] = 1 - df_s[choice_column_name]

        # Read data into pyDDM sample object
        sample = Sample.from_pandas_dataframe(
            df_s[
                [f"{choice_column_name}_formatted", rt_column_name]
                + required_conditions
            ],
            rt_column_name=rt_column_name,
            correct_column_name=f"{choice_column_name}_formatted",
        )

        # Fit model
        loss = np.inf
        best_fit = None
        for run in range(n_runs):
            if verbose:
                print(f"  Fitting model (run {run + 1} of {n_runs})...")
            fit = fit_adjust_model(
                sample=sample,
                model=model,
                verbose=False,
                lossfunction=LossRobustBIC,
            )
            if fit.fitresult.value() < loss:
                if verbose:
                    print("    Best value:", fit.fitresult.value())
                loss = fit.fitresult.value()
                best_fit = fit

        # Save fitted object as a string
        with open(join(fitted_models_path, f"{label}_{subject}.txt"), "w") as text_file:
            print(best_fit, file=text_file)

        # Format results and save to .csv after each subject
        estimates.append(deepcopy(best_fit))
        fit_df = make_result_df(estimates).round(4)
        fit_df["model"] = label
        fit_df[subject_column_name] = subject
        fit_df.to_csv(estimates_file)

        # For each condition, predict mean RT and accuracy, and create synthetic data
        if verbose:
            print("Generating model predictions...")
        for index, trial in df_s.iterrows():
            conditions = {cond: trial[cond] for cond in required_conditions}
            solution = model.solve(conditions=conditions)
            mean_rt = solution.mean_decision_time()
            p_choose_0 = solution.prob_correct()

            # Save predictions
            trial_df = pd.DataFrame(trial).T
            trial_df[condition_column_name] = trial[condition_column_name]
            trial_df["model"] = label
            # Mean accuracy and RT
            trial_df["pred_mean_rt"] = np.round(mean_rt, 4)
            trial_df["pred_p_choose_0"] = np.round(p_choose_0, 4)
            predictions.append(trial_df)

            # Synthetic data
            synthetic_trial = trial_df.copy()
            try:
                resample = solution.resample(k=n_reps, seed=seed).to_pandas_dataframe(
                    rt_column_name="pred_" + rt_column_name,
                    correct_column_name="pred_" + choice_column_name,
                )
            except ValueError:  # predicted undecided
                resample = pd.DataFrame(
                    {
                        "pred_" + rt_column_name: np.nan,
                        "pred_" + choice_column_name: np.nan,
                    },
                    index=[index],
                )

            synthetic_trial["pred_" + choice_column_name] = resample[
                "pred_" + choice_column_name
            ].values[0]
            synthetic_trial["pred_" + rt_column_name] = np.round(
                resample["pred_" + rt_column_name].values[0], 4
            )

            synthetic.append(synthetic_trial)
        pred_df = pd.concat(predictions).reset_index(drop=True)
        pred_df.to_csv(prediction_file)
        synth_df = pd.concat(synthetic).reset_index(drop=True)
        synth_df.to_csv(synthetic_file)

    else:
        print(
            f"Found existing results at: '{estimates_file}'. Skipping estimation and reading those results instead."
        )
        fit_df = pd.read_csv(estimates_file, index_col=0)
        pred_df = pd.read_csv(prediction_file, index_col=0)
        synth_df = pd.read_csv(synthetic_file, index_col=0)

    return fit_df, pred_df, synth_df


def format_result(fit):
    """
    Formats a pyDDM fit object into something more useful
    by extracting parameter estimates and the loss function value.
    """
    parameter_names = fit.get_model_parameter_names()
    estimates = [fitted.real for fitted in fit.get_model_parameters()]

    estimate_dict = {
        parameter: estimate for parameter, estimate in zip(parameter_names, estimates)
    }

    lossfun = fit.fitresult.loss
    lossval = fit.fitresult.value()

    return {
        "estimates": estimate_dict,
        "loss": fit.fitresult.value(),
        "lossfun": fit.fitresult.loss,
        "model": fit.name,
        "n_params": None,
    }


def make_result_df(fit_list):
    """
    Make a pandas.DataFrame from a list of pyDDM fit objects.
    """
    dfs = []
    for fit in fit_list:
        result_dict = format_result(fit)
        df_i = pd.DataFrame(result_dict["estimates"], index=[0])
        df_i["loss"] = result_dict["loss"]
        df_i["lossfun"] = result_dict["lossfun"]
        dfs.append(df_i)
    return pd.concat(dfs).reset_index(drop=True)