#!/usr/bin/python
import argparse
from os.path import join

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import myfuncs as my
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from src.utilities import mkdir_if_needed, save_idata_results


def run_lmm_random_slopes_intercepts(
    data, dependent_var, predictors, interactions, subject_col, family, **fit_kwargs
):
    """Build and estimate a linear mixed model
    with random slopes and intercepts for each subject.

    Args:
        data (pandas.DataFrame): The data.
        dependent_var (str): The dependent variable.
        predictors (list): A list of predictors.
        interactions (list): List of interaction terms.
        subject_col (str): The name of the variable coding subjects in the data. Used to create random effects over subjects.
        family (str): Model family, e.g., `"gaussian"` or `"bernoulli"`. See https://bambinos.github.io/bambi/notebooks/getting_started.html#Generalized-linear-mixed-models

    Returns:
        arviz.InferenceData
    """
    model = bmb.Model(data=data[[subject_col, dependent_var] + predictors], dropna=True)
    result = model.fit(
        f"{dependent_var} ~ 0 + (1|{subject_col}) + "
        + " + ".join(predictors)
        + " + "
        + " + ".join([f"({predictor}|{subject_col})" for predictor in predictors])
        + " + "
        + " + ".join(interactions)
        + " + "
        + " + ".join(
            [f"({interaction}|{subject_col})" for interaction in interactions]
        ),
        family=family,
        **fit_kwargs,
    )
    return result


def run_choice_analyses():

    # Load choice data
    choices = pd.read_csv(args.choice_file)

    # Subset to experimental conditions, WITHOUT CATCH TRIALS
    choices = choices.loc[choices["condition"].str.startswith("exp")].copy()

    # Effect coding of categorical variables
    for predictor in ["duration_favours", "last_stage_favours", "by_attribute"]:
        choices[f"{predictor}_fx"] = np.where(
            choices[predictor] == 0,
            -0.5,
            np.where(choices[predictor] == 1, +0.5, np.nan),
        )

    # 1. GLM: choice ~ 1 + ΔEV + longer_shown + favoured_by_last_stage
    glm_idata = run_lmm_random_slopes_intercepts(
        data=choices,
        dependent_var="choice",
        predictors=[
            "delta_ev_z",
            "duration_favours_fx",
            "last_stage_favours_fx",
            "by_attribute_fx",
        ],
        interactions=[
            "duration_favours_fx:by_attribute_fx",
            "last_stage_favours_fx:by_attribute_fx",
        ],
        subject_col="subject_id",
        family="bernoulli",
        cores=1,
    )
    save_idata_results(idata=glm_idata, label="glm", output_dir=args.output_dir)

    # 2. Run BF t-tests of sequence and duration main effects
    ## Compute individual marginal choice probabilities for each condition
    for independent_var, ivar_label in zip(
        ["duration_favours", "last_stage_favours"], ["duration", "sequence"]
    ):
        cp = (
            choices.groupby([independent_var, "subject_id"])["choice"]
            .mean()
            .reset_index()
            .pivot_table(values="choice", index="subject_id", columns=independent_var)
        )
        # BF undirected
        bf = my.stats.ttestbf.two_sample_ttestbf(cp[1], cp[0], paired=True)
        bf.to_csv(join(args.output_dir, f"ttestbf_{ivar_label}.csv"))
        # BF directed
        bf_directed = my.stats.ttestbf.BayesFactor.extractBF(
            my.stats.ttestbf.BayesFactor.ttestBF(
                cp[1], cp[0], paired=True, nullInterval=[0, np.inf]
            )
        )
        bf_directed.to_csv(join(args.output_dir, f"ttestbf-directed_{ivar_label}.csv"))
        # BEST
        best_idata = my.stats.best.one_sample_best(
            cp[1] - cp[0], sigma_low=0.0001, sample_kwargs={"cores": 1}
        )
        save_idata_results(
            idata=best_idata,
            label=f"best_{ivar_label}",
            output_dir=args.output_dir,
        )

    # 3. Run 1-Sample BEST and BF t-Tests of P(choose higher p | duration_favours / last_stage_favours == {higher_p, higher_m})

    for independent_var, ivar_label in zip(
        ["duration_favours_str", "last_stage_favours_str"], ["duration", "sequence"]
    ):
        # Format data: Compute P(Choose higher p) for each subject, presentation and manipulation condition
        p_choose_higher_p = (
            choices.groupby(["subject_id", "presentation", independent_var])[
                "choose_higher_p"
            ]
            .mean()
            .reset_index()
            .pivot_table(
                index=["subject_id", "presentation"],
                columns=independent_var,
                values="choose_higher_p",
            )
            .reset_index()
        )

        for presentation in ["alternatives", "attributes"]:
            p_choose_higher_p_pres = p_choose_higher_p.loc[
                p_choose_higher_p["presentation"] == presentation
            ].reset_index(drop=True)

            # Save data
            p_choose_higher_p_pres.to_csv(
                join(args.output_dir, f"best_{ivar_label}_{presentation}_data.csv")
            )

            # Compute difference between choice probabilities for different manipulation targets (higher p vs. higher m)
            difference = (
                p_choose_higher_p_pres["higher_p"] - p_choose_higher_p_pres["higher_m"]
            )

            # Run frequentist tests
            W, p = wilcoxon(difference)
            print(f"\n{ivar_label.capitalize()} manipulation")
            print(f"  Presentation: {presentation[:-1]}-wise")
            print(f"    Mean difference {difference.mean():.2f}")
            print(f"    Wilcoxon test: W = {W:.2f}, p = {p:.4f}\n")

            # Run BEST
            best_idata = my.stats.best.one_sample_best(
                difference, sigma_low=0.0001, sample_kwargs={"cores": 1}
            )
            save_idata_results(
                idata=best_idata,
                label=f"best_{ivar_label}_{presentation}",
                output_dir=args.output_dir,
            )

            # Run Bayes Factor t-Test
            bf = my.stats.ttestbf.one_sample_ttestbf(difference)
            bf.to_csv(join(args.output_dir, f"ttestbf_{ivar_label}_{presentation}.csv"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", default=0, type=int, help="Set verbosity (0, 1, >1)."
    )
    parser.add_argument(
        "--choice-file",
        type=str,
        default="data/processed/choices.csv",
        help="Relative path to preprocessed choice data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/choice_analyses",
        help="Where to save results.",
    )

    parser.add_argument("--seed", type=int, default=2021, help="Random number seed.")

    args = parser.parse_args()

    mkdir_if_needed(args.output_dir)

    np.random.seed(args.seed)

    run_choice_analyses()
