# -*- coding: utf-8 -*-
import argparse
import json
from os import listdir
from os.path import join

import numpy as np
import pandas as pd
from src.utilities import mkdir_if_needed


def read_presentation_type(sequence):
    """
    This function extracts the presentation_type variable from a sequence dictionary.
    """
    if sequence["alternatives"][0] in [0, 1]:
        return "alternatives"
    elif sequence["attributes"][0] in ["p", "m"]:
        return "attributes"


def compute_durations(sequence, alternative=None, attribute=None):
    """Computes the relative presentation duration of alternatives, attributes, or combinations of both for a given sequence.

    Args:
        sequence (dict): Sequence dictionary with keys "attributes", "alternatives" and "durations", each containing a list.
        alternative (int, optional): Index of alternative for which overall relative duration should be computed. Defaults to None.
        attribute (str, optional): Attribute for which overall relative duration should be computed. For example "p" or "m". Defaults to None.

    Returns:
        float: Relative duration measure.
    """
    if alternative is not None:
        alt_mask = np.array(
            [alt in [alternative, "all"] for alt in sequence["alternatives"]]
        )
    else:
        alt_mask = np.ones(len(sequence["alternatives"])).astype(bool)

    if attribute is not None:
        att_mask = np.array(
            [att in [attribute, "all"] for att in sequence["attributes"]]
        )
    else:
        att_mask = np.ones(len(sequence["attributes"])).astype(bool)

    g = np.sum(np.array(sequence["durations"])[alt_mask & att_mask]) / np.sum(
        np.array(sequence["durations"])
    )
    return g


def add_duration_vars(df):
    """Adds variables for relative durations towards alernatives and attributes.

    Args:
        df (pandas.DataFrame): Dataframe with `sequence` variable containing the presentation sequence.

    Returns:
        pandas.DataFrame: The DataFrame with added variables.
    """
    for alt in [0, 1]:
        df[f"g{alt}r"] = df.apply(
            lambda x: compute_durations(json.loads(x["sequence"]), alternative=alt),
            axis=1,
        )
    for att in ["p", "m"]:
        df[f"g{att}r"] = df.apply(
            lambda x: compute_durations(json.loads(x["sequence"]), attribute=att),
            axis=1,
        )

    # Normalize durations to 1 in each trial
    df["g0"] = df["g0r"] / df[["g0r", "g1r"]].sum(axis=1)
    df["g1"] = df["g1r"] / df[["g0r", "g1r"]].sum(axis=1)
    df["gm"] = df["gmr"] / df[["gmr", "gpr"]].sum(axis=1)
    df["gp"] = df["gpr"] / df[["gmr", "gpr"]].sum(axis=1)

    return df.drop(["g0r", "g1r", "gmr", "gpr"], axis=1)


def add_last_stage_favours_var(df):
    """Adds variable that describes which alternative is favoured by the last presentation step in the sequence.

    Args:
        df (pandas.DataFrame): DataFrame with conditions. Must contain columns `presentation`, `targetFirst`, `target`, `other`, `p0`, `p1`, `m0`, `m1`.

    Returns:
        pandas.DataFrame: The DataFrame with added `lastFavours` column.
    """
    df["last_stage_favours"] = np.where(
        df["presentation"] == "alternatives",
        df["sequence"].apply(lambda x: json.loads(x)["alternatives"][-1]),
        np.where(
            df["presentation"] == "attributes",
            np.where(
                df["sequence"].apply(lambda x: json.loads(x)["attributes"][-1] == "p"),
                df["higher_p"],
                df["higher_m"],
            ),
            np.nan,
        ),
    ).astype(float)
    return df


def add_duration_favours_var(choices):
    # Add target variable, coding which alternative is favoured by hypothesized duration effect
    choices["duration_favours"] = np.where(
        choices["condition"].str.startswith("exp_"),
        np.where(
            choices["presentation"] == "alternatives",
            choices[["g0", "g1"]].idxmax(axis=1).str[1],
            np.where(
                choices[["gp", "gm"]].idxmax(axis=1).str[1] == "p",
                choices["higher_p"],
                choices["higher_m"],
            ),
        ),
        np.nan,
    ).astype(float)
    return choices


def add_misc_variables(choices):
    # Add necessary variables
    choices["label0"] = np.where(
        choices["condition"].str.startswith("catch"),
        "dominated",
        np.where(choices["higher_p"] == 0, "higher_p", "higher_m"),
    )
    choices["label1"] = np.where(
        choices["condition"].str.startswith("catch"),
        "dominant",
        np.where(choices["higher_p"] == 1, "higher_p", "higher_m"),
    )
    choices["duration_favours_str"] = np.where(
        choices["duration_favours"] == 0,
        choices["label0"],
        np.where(choices["duration_favours"] == 1, choices["label1"], np.nan),
    )
    choices["last_stage_favours_str"] = np.where(
        choices["last_stage_favours"] == 0,
        choices["label0"],
        np.where(choices["last_stage_favours"] == 1, choices["label1"], np.nan),
    )

    choices["ev0"] = choices["p0"] * choices["m0"]
    choices["ev1"] = choices["p1"] * choices["m1"]
    choices["delta_ev"] = choices["ev0"] - choices["ev1"]
    choices["delta_ev_z"] = (
        choices["delta_ev"] - choices["delta_ev"].mean()
    ) / choices["delta_ev"].std(ddof=1)

    choices["choose_higher_p"] = choices["choice"] == choices["higher_p"]

    choices["by_attribute"] = choices["presentation"] == "attributes"

    choices["left_alternative"] = np.where(
        choices["pL"] == choices["p0"],
        0,
        np.where(choices["pL"] == choices["p1"], 1, np.nan),
    )

    return choices


def preprocess_choice_data(raw_data):
    """
    This function extracts and processes choice data from raw single subject jsPsych data.
    """
    # Extract only choice data
    choices = (
        raw_data.loc[
            (raw_data["trial_type"] == "two-gamble-sequence")
            & ~(raw_data["condition.1"].str.startswith("practice_"))
        ][
            [
                "condition.1",
                "rt",
                "key_press",
                "choice",
                "p0",
                "p1",
                "m0",
                "m1",
                "pL",
                "sequence",
                "webgazer_data",
            ]
        ]
        .rename({"condition.1": "condition"}, axis=1)
        .reset_index(drop=True)
        .astype({"p0": float, "p1": float, "m0": float, "m1": float, "pL": float})
    )

    # Adjust outcome values
    choices[["m0", "m1"]] *= 10

    # Handle missing responses, recode choice to integer
    for var in ["choice", "rt"]:
        choices[var] = np.where(choices[var] == '"', np.nan, choices[var])
        choices = choices.astype({var: float})

    # Identify options with higher P and higher M in each trial
    choices["higher_p"] = (
        choices[["p0", "p1"]].idxmax(axis=1).apply(lambda x: int(x[-1]))
    )
    choices["higher_m"] = (
        choices[["m0", "m1"]].idxmax(axis=1).apply(lambda x: int(x[-1]))
    )

    # Add presentation variable
    choices["presentation"] = choices.apply(
        lambda x: read_presentation_type(json.loads(x["sequence"])), axis=1
    )
    # Add numerical `presentation` variable for pyDDM
    choices["presentation01"] = np.where(
        choices["presentation"] == "alternatives",
        0,
        np.where(choices["presentation"] == "attributes", 1, np.nan),
    )

    # Add dependent variable
    choices["choose_higher_p"] = choices["choice"] == choices["higher_p"]

    # Add variables for relative presentation durations
    choices = add_duration_vars(choices)
    # Add variable coding which alternative is favoured by the last presentation stage
    choices = add_last_stage_favours_var(choices)

    choices = add_duration_favours_var(choices)

    choices = add_misc_variables(choices)

    return choices


def main():

    # Process choice data
    choices = []
    # Read keys mapping PIDs and run_id to subject keys
    subject_summary = pd.read_csv(args.subject_summary, index_col=0)
    files = [file for file in listdir(args.input_path) if file.endswith(".csv")]
    print(f"Processing choice data from {len(files)} files:")
    for file in files:
        print(f"\t{join(args.input_path, file)}")
        df = pd.read_csv(
            join(args.input_path, file), error_bad_lines=False, escapechar="\\"
        )

        # Read subject_id
        subject_id = subject_summary.loc[
            subject_summary["run_id"] == df["run_id"].values[0], "subject_id"
        ].values[0]

        # Read screen dimensions
        width = df["screen_width"].values[0]
        height = df["screen_height"].values[0]

        # Check for exclusion
        if subject_summary.loc[
            subject_summary["run_id"] == df["run_id"].values[0], "exclude"
        ].values[0]:
            print(f"Skipping data for subject {subject_id} due to exclusion criteria.")
            continue

        choices_s = preprocess_choice_data(df)
        choices_s["subject_id"] = subject_id
        choices_s["trial"] = np.arange(len(choices_s))
        choices_s["block"] = np.repeat([0, 1], len(choices_s) // 2)
        choices_s["screen_width"] = width
        choices_s["screen_height"] = height
        choices.append(
            choices_s[
                [
                    "subject_id",
                    "block",
                    "trial",
                    "condition",
                    "choice",
                    "rt",
                    "p0",
                    "p1",
                    "m0",
                    "m1",
                    "label0",
                    "label1",
                    "higher_p",
                    "higher_m",
                    "presentation",
                    "presentation01",
                    "by_attribute",
                    "left_alternative",
                    "ev0",
                    "ev1",
                    "delta_ev",
                    "delta_ev_z",
                    "g0",
                    "g1",
                    "gp",
                    "gm",
                    "duration_favours",
                    "last_stage_favours",
                    "duration_favours_str",
                    "last_stage_favours_str",
                    "choose_higher_p",
                    "sequence",
                    "webgazer_data",
                    "screen_width",
                    "screen_height",
                ]
            ]
        )

    choices = pd.concat(choices).reset_index(drop=True)
    choices.to_csv(join(args.output_path, "choices.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--subject-summary", type=str)
    args = parser.parse_args()
    mkdir_if_needed(args.output_path)

    main()
