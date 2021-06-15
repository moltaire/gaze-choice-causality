# -*- coding: utf-8 -*-
import argparse
import json
from os import listdir
from os.path import join

import numpy as np
import pandas as pd
from src.utilities import mkdir_if_needed


def summarise_subject(raw_data):
    """
    This function creates a one line summary of a single subject's data.
    The summary includes
    - Number of recorded choices
    - Number of missing responses
    - Number of choices of higher-p alternative
    - Number of choices of dominated alternatives in catch trials
    - Bonus amount won
    - Responses to red-green colourblindness and difficulties
    - Response to seriousness
    - Self-reported choice strategy
    - Subject-reported comments
    """

    # Read participant ID
    pid = raw_data["PROLIFIC_PID"].values[0]

    # Read won amount
    won_amount = raw_data["wonAmt"].values[0]
    lucky_number = raw_data["luckyNumber"].values[0]
    chosen_trial = raw_data["chosenTrial"].values[0]

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
                "sequence",
                "webgazer_data",
            ]
        ]
        .rename({"condition.1": "condition"}, axis=1)
        .astype({"p0": float, "p1": float, "m0": float, "m1": float})
        .reset_index(drop=True)
    )

    # Handle missing responses, recode choice to integer
    choices["choice"] = np.where(choices["choice"] == '"', np.nan, choices["choice"])
    choices = choices.astype({"choice": float})

    # Identify options with higher P and higher M in each trial
    choices["higher_p"] = (
        choices[["p0", "p1"]].idxmax(axis=1).apply(lambda x: int(x[-1]))
    )
    choices["higher_m"] = (
        choices[["m0", "m1"]].idxmax(axis=1).apply(lambda x: int(x[-1]))
    )

    n_records = len(choices)

    # Compute number of choices for alternatives 0, 1, and missed responses
    n_choose_0_all = np.sum(choices["choice"] == 0)
    n_choose_1_all = np.sum(choices["choice"] == 1)

    n_choose_nan = n_records - (n_choose_0_all + n_choose_1_all)
    n_choose_higher_p = np.sum(
        choices.loc[choices["condition"].str.startswith("exp")]["choice"]
        == choices.loc[choices["condition"].str.startswith("exp")]["higher_p"]
    )

    # Choose number of choices for dominated alternative in catch trials
    n_choose_dominated = (
        20
        - np.sum(
            choices.loc[choices["condition"].str.startswith("catch")]["choice"]
            == choices.loc[choices["condition"].str.startswith("catch")]["higher_p"]
        )
        - np.sum(
            pd.isnull(
                choices.loc[choices["condition"].str.startswith("catch")]["choice"]
            )
        )
    )

    # Read gender, red-green difficulty and seriousness
    mc_questionnaire = json.loads(
        raw_data.loc[raw_data["trial_type"] == "survey-multi-choice"][
            ["response"]
        ].values[-1][0]
    )

    # Read strategy and comment
    reports_string = raw_data.loc[raw_data["trial_type"] == "survey-text"][
        "response"
    ].values[0]
    age = reports_string.split('"selfReport"')[0][8:-2]
    self_report = reports_string.split('"comments":"')[0][26:-2]
    comment = reports_string.split('"comments":"')[1][:-2]
    # Put everything together
    out = pd.DataFrame(
        dict(
            pid=pid,
            gender=mc_questionnaire.get("gender", np.nan),
            age=age,
            n_records=n_records,
            n_choose_nan=n_choose_nan,
            n_choose_higher_p=n_choose_higher_p,
            n_choose_dominated=n_choose_dominated,
            chosen_trial=chosen_trial,
            lucky_number=lucky_number,
            won_amount=won_amount,
            rg_blind=(mc_questionnaire["redGreenColorBlind"] == "yes"),
            rg_difficult=(mc_questionnaire["redGreenDifficulties"] == "yes"),
            serious=(mc_questionnaire["seriousness"] == "I have taken part seriously."),
            self_report=self_report,
            comment=comment,
        ),
        index=[0],
    )
    return out


def main():

    # Summarise data quality
    summary = []
    files = [file for file in listdir(args.input_path) if file.endswith(".csv")]
    print(f"Making data overview from {len(files)} files:")
    for i, file in enumerate(files):
        print(f"\t{join(args.input_path, file)}")

        df = pd.read_csv(
            join(args.input_path, file), error_bad_lines=False, escapechar="\\"
        )

        summary_s = summarise_subject(df)
        summary_s["subject_id"] = i
        summary_s["run_id"] = df["run_id"].values[0]
        summary.append(
            summary_s[
                [
                    "subject_id",
                    "pid",
                    "gender",
                    "age",
                    "run_id",
                    "n_records",
                    "n_choose_nan",
                    "n_choose_dominated",
                    "n_choose_higher_p",
                    "chosen_trial",
                    "lucky_number",
                    "won_amount",
                    "rg_blind",
                    "rg_difficult",
                    "serious",
                    "self_report",
                    "comment",
                ]
            ]
        )

    summary = pd.concat(summary).reset_index(drop=True)
    summary.to_csv(join(args.output_path, "subject_summary.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()
    mkdir_if_needed(args.output_path)

    main()
