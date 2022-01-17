import json
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_conditions(
    n_blocks=2,
    n_core=15,  # desired number of trials
    n_catch=20,
    ms=np.arange(0.2, 0.81, 0.1),
    m_max=10,
    ps=np.arange(0.2, 0.81, 0.1),
    m_range=0.03,
    p_range=0.03,
    implied_alpha_min=0.1,
    implied_alpha_max=3,
    catch_ev_max=0.3,
    min_d=0.5,
    min_d_dec=0.01,
    max_iters=2000,
    alpha_dist={
        (0.1, 0.3): 1,
        (0.3, 0.5): 1,
        (0.5, 0.7): 2,
        (0.7, 0.9): 2,
        (0.9, 1.1): 2,
        (1.1, 1.3): 2,
        (1.3, 1.5): 1,
        (1.5, 1.7): 1,
        (1.7, 1.9): 1,
        (1.9, 2.5): 1,
        (2.5, 3.0): 1,
    },
    n_presentations_each=2,
    target_duration=1500,
    other_duration=1000,
    catch_duration=1250,
    seed=None,
):
    """Main function to generate the core conditions DataFrame.

    Additional variables are added using additional functions below.
    """
    if seed is not None:
        np.random.seed(seed)

    conditions = []

    # 1. Generate all possible pairs with a high-p low-m ("l") and a low-p high-m ("h") lottery
    trials = []
    i = 0
    for p_l in ps:
        for p_h in ps:
            if p_h >= p_l:
                continue
            for m_l in ms:
                for m_h in ms:
                    if m_l >= m_h:
                        continue
                    trials.append(
                        pd.DataFrame(
                            dict(
                                p_l=np.round(
                                    p_l + np.random.uniform(-p_range, p_range), 2
                                ),
                                p_h=np.round(
                                    p_h + np.random.uniform(-p_range, p_range), 2
                                ),
                                m_l=np.round(
                                    m_l + np.random.uniform(-m_range, m_range), 2
                                ),
                                m_h=np.round(
                                    m_h + np.random.uniform(-m_range, m_range), 2
                                ),
                            ),
                            index=[i],
                        )
                    )
                    i += 1
    trials = pd.concat(trials)

    # Implied alpha if indifferent
    trials["implied_alpha"] = np.log(trials["p_l"] / trials["p_h"]) / np.log(
        trials["m_h"] / trials["m_l"]
    )

    # 2. Filtering and sampling
    # Obviously, these are too many trials, so we try to filter and sample a subset that makes sense. First we restrict the range of implied alphas:
    # Restrict range of implied alphas
    trials_trunc = trials.loc[
        (trials["implied_alpha"] < implied_alpha_max)
        & (trials["implied_alpha"] > implied_alpha_min)
    ]

    # Next, we sample from the trial pool so that
    # 1. Trials in our subset are different from each other
    # 2. We control the distribution of implied alphas

    # Initialize counting dictionary to keep track of alpha distribution
    alpha_counts = {alpha_bin: 0 for alpha_bin in alpha_dist.keys()}

    trials_sample = trials.loc[
        :-1
    ]  # start with an empty DataFrame that has the correct columns
    done = False
    while not done:
        i = 0
        while (len(trials_sample) < n_core) & (i <= max_iters):
            # Sample a trial
            trial = trials_trunc.sample(1)
            i += 1

            # Check alpha distribution
            ia = trial["implied_alpha"].values[0]
            for key, value in alpha_dist.items():
                (lower, upper) = key
                if (
                    (ia >= lower)
                    & (ia <= upper)
                    & (alpha_counts[key] < alpha_dist[key])
                ):

                    # Compute distances if this is not the first trial
                    if len(trials_sample) > 0:
                        d = np.sqrt(
                            (trial["p_l"].values - trials_sample["p_l"].values) ** 2
                            + (trial["p_h"].values - trials_sample["p_h"].values) ** 2
                            + (trial["m_l"].values - trials_sample["m_l"].values) ** 2
                            + (trial["m_h"].values - trials_sample["m_h"].values) ** 2
                        )
                    else:
                        d = np.array(
                            [np.inf]
                        )  # first trial is infinitely different to "others"
                    # Check distances
                    if np.any(d < min_d):
                        continue

                    # If all good, append the trial
                    else:
                        trials_sample = trials_sample.append(trial)
                        alpha_counts[key] += 1
                        print(
                            f"  Added a trial with min_d = {d.min():.2f} and ia = {ia:.2f}"
                        )
                        print(f"    {alpha_counts}")

                    # Quit if there's enough trials
                    if len(trials_sample) == n_core:
                        done = True

        # If this failed, reduce minimal distance
        if not done:
            print(
                f"Reached {max_iters} iterations with min_d = {min_d:.2f}. Decreasing min_d by {min_d_dec:.2f}."
            )
            min_d -= min_d_dec
            # If minimal distance can't be lowered, quit
            if min_d <= 0:
                print(f"Could only find {len(trials_sample)} trials.")
                break

    # 3. Make factorial design conditions (2 x 2 x 2 = 8) for each trial in trials_sample
    experimental = []
    i = 0
    for index, trial in trials_sample.iterrows():
        for target in [0, 1]:
            other = 1 - target
            # 2 target first, target second
            for target_first in [True, False]:
                if target_first:
                    durations = [
                        target_duration,
                        other_duration,
                    ] * n_presentations_each
                else:
                    durations = [
                        other_duration,
                        target_duration,
                    ] * n_presentations_each

                # 2 alternative-wise or attribute-wise presentation
                for sequence_kind in ["alternatives", "attributes"]:
                    if sequence_kind == "alternatives":
                        attributes = ["all"] * n_presentations_each * 2
                        targetId = target
                        otherId = other
                        if target_first:
                            alternatives = [
                                target,
                                other,
                            ] * n_presentations_each
                        else:
                            alternatives = [
                                other,
                                target,
                            ] * n_presentations_each
                    else:
                        alternatives = ["all"] * n_presentations_each * 2
                        targetId = ["p", "m"][target]
                        otherId = ["p", "m"][other]
                        if target_first:
                            attributes = [
                                targetId,
                                otherId,
                            ] * n_presentations_each
                        else:
                            attributes = [
                                otherId,
                                targetId,
                            ] * n_presentations_each
                    sequence = dict(
                        durations=durations,
                        alternatives=alternatives,
                        attributes=attributes,
                    )

                    condition = pd.DataFrame(
                        dict(
                            condition="exp_{}".format(i),
                            p0=trial["p_l"],
                            p1=trial["p_h"],
                            m0=trial["m_l"],
                            m1=trial["m_h"],
                            target=targetId,
                            other=otherId,
                            target_first=target_first,
                            presentation=sequence_kind,
                            sequence=json.dumps(sequence),
                            core_id=index,
                            alt0="high_p",
                            alt1="high_m",
                        ),
                        index=np.ones(1) * i,
                    )
                    experimental.append(condition)
                    i += 1
    experimental = pd.concat(experimental)

    # Add block variable. Trials are randomly assigned to blocks
    block = np.repeat(np.arange(n_blocks), (len(experimental) // n_blocks))
    np.random.shuffle(block)
    experimental["block"] = block

    # Expected values and their differences
    experimental["ev0"] = np.round(experimental["p0"] * experimental["m0"], 2)
    experimental["ev1"] = np.round(experimental["p1"] * experimental["m1"], 2)
    experimental["ev_diff"] = np.round(experimental["ev0"] - experimental["ev1"], 2)

    # Implied alpha if indifferent
    experimental["implied_alpha"] = np.round(
        np.log(experimental["p0"] / experimental["p1"])
        / np.log(experimental["m1"] / experimental["m0"]),
        2,
    )

    # 4. Add catch trials:
    # First, create a large number of trials where alternative h (1) dominates alternative l (0)
    catch = []
    i = 0
    for p_hi in ps:
        for p_lo in ps:
            if p_hi <= p_lo:
                continue
            for m_hi in ms:
                for m_lo in ms:
                    if m_hi <= m_lo:
                        continue
                    catch.append(
                        pd.DataFrame(
                            dict(
                                p0=np.round(
                                    p_lo + np.random.uniform(-p_range, p_range), 2
                                ),
                                p1=np.round(
                                    p_hi + np.random.uniform(-p_range, p_range), 2
                                ),
                                m0=np.round(
                                    m_lo + np.random.uniform(-m_range, m_range), 2
                                ),
                                m1=np.round(
                                    m_hi + np.random.uniform(-m_range, m_range), 2
                                ),
                            ),
                            index=[i],
                        )
                    )
                    i += 1
    catch = pd.concat(catch)

    # Now filter catch trials
    # Expected values and their differences
    catch["ev0"] = np.round(catch["p0"] * catch["m0"], 2)
    catch["ev1"] = np.round(catch["p1"] * catch["m1"], 2)
    catch["ev_diff"] = np.round(catch["ev0"] - catch["ev1"], 2)

    # Implied alpha if indifferent
    catch["implied_alpha"] = np.round(
        np.log(catch["p0"] / catch["p1"]) / np.log(catch["m1"] / catch["m0"]), 2
    )

    # Filter based on EV of dominating option
    catch_sample = catch.loc[catch["ev1"] < catch_ev_max].sample(n_catch)

    # Add additional variables
    catch_sample["block"] = np.repeat(range(n_blocks), n_catch // n_blocks)
    catch_sample["condition"] = [f"catch_{i}" for i in range(n_catch)]
    catch_sample["alt0"] = "dominated"
    catch_sample["alt1"] = "dominating"
    catch_sample["presentation"] = (n_catch // 2) * ["alternatives", "attributes"]
    catch_sample["sequence"] = (
        n_catch
        // 2
        * [
            # Alternative-wise sequence
            json.dumps(
                dict(
                    durations=[catch_duration] * 2 * n_presentations_each,
                    attributes=["all"] * 2 * n_presentations_each,
                    alternatives=[0, 1] * n_presentations_each,
                )
            ),
            # Attribute-wise sequence
            json.dumps(
                dict(
                    durations=[catch_duration] * 2 * n_presentations_each,
                    attributes=["p", "m"] * n_presentations_each,
                    alternatives=["all"] * 2 * n_presentations_each,
                )
            ),
        ]
    )

    # Save and output
    conditions.append(experimental)
    conditions.append(catch_sample)
    conditions = pd.concat(conditions)
    conditions[["m0", "m1"]] *= m_max
    conditions.reset_index(drop=True, inplace=True)

    return conditions


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
    """Adds variables for relative durations towards alternatives and attributes.

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


def add_last_favours_var(df):
    """Adds variable that describes which alternative is favoured by the last presentation step in the sequence.

    Args:
        df (pandas.DataFrame): DataFrame with conditions. Must contain columns `presentation`, `target_first`, `target`, `other`, `p0`, `p1`, `m0`, `m1`.

    Returns:
        pandas.DataFrame: The DataFrame with added `last_favours` column.
    """
    df["last_favours"] = np.where(
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


if __name__ == "__main__":

    conditions = build_conditions(
        n_blocks=2,
        n_core=15,  # desired number of trials
        n_catch=20,
        ms=np.arange(0.2, 0.81, 0.1),
        ps=np.arange(0.2, 0.81, 0.1),
        m_range=0.03,
        p_range=0.03,
        implied_alpha_min=0.1,
        implied_alpha_max=3,
        min_d=0.5,
        min_d_dec=0.01,
        max_iters=2000,
        alpha_dist={
            (0.1, 0.3): 1,
            (0.3, 0.5): 1,
            (0.5, 0.7): 2,
            (0.7, 0.9): 2,
            (0.9, 1.1): 2,
            (1.1, 1.3): 2,
            (1.3, 1.5): 1,
            (1.5, 1.7): 1,
            (1.7, 1.9): 1,
            (1.9, 2.5): 1,
            (2.5, 3.0): 1,
        },
        n_presentations_each=2,
        target_duration=1500,
        other_duration=1000,
        catch_duration=1250,
        seed=2,
    )

    conditions["phase"] = "experimental"

    # Identify options with higher P and higher M in each trial
    conditions["higher_p"] = (
        conditions[["p0", "p1"]].idxmax(axis=1).apply(lambda x: int(x[-1]))
    )
    conditions["higher_m"] = (
        conditions[["m0", "m1"]].idxmax(axis=1).apply(lambda x: int(x[-1]))
    )

    conditions = add_duration_vars(conditions)
    conditions = add_last_favours_var(conditions)

    # Add numerical `presentation` variable for pyDDM
    conditions["presentation01"] = np.where(
        conditions["presentation"] == "alternatives",
        0,
        np.where(conditions["presentation"] == "attributes", 1, np.nan),
    )

    conditions.to_json(join("stimuli", "conditions_main.json"), orient="records")
    conditions.to_csv(join("stimuli", "conditions_main.csv"))