import argparse
import pickle
from itertools import product
from os import listdir
from os.path import join

import pandas as pd
from multiprocess import Pool
from src.models.ddms.fitting import fit_predict_individual_model
from src.models.ddms.TwoStageBetween import TwoStageBetween
from src.models.ddms.TwoStageWithin import TwoStageWithin
from src.utilities import mkdir_if_needed, str2bool
from tqdm import tqdm


def model_generator(model_classes, model_labels=None, dx=0.01, dt=0.01, T_dur=3):
    for m, model_class in enumerate(model_classes):
        if model_labels is not None:
            model_label = model_labels[m]
        else:
            model_label = None
        model_instance = model_class(dx=dx, dt=dt, T_dur=T_dur, label=model_label)
        model = model_instance.build_model()

        yield model


def main():

    model_classes = [
        TwoStageBetween,
        TwoStageWithin,
    ]
    model_names = [
        "TwoStageBetween",
        "TwoStageWithin",
    ]

    mkdir_if_needed(args.output_dir)

    # Read synthetic data
    # Note that every subject-model combination is considered a single subject now
    df_ddm = []

    for model in model_names:
        # Find all subject IDs (integer numbers) for which synthetic data from this model exists
        subjects = [
            int(filename.split("_")[-2])
            for filename in listdir(join(args.ddm_fitting_dir, model, "synthetic"))
            if filename.endswith("_synthetic.csv")
        ]
        for subject in subjects:
            filename = f"{model}_{subject}_synthetic.csv"
            df = pd.read_csv(
                join(args.ddm_fitting_dir, model, "synthetic", filename),
                index_col=0,
            )
            df["gen"] = model
            df["subject_id"] = f"{subject}-{model}"
            df_ddm.append(df)

    df_ddm = pd.concat(df_ddm).reset_index(drop=True)

    # We need to add additional columns needed by the models, because the synthetic data from each model only inlcudes variables needed by itself
    choices = pd.read_csv(args.data_file)
    choices_s = choices.loc[choices["subject_id"] == choices["subject_id"].unique()[0]]
    df_ddm = df_ddm.drop(["g0", "g1", "gm", "gp"], axis=1)
    df_ddm = df_ddm.merge(
        choices_s[["condition", "g0", "g1", "gm", "gp"]], on="condition", how="left"
    )

    subjects = df_ddm["subject_id"].unique()
    df_ddm["choice"] = df_ddm["pred_choice"]
    df_ddm["rt"] = df_ddm["pred_rt"]

    # %% Build models and combine with subject list to feed parallel processing
    def fit_predict_individual_model_wrap(input_args):
        subject, model = input_args
        result = fit_predict_individual_model(
            df=df_ddm,
            subject=subject,
            model=model,
            label=model.name,
            required_conditions=model.required_conditions,
            output_dir=args.output_dir,
            seed=args.seed,
            n_reps=args.n_reps,
            n_runs=args.n_runs,
            overwrite=args.overwrite,
            upper_alternative=1,
        )
        return result

    # %% Fit models in parallel
    if args.n_cores == 1:
        results = []
        for subject in tqdm(subjects):
            for model in model_generator(
                model_classes=model_classes, dx=args.dx, dt=args.dt, T_dur=args.T_dur
            ):
                result_s = fit_predict_individual_model(
                    df=df_ddm,
                    subject=subject,
                    model=model,
                    label=model.name,
                    required_conditions=model.required_conditions,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    n_reps=args.n_reps,
                    n_runs=args.n_runs,
                    overwrite=args.overwrite,
                    upper_alternative=1,
                )
                results.append(result_s)
    else:
        with Pool(args.n_cores) as pool:
            results = pool.map(
                fit_predict_individual_model_wrap,
                product(
                    subjects,
                    model_generator(
                        model_classes=model_classes,
                        dx=args.dx,
                        dt=args.dt,
                        T_dur=args.T_dur,
                    ),
                ),
            )

    # %% Save results (in addition to pandas.DataFrames created and saved for each subject by the fitting that ran in parallel)
    with open(join(args.output_dir, "ddm_recovery_results.pkl"), "wb") as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

    # Save DataFrames
    # Estimates
    est = (
        pd.concat([result[0] for result in results])
        .sort_values(["subject_id", "model"])
        .reset_index(drop=True)
    )
    est.to_csv(join(args.output_dir, "estimates.csv"))

    # Mean RT and accuracy predictions
    pred = (
        pd.concat([result[1] for result in results])
        .sort_values(["subject_id", "model"])
        .reset_index(drop=True)
    )
    pred.to_csv(join(args.output_dir, "predictions.csv"))

    # Synthetic model-simulated dataset
    # We need to be a bit more verbose here, to also read out the model variable
    synth = []
    for result in results:
        synth_s = result[2]
        model = result[0]["model"].values[0]
        synth_s["model"] = model
        synth.append(synth_s)
    synth = pd.concat(synth).sort_values(["subject_id", "model"]).reset_index(drop=True)
    synth.to_csv(join(args.output_dir, "synthetic.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddm-fitting-dir",
        help="Directory with results from ddm_fitting.py",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory",
    )
    parser.add_argument(
        "--data-file",
        help="Processed choices.csv file",
        default="data/processed/main/choices.csv",
    )

    parser.add_argument(
        "--n-cores", help="Number of CPU cores to use.", type=int, default=1
    )
    parser.add_argument(
        "--n-runs", help="Number of optimization runs per model.", type=int, default=1
    )
    parser.add_argument(
        "--n-reps",
        help="Number of simulated trials per observed trial.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--seed", help="Random seed for optimization.", type=int, default=1
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Toggle overwriting of existing result files.",
    )
    parser.add_argument("--dx", help="dx for pyDDM.", type=float, default=0.01)
    parser.add_argument("--dt", help="dt for pyDDM.", type=float, default=0.01)
    parser.add_argument("--T-dur", help="T_dur for pyDDM.", type=float, default=4)

    args = parser.parse_args()

    main()
