import argparse
import pickle
from itertools import product
from os.path import join

import pandas as pd
from multiprocess import Pool
from src.models.ddms.fitting import fit_predict_individual_model
from src.models.ddms.TwoStageBetween import TwoStageBetween
from src.models.ddms.TwoStageWithin import TwoStageWithin
from src.models.ddms.TwoStageBetween_altwise import TwoStageBetween_altwise
from src.models.ddms.TwoStageWithin_attwise import TwoStageWithin_attwise
from src.utilities import mkdir_if_needed, str2bool
from tqdm import tqdm


def model_generator(presentation, dx=0.01, dt=0.01, T_dur=3):
    if presentation == "all":
        model_classes = [TwoStageBetween, TwoStageWithin]
    elif presentation == "alternatives":
        model_classes = [TwoStageWithin, TwoStageBetween_altwise]
    elif presentation == "attributes":
        model_classes = [TwoStageWithin_attwise, TwoStageBetween]
    for model_class in model_classes:
        model_instance = model_class(dx=dx, dt=dt, T_dur=T_dur)
        model = model_instance.build_model()

        yield model


def main():

    mkdir_if_needed(args.output_dir)

    df_ddm = pd.read_csv(args.data_file)
    df_ddm["rt"] /= 1000
    df_ddm[["m0", "m1"]] /= 10
    if args.split_by_presentation:
        df_ddm["subject_id"] = (
            df_ddm["subject_id"].astype(str) + "-" + df_ddm["presentation"]
        )
    subjects = df_ddm["subject_id"].unique()

    # %% Build models and combine with subject list to feed parallel processing
    def fit_predict_individual_model_wrap(input_args):
        subject, model_name = input_args
        if args.split_by_presentation:
            presentation = subject.split("-")[1]
        else:
            presentation = "all"
        if presentation == "all":
            model_class = {
                "TwoStageWithin": TwoStageWithin,
                "TwoStageBetween": TwoStageBetween,
            }[model_name]
        elif presentation == "attributes":
            model_class = {
                "TwoStageWithin": TwoStageWithin_attwise,
                "TwoStageBetween": TwoStageBetween,
            }[model_name]
        elif presentation == "alternatives":
            model_class = {
                "TwoStageWithin": TwoStageWithin,
                "TwoStageBetween": TwoStageBetween_altwise,
            }[model_name]
        model_instance = model_class(dx=args.dx, dt=args.dt, T_dur=args.T_dur)
        model = model_instance.build_model()

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
            upper_alternative=0,
        )
        return result

    # %% Fit models in parallel
    if args.n_cores == 1:
        results = []
        for subject in tqdm(subjects):
            if args.split_by_presentation:
                presentation = subject.split("-")[1]
            else:
                presentation = "all"
            for model in model_generator(
                presentation=presentation, dx=args.dx, dt=args.dt, T_dur=args.T_dur
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
                    upper_alternative=0,
                )
                results.append(result_s)
    else:
        with Pool(args.n_cores) as pool:
            if args.split_by_presentation:
                model_names = ["TwoStageWithin", "TwoStageBetween"]
            else:
                model_names = ["TwoStageWithin", "TwoStageBetween"]
            results = pool.map(
                fit_predict_individual_model_wrap,
                product(
                    subjects,
                    model_names,
                ),
            )

    # %% Save results (in addition to pandas.DataFrames created and saved for each subject by the fitting that ran in parallel)
    with open(join(args.output_dir, "ddm_fitting_results.pkl"), "wb") as output:
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
        "--data-file",
        help="Path to processed DDM formatted csv file",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory",
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

    parser.add_argument(
        "--split-by-presentation",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Toggle separate fitting for by-attribute and by-alternative trials.",
    )

    parser.add_argument("--dx", help="dx for pyDDM.", type=float, default=0.01)
    parser.add_argument("--dt", help="dt for pyDDM.", type=float, default=0.01)
    parser.add_argument("--T-dur", help="T_dur for pyDDM.", type=float, default=4)

    args = parser.parse_args()

    main()
