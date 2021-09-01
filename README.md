# Presentation order but not duration affects binary risky choice

Felix Molter & Peter N. C. Mohr

[Preprint at PsyArXiv]() · [Preregistration on OSF](https://osf.io/bth3m)

## Abstract

Risky choice behaviour often deviates from the predictions of normative models. The information search process has been suggested as a source of some reported "biases".
Specifically, gaze-dependent evidence accumulation models, where unfixated alternatives' signals are discounted, propose a mechanistic account of observed associations between eye movements, choices and response times, with longer fixated alternatives being chosen more frequently.
It remains debated, however, whether gaze causally influences the choice process, or rather reflects emerging preferences. Furthermore, other aspects the information search process, like the order in which information is inspected, can be confounded with gaze duration, complicating the identification of their causal influences.
In our preregistered study 179 participants made repeated incentivized choices between two sequentially presented risky gambles, allowing the experimental control of presentation duration, order, and format (i.e., alternative-wise or attribute-wise). Across presentation formats, we find evidence against an influence of presentation duration on choice. The order in which participants were shown stimulus information, however, causally affected choices, with alternatives shown last being chosen more frequently.
Notably, while gaze-dependent accumulation models generally capture effects of gaze duration, causal effects of stimulus order are only predicted by some models, identifying potential for future theory development. 

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with targets to run preprocessing, analyses and visualization
    ├── README.md          <- This README file.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump obtained from jsPsych.│
    │
    ├── models             <- Results from statistical analyses and behavioral modeling.
    │
    ├── reports            <- Reports and manuscripts associated with the project.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── build_conditions.py
    │   │   ├── make_data_overview.py
    │   │   └── process_choice_data.py
    │   │
    │   ├── models         <- Scripts to perform statistical analyses and behavioral modeling
    │   │   ├── choice_analyses.py      <- Runs behavioral analyses of choice data (GLM, BayesFactor t-tests, BEST)
    │   │   ├── ddm_analyses.py         <- Runs BayesFactor t-test, BEST of relative model fit between presentation formats
    │   │   ├── ddm_fitting.py          <- Fits the two DDMs to all trials, and separately for each presentation format
    │   │   ├── ddm_recovery.py         <- Runs parameter and model recoveries of the DDMs
    │   │   ├── model_comparison.py     <- Performs model comparison and selection
    │   │   ├── static_models_fitting.py  <- Performs parameter estimation and recovery of additional, exploratory softmax models
    │   │   ├── staticmodels.py         <- Contains model classes of additional exploratory softmax models
    │   │   ├── utils.py                <- Contains helper functions of static models
    │   │   └── ddms                    <- Contains DDM classes and functions for fitting
    │   │
    │   └── visualization  <- Contains scripts to create figures which are saved to reports/figures
    └── task               <- Contains jsPsych code of the behavioral task, including instructions, etc.


---

## Data description

The processed choice data file `data/processed/main/choices.csv` contains one line per trial with the following variables:

| Variable name            | Type  | Description                                                                                                                                                                 |
| ------------------------ | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `subject_id`             | int   | Subject ID                                                                                                                                                                  |
| `block`                  | int   | Experiment block (0 or 1)                                                                                                                                                   |
| `trial`                  | int   | Trial number                                                                                                                                                                |
| `condition`              | str   | Unique condition (trial) identifier matching `task/stimuli/conditions.csv`                                                                                                  |
| `choice`                 | int   | The participant's choice. In experimental trials 0 = Hp, 1 = Hm. In catch trials 0 = dominated, 1 = dominant                                                                |
| `rt`                     | float | Response time (in ms)                                                                                                                                                       |
| `p0`                     | float | Alternative 0 probability                                                                                                                                                   |
| `p1`                     | float | Alternative 1 probability                                                                                                                                                   |
| `m0`                     | float | Alternative 0 winning amount                                                                                                                                                |
| `m1`                     | float | Alternative 1 winning amount                                                                                                                                                |
| `label0`                 | str   | Text label for alternative 0                                                                                                                                                |
| `label1`                 | str   | Text label for alternative 1                                                                                                                                                |
| `higher_p`               | int   | Indicating which alternative has a higher probability                                                                                                                       |
| `higher_m`               | int   | Indicating which alternative has a higher winning amount                                                                                                                    |
| `presentation`           | str   | Presentation format: Either by `alternatives` or `attributes`                                                                                                               |
| `left_alternative`       | int   | Indicating which alternative is left                                                                                                                                        |
| `ev0`                    | float | Alternative 0 expected value                                                                                                                                                |
| `ev1`                    | float | Alternative 1 expected value                                                                                                                                                |
| `delta_ev`               | float | Difference in expected value (alt. 0 - alt. 1)                                                                                                                              |
| `delta_ev_z`             | float | z-scored Difference in expected value (alt. 0 - alt. 1)                                                                                                                     |
| `g0`                     | float | Relative presentation duration for alternative 0                                                                                                                            |
| `g1`                     | float | Relative presentation duration for alternative 1                                                                                                                            |
| `gp`                     | float | Relative presentation duration for probability attribute                                                                                                                    |
| `gm`                     | float | Relative presentation duration for amount attribute                                                                                                                         |
| `duration_favours`       | int   | Indicates which alternative is favoured by presentation duration                                                                                                            |
| `duration_favours_str`   | int   | Text label for `duration_favours`                                                                                                                                           |
| `last_stage_favours`     | int   | Indicates which alternative is favoured by the last presentation stage                                                                                                      |
| `last_stage_favours_str` | int   | Text label for `last_stage_favours`                                                                                                                                         |
| `choose_higher_p`        | int   | Indicates whether Hp was chosen                                                                                                                                             |
| `sequence`               | json  | Object containing information over presentation sequence                                                                                                                    |
| `webgazer_data`          | json  | Webcam-based eyetracking data recorded with [webgazer.js](https://webgazer.cs.brown.edu) with [jsPsych extension](https://www.jspsych.org/extensions/jspsych-ext-webgazer/) |
| `screen_width`           | int   | Participant screen width (in pixel)                                                                                                                                         |
| `screen_height`          | int   | Participant screen height (in pixel)                                                                                                                                        |


The file `data/processed/main/subject_summary.csv` contains a participant-level overview. It contains the following variables:

| Variable name        | Type  | Description                                       |
| -------------------- | ----- | ------------------------------------------------- |
| `subject_id`         | int   | Subject ID                                        |
| `run_id`             | int   | disregard, dataset ID on server                   |
| `exclude`            | bool  | Whether the participant is excluded               |
| `exclusion_reason`   | str   | Reason for which participant is excluded          |
| `gender`             | str   | Gender (Female, Male, Other)                      |
| `age`                | int   | Age in years                                      |
| `n_records`          | int   | Number of recorded trials                         |
| `n_choose_nan`       | int   | Number of timed-out responses                     |
| `n_choose_dominated` | int   | Number of choices of dominated alternatives       |
| `n_choose_higher_p`  | int   | Number of Hp choices                              |
| `chosen_trial`       | str   | ID of randomly determined trial for bonus payment |
| `lucky_number`       | float | Random number generated to determine bonus win    |
| `won_amount`         | float | Bonus amount (GBP) won by participant             |
| `rg_blind`           | bool  | Self-reported red-green blindness                 |
| `rg_difficult`       | bool  | Self-reported difficulty in distinguishing colors |
| `serious`            | bool  | Self-reported seriousness of participation        |
| `self_report`        | str   | Self-reported decision strategy                   |
| `comment`            | str   | Comments by the participant (optional)            |


---

## How to reproduce analyses

Analyses can be reproduced by installing required packages described in `requirements.txt`, installing the source package (`pip install -e .`) and using the included `Makefile`. The meta-target `all` includes all registered analyses and is automatically run with the `make` command.

The `src` folder additionally contains a copy of the [`myfuncs`](https://github.com/moltaire/myfuncs) package used for statistical tests and plotting setup.

| Target                    | Description                                                                                                                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `all`                     | **Runs all preprocessing, planned analyses and produces all figures.**                                                                                                                                               |
| `conditions`              | Re-constructs trial conditions (output in task/stimuli/conditions.csv) and makes the corresponding figure.                                                                                                           |
| `data`                    | Runs preprocessing pipeline. Creates choices.csv DataFrame and makes a summary of participants.                                                                                                                      |
| `choice_analyses`         | Runs planned analyses of choice data.                                                                                                                                                                                |
| `choice_analyses_figures` | Makes corresponding figures.                                                                                                                                                                                         |
| `ddm_fitting`             | Fits DDMs to data. Accepts optional arguments: `N_CORES` (`int`, number of cores to use), `N_RUNS_DDM` (`int`, number of estimation runs) , `OVERWRITE` (`True` or `False`, whether to overwrite estimation results) |
| `ddm_analyses`            | Runs analysis of differential relative DDM fit between presentation formats.                                                                                                                                         |
| `ddm_comparison`          | Runs model comparison and selection procedure.                                                                                                                                                                       |
| `ddm_comparison_figure`   | Makes the coresponding figure.                                                                                                                                                                                       |
| `ddm_recovery`            | Runs recovery of the DDMs based on empirical estimates.                                                                                                                                                              |
| `ddm_recovery_figures`    | Visualizes results from DDM recovery.                                                                                                                                                                                |

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
