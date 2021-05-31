# causality

Visual gaze is associated with decision making, such that alternatives that are looked at longer are more likely to be chosen. The causal direction of this association, however, is the subject of ongoing debates, with increasing evidence for a causal effect of gaze on choice. Here, we test multiple facets of this causal effect in an incentive-compatible binary risky choice task, where we externally manipulate presentation duration and sequence of choice alternatives and their attributes.

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

## How to reproduce analyses

Analyses can be reproduced by installing required packages described in `requirements.txt`, installing the source package (`pip install -e .`) and using the included `Makefile`. The meta-target `all` includes all registered analyses and is automatically run with the `make` command.

| Target                    | Description                                                                                                                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `all`                     | **Runs all preprocessing, planned analyses and produces all figures.**                                                                                                                                                   |
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
