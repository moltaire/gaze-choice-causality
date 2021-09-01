.PHONY: all conditions data choice_analyses choice_analyses_figures ddm_fitting  ddm_analyses ddm_prediction_figure ddm_recovery ddm_recovery_figures static_model_fitting static_model_comparison static_model_comparison_figure 

# Default arguments
TASK_VERSION = main
N_CORES = 8
N_RUNS = 100
N_RUNS_DDM = 1
OPTMETHOD = differential_evolution
OVERWRITE = False
CFLAGS = -c -g -D $(N_CORES) -D $(N_RUNS) $(N_RUNS_DDM) -D $(OPTMETHOD) -D $(OVERWRITE) -D $(TASK_VERSION)

# Meta target to make ALL
all : conditions hypotheses data choice_analyses choice_analyses_figures ddm_fitting ddm_analyses ddm_recovery ddm_recovery_figures

###############################################
# 0. Create choice problems and make a figure # 
###############################################
conditions : task/stimuli/conditions.csv reports/figures/conditions.pdf reports/figures/conditions.png

task/stimuli/conditions.csv : src/data/build_conditions.py
	mkdir -p task/stimuli
	python3 src/data/build_conditions.py --output-dir task/stimuli/

reports/figures/conditions.pdf reports/figures/conditions.png &: task/stimuli/conditions.csv src/visualization/conditions_figure.py
	mkdir -p reports/figures
	python3 src/visualization/conditions_figure.py --conditions-file task/stimuli/conditions.csv --output-dir reports/figures

#########################
# 0.1 Hypotheses figure # 
#########################
hypotheses : reports/figures/hypotheses.pdf reports/figures/hypotheses.png

reports/figures/hypotheses.pdf reports/figures/hypotheses.png &: src/visualization/hypotheses_figure.py
	mkdir -p reports/figures
	python3 src/visualization/hypotheses_figure.py --output-dir reports/figures

######################
# 1. Data processing #
######################
data : data/processed/$(TASK_VERSION)/subject_summary.csv data/processed/$(TASK_VERSION)/choices.csv

# 1.1 Data summary
data/processed/$(TASK_VERSION)/subject_summary.csv : src/data/make_data_overview.py
	mkdir -p data/processed
	python3 src/data/make_data_overview.py --input-path data/raw/$(TASK_VERSION) --output-path data/processed/$(TASK_VERSION)

# 1.2 Choice data
data/processed/$(TASK_VERSION)/choices.csv : src/data/process_choice_data.py data/processed/$(TASK_VERSION)/subject_summary.csv
	mkdir -p data/processed
	python3 src/data/process_choice_data.py --input-path data/raw/$(TASK_VERSION) --output-path data/processed/$(TASK_VERSION) --subject-summary data/processed/$(TASK_VERSION)/subject_summary.csv

##############################
# 2. Analyses of choice data #
##############################
choice_analyses : models/$(TASK_VERSION)/choice_analyses/choice_probability_table.csv models/$(TASK_VERSION)/choice_analyses/glm_summary.csv models/$(TASK_VERSION)/choice_analyses/glm_traceplot.png models/$(TASK_VERSION)/choice_analyses/glm_idata.nc models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_summary.csv models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_data.csv models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_traceplot.png models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_idata.nc models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_summary.csv models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_data.csv models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_traceplot.png models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_idata.nc models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_summary.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_data.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_traceplot.png models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_idata.nc models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_summary.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_data.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_traceplot.png models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_idata.nc

models/$(TASK_VERSION)/choice_analyses/choice_probability_table.csv models/$(TASK_VERSION)/choice_analyses/glm_summary.csv models/$(TASK_VERSION)/choice_analyses/glm_traceplot.png models/$(TASK_VERSION)/choice_analyses/glm_idata.nc models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_summary.csv models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_data.csv models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_traceplot.png models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_idata.nc models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_summary.csv models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_data.csv models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_traceplot.png models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_idata.nc models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_summary.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_data.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_traceplot.png models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_idata.nc models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_summary.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_data.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_traceplot.png models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_idata.nc &: src/models/choice_analyses.py data/processed/$(TASK_VERSION)/choices.csv
	mkdir -p models/$(TASK_VERSION)/choice_analyses
	python3 src/models/choice_analyses.py --choice-file data/processed/$(TASK_VERSION)/choices.csv --output-dir models/$(TASK_VERSION)/choice_analyses

# 2.1 Choice analyses figures
choice_analyses_figures : reports/figures/$(TASK_VERSION)/choice_analyses_psychometrics_$(TASK_VERSION).pdf reports/figures/$(TASK_VERSION)/choice_analyses_psychometrics_$(TASK_VERSION).png reports/figures/$(TASK_VERSION)/choice_analyses_individual_changes_$(TASK_VERSION).pdf reports/figures/$(TASK_VERSION)/choice_analyses_individual_changes_$(TASK_VERSION).png 

reports/figures/$(TASK_VERSION)/choice_analyses_psychometrics_$(TASK_VERSION).pdf reports/figures/$(TASK_VERSION)/choice_analyses_psychometrics_$(TASK_VERSION).png reports/figures/$(TASK_VERSION)/choice_analyses_individual_changes_$(TASK_VERSION).pdf reports/figures/$(TASK_VERSION)/choice_analyses_individual_changes_$(TASK_VERSION).png &: src/visualization/choice_analyses_figures.py data/processed/$(TASK_VERSION)/choices.csv models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_summary.csv models/$(TASK_VERSION)/choice_analyses/best_duration_alternatives_data.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_summary.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_alternatives_data.csv models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_summary.csv models/$(TASK_VERSION)/choice_analyses/best_duration_attributes_data.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_summary.csv models/$(TASK_VERSION)/choice_analyses/best_sequence_attributes_data.csv
	mkdir -p reports/figures
	python3 src/visualization/choice_analyses_figures.py --data-file data/processed/$(TASK_VERSION)/choices.csv --choice-analyses-dir models/$(TASK_VERSION)/choice_analyses --output-dir reports/figures/$(TASK_VERSION) --label $(TASK_VERSION)

###################
# 3. DDM modeling #
###################
ddm_fitting: models/$(TASK_VERSION)/ddm_fitting/estimates.csv models/$(TASK_VERSION)/ddm_fitting/predictions.csv models/$(TASK_VERSION)/ddm_fitting/synthetic.csv models/$(TASK_VERSION)/ddm_fitting_by-presentation/estimates.csv models/$(TASK_VERSION)/ddm_fitting_by-presentation/predictions.csv

# 3.1.1 Parameter estimation, all trials
models/$(TASK_VERSION)/ddm_fitting/ddm_fitting_results.pkl models/$(TASK_VERSION)/ddm_fitting/estimates.csv models/$(TASK_VERSION)/ddm_fitting/predictions.csv models/$(TASK_VERSION)/ddm_fitting/synthetic.csv &: src/models/ddm_fitting.py src/models/ddms/fitting.py src/models/ddms/TwoStageBetween.py src/models/ddms/TwoStageWithin.py src/models/ddms/TwoStageMixture.py src/models/ddms/agent.py data/processed/$(TASK_VERSION)/choices.csv
	mkdir -p models/$(TASK_VERSION)/ddm_fitting
	python3 src/models/ddm_fitting.py --data-file data/processed/$(TASK_VERSION)/choices.csv --output-dir models/$(TASK_VERSION)/ddm_fitting --overwrite $(OVERWRITE) --n-cores $(N_CORES) --n-runs $(N_RUNS_DDM) 

# 3.1.2 Parameter estimation, separately for presentation formats
models/$(TASK_VERSION)/ddm_fitting_by-presentation/estimates.csv models/$(TASK_VERSION)/ddm_fitting_by-presentation/predictions.csv &: src/models/ddm_fitting.py data/processed/$(TASK_VERSION)/choices.csv
	mkdir -p models/$(TASK_VERSION)/ddm_fitting_by-presentation
	python3 src/models/ddm_fitting.py --n-cores $(N_CORES) --data-file data/processed/$(TASK_VERSION)/choices.csv --n-runs $(N_RUNS_DDM) --output-dir models/$(TASK_VERSION)/ddm_fitting_by-presentation --split-by-presentation --overwrite $(OVERWRITE)


# 3.2 DDM comparison
models/$(TASK_VERSION)/ddm_comparison/bic_summary.csv models/$(TASK_VERSION)/ddm_comparison/best_model_counts.csv models/$(TASK_VERSION)/ddm_comparison/bms_result.pkl &: src/models/model_comparison.py models/$(TASK_VERSION)/ddm_fitting/estimates.csv
	mkdir -p models/$(TASK_VERSION)/ddm_comparison
	python3 src/models/model_comparison.py --estimates-file models/$(TASK_VERSION)/ddm_fitting/estimates.csv  --output-dir models/$(TASK_VERSION)/ddm_comparison

# 3.3 DDM comparison figure
reports/figures/$(TASK_VERSION)/ddm_comparison_$(TASK_VERSION).pdf reports/figures/$(TASK_VERSION)/ddm_comparison_$(TASK_VERSION).png &: src/visualization/model_comparison_figure.py models/$(TASK_VERSION)/ddm_comparison/bic_summary.csv models/$(TASK_VERSION)/ddm_comparison/best_model_counts.csv models/$(TASK_VERSION)/ddm_comparison/bms_result.pkl
	mkdir -p reports/figures
	python3 src/visualization/model_comparison_figure.py --estimates-file models/$(TASK_VERSION)/ddm_fitting/estimates.csv --best-model-counts-file models/$(TASK_VERSION)/ddm_comparison/best_model_counts.csv --bms-result-file models/$(TASK_VERSION)/ddm_comparison/bms_result.pkl --output-dir reports/figures/$(TASK_VERSION) --label $(TASK_VERSION) --filename ddm_comparison

ddm_comparison_figure : reports/figures/$(TASK_VERSION)/ddm_comparison_$(TASK_VERSION).pdf reports/figures/$(TASK_VERSION)/ddm_comparison_$(TASK_VERSION).png

# 3.4 DDM recovery
ddm_recovery: models/$(TASK_VERSION)/ddm_recovery/ddm_recovery_results.pkl models/$(TASK_VERSION)/ddm_recovery/estimates.csv models/$(TASK_VERSION)/ddm_recovery/predictions.csv models/$(TASK_VERSION)/ddm_recovery/synthetic.csv

models/$(TASK_VERSION)/ddm_recovery/ddm_recovery_results.pkl models/$(TASK_VERSION)/ddm_recovery/estimates.csv models/$(TASK_VERSION)/ddm_recovery/predictions.csv models/$(TASK_VERSION)/ddm_recovery/synthetic.csv &: src/models/ddm_recovery.py models/$(TASK_VERSION)/ddm_fitting/ddm_fitting_results.pkl
	mkdir -p models/$(TASK_VERSION)/ddm_recovery
	python3 src/models/ddm_recovery.py --ddm-fitting-dir models/$(TASK_VERSION)/ddm_fitting --output-dir models/$(TASK_VERSION)/ddm_recovery --data-file data/processed/$(TASK_VERSION)/choices.csv --overwrite $(OVERWRITE) --n-cores $(N_CORES) --n-runs $(N_RUNS_DDM)

# 3.5 DDM Recovery Figures
ddm_recovery_figures: models/$(TASK_VERSION)/ddm_recovery/ddm_recovery_results.pkl models/$(TASK_VERSION)/ddm_recovery/estimates.csv models/$(TASK_VERSION)/ddm_recovery/predictions.csv models/$(TASK_VERSION)/ddm_recovery/synthetic.csv src/visualization/ddm_recovery_figures.py
	python3 src/visualization/ddm_recovery_figures.py --estimates-file-recovery models/$(TASK_VERSION)/ddm_recovery/estimates.csv --estimates-file-fitting models/$(TASK_VERSION)/ddm_fitting/estimates.csv --output-dir reports/figures/$(TASK_VERSION)

# 3.6 DDM Analyses
ddm_analyses: models/$(TASK_VERSION)/ddm_analyses/ttestbf_relative-fit.csv models/$(TASK_VERSION)/ddm_analyses/ttestbf-directed_relative-fit.csv models/$(TASK_VERSION)/ddm_analyses/best_relative-fit_summary.csv

models/$(TASK_VERSION)/ddm_analyses/ttestbf_relative-fit.csv models/$(TASK_VERSION)/ddm_analyses/ttestbf-directed_relative-fit.csv models/$(TASK_VERSION)/ddm_analyses/best_relative-fit_summary.csv &: models/$(TASK_VERSION)/ddm_fitting_by-presentation/estimates.csv src/models/ddm_analyses.py
	mkdir -p models/$(TASK_VERSION)/ddm_analyses
	python3 src/models/ddm_analyses.py --estimates-file models/$(TASK_VERSION)/ddm_fitting_by-presentation/estimates.csv --output-dir models/$(TASK_VERSION)/ddm_analyses


# 3.7 DDM predictions figure
ddm_prediction_figure: reports/figures/ddm_predictions.pdf reports/figures/ddm_predictions.png

reports/figures/ddm_predictions.pdf reports/figures/ddm_predictions.png &: src/visualization/ddm_predictions_figure.py
	python3 src/visualization/ddm_predictions_figure.py --output-dir reports/figures

################################################################
# 4. Exploratory analyses with other static behavioural models #
################################################################
static_model_fitting : models/$(TASK_VERSION)/model_fitting/estimates/estimates.csv models/$(TASK_VERSION)/model_fitting/estimates/predictions.csv models/$(TASK_VERSION)/model_fitting_by-presentation/estimates/estimates.csv models/$(TASK_VERSION)/model_fitting_by-presentation/predictions/predictions.csv

# 4.1.1 Parameter estimation, all trials
models/$(TASK_VERSION)/model_fitting/estimates/estimates.csv models/$(TASK_VERSION)/model_fitting/estimates/predictions.csv &: src/models/static_model_fitting.py src/models/staticmodels.py data/processed/$(TASK_VERSION)/choices.csv
	mkdir -p models/$(TASK_VERSION)/model_fitting
	python3 src/models/static_model_fitting.py --verbose 10 --n-cores $(N_CORES) --data-file data/processed/$(TASK_VERSION)/choices.csv --n-runs $(N_RUNS) --overwrite $(OVERWRITE) --optmethod $(OPTMETHOD) --output-dir models/$(TASK_VERSION)/model_fitting

# 4.1.2 Parameter estimation, separately for presentation formats
models/$(TASK_VERSION)/model_fitting_by-presentation/estimates/estimates.csv models/$(TASK_VERSION)/model_fitting_by-presentation/predictions/predictions.csv &: src/models/static_model_fitting.py data/processed/$(TASK_VERSION)/choices.csv
	mkdir -p models/$(TASK_VERSION)/model_fitting_by-presentation
	python3 src/models/static_model_fitting.py --verbose 10 --n-cores $(N_CORES) --data-file data/processed/$(TASK_VERSION)/choices.csv --n-runs $(N_RUNS) --optmethod $(OPTMETHOD) --output-dir models/$(TASK_VERSION)/model_fitting_by-presentation --split-by-presentation --overwrite $(OVERWRITE)

# 4.2 Static model comparison
static_model_comparison : models/$(TASK_VERSION)/model_comparison/bic_summary.csv models/$(TASK_VERSION)/model_comparison/best_model_counts.csv models/$(TASK_VERSION)/model_comparison/bms_result.pkl

models/$(TASK_VERSION)/model_comparison/bic_summary.csv models/$(TASK_VERSION)/model_comparison/best_model_counts.csv models/$(TASK_VERSION)/model_comparison/bms_result.pkl &: src/models/model_comparison.py models/$(TASK_VERSION)/model_fitting/estimates/estimates.csv
	mkdir -p models/$(TASK_VERSION)/model_comparison
	python3 src/models/model_comparison.py --estimates-file models/$(TASK_VERSION)/model_fitting/estimates/estimates.csv  --output-dir models/$(TASK_VERSION)/model_comparison

# 4.3 Static model comparison figure
model_comparison_figure : reports/figures/$(TASK_VERSION)/model_comparison_$(TASK_VERSION).pdf reports/figures/$(TASK_VERSION)/model_comparison_$(TASK_VERSION).png

reports/figures/$(TASK_VERSION)/model_comparison_$(TASK_VERSION).pdf reports/figures/$(TASK_VERSION)/model_comparison_$(TASK_VERSION).png &: src/visualization/model_comparison_figure.py models/$(TASK_VERSION)/model_comparison/bic_summary.csv models/$(TASK_VERSION)/model_comparison/best_model_counts.csv models/$(TASK_VERSION)/model_comparison/bms_result.pkl
	mkdir -p reports/figures
	python3 src/visualization/model_comparison_figure.py --estimates-file models/$(TASK_VERSION)/model_fitting/estimates/estimates.csv --best-model-counts-file models/$(TASK_VERSION)/model_comparison/best_model_counts.csv --bms-result-file models/$(TASK_VERSION)/model_comparison/bms_result.pkl --output-dir reports/figures/$(TASK_VERSION) --label $(TASK_VERSION)
