# AutoMLPipe-BC Summary
AutoMLPipe-BC is an automated, rigorous, and largely scikit-learn based machine learning (ML) analysis pipeline for binary classification. Adopts current best practices to avoid bias, optimize performance, ensure replicatability, capture complex associations (e.g. interactions and heterogeneity), and enhance interpretability. Includes (1) exploratory analysis, (2) data cleaning, (3) partitioning, (4) scaling, (5) imputation, (6) filter-based feature selection, (7) collective feature selection, (8) modeling with 'Optuna' hyperparameter optimization across 13 implemented ML algorithms (including three rule-based machine learning algorithms: ExSTraCS, XCS, and eLCS), (9) testing evaluations with 16 classification metrics, model feature importance estimation, (10) automatically saves all results, models, and publication-ready plots (including proposed composite feature importance plots), (11) non-parametric statistical comparisons across ML algorithms and analyzed datasets, and (12) automatically generated PDF summary reports.

# Overview
This AutoML tool empowers anyone with a basic understanding of python to easily run a comprehensive and customizable machine learning analysis. Unlike most other AutoML tools, AutoMLPipe-BC was designed as a framework to rigorously apply and compare a variety of ML modeling algorithms and collectively learn from them as opposed to simply identifying a best performing model and/or attempting to evolutionarily optimize the analysis pipeline itself. Instead, its design focused on automating (1) application of best practices in data science and ML for binary classification, (2) avoiding potential sources of bias (e.g. by conducting data transformations, imputation, and feature selection within distinct CV partitions), (3) providing transparency in the modeling and evaluation of models, (4) the detection and characterization of complex patterns of association (e.g. interactions and heterogeneity), (5) generation of publication-ready plots/figures, and (6) generation of a PDF summary report for quick interpretation. Overall, the goal of this pipeline is to provide an interpretable framework to learn from the data as well as the strengths and weaknesses of the ML algorithms or as a baseline to compare other AutoML strategies.

The following 13 ML modeling algorithms are currently included as options: 1. Naive Bayes (NB), 2. Logistic Regression (LR), 3. Decision Tree (DT), 4. Random Forest (RF), 5. Gradient Boosting (GB), 6. XGBoost (XGB), 7. LGBoost (LGB), 8. Support Vector Machine (SVM), 9. Artificial Neural Network (ANN), 10. K-Nearest Neighbors (k-NN), 11. Eductional Learning Classifier System (eLCS), 12. 'X' Classifier System (XCS), and 13. Extended Supervised Tracking and Classifying System (ExSTraCS). Classification-relevant hyperparameter values and ranges have been included for the (Optuna-driven) automated hyperparameter sweep.

This pipeline does NOT: (1) conduct feature engineering, or feature construction, (2) conduct feature encoding (e.g. apply one-hot-encoding to categorical features, or numerically encode text-based feature values), (3) account for bias in data collection, or (4) conduct anything beyond simple data cleaning (i.e. it only removes instances with no class label, or where all features are missing). These elements should be conducted externally at the discretion of the user.

We do not claim that this is the best or only viable way to assemble an ML analysis pipeline for a given classification problem, nor that the included ML modeling algorithms are necessarily the best options for inclusion. Certainly, this pipeline could be expanded much further and adapted to different problems or goals. We welcome feedback and suggestions for improvement.

***
## Schematic of AutoMLPipe-BC
This schematic breaks the overall pipeline down into 4 generalized stages: (1) preprocessing and feature transformation, (2) feature importance evaluation and selection, (3) modeling, and (4) postprocessing.

![alttext](https://github.com/UrbsLab/AutoMLPipe-BC/blob/main/ML_pipe_schematic.png?raw=true)

***
## Implementation
AutoMLPipe-BC is coded in Python 3 relying heavily on pandas and scikit-learn as well as a variety of other python packages.

***
## Run Modes
This multi-phase pipeline has been set up in a way that it can be easily run in one of three ways:
* A series of scripts (not parallelized) running on a local PC from the command line.
* A series of scripts that are run as parallelized jobs within a Linux-based computing cluster (see https://github.com/UrbsLab/I2C2-Documentation for a description of the computing cluster for which this functionality was designed).
* As an editable Jupyter Notebook that can be run all at once utilizing the associated code from the aforementioned scripts.

***
## Suggested Uses
* To easily conduct a rigorous, customizable ML analysis of one or more datasets using one or more of the included ML algorithms.
* As an analysis framework to evaluate and compare existing or other new ML modeling approaches.
* As a standard (or negative control) with which to compare other AutoML tools and determine if the added computational effort of searching pipeline configurations is paying off.
* As the basis to create a new expanded, adapted, or modified AutoML tool.
* As an educational example of how to program many of the most commonly used ML analysis procedures, and generate a variety of standard and novel plots.

***
## Assumptions For Use (data and run preparation)
* 'Target' datasets for analysis are in comma-separated format (.txt or .csv)
* Missing data values should be empty or indicated with an 'NA'.
* Dataset includes a header with column names.
* Data columns include features, class label, and optionally instance (i.e. row) labels, or match labels (if matched cross validation will be used)
* Binary class values are encoded as 0 (e.g. negative), and 1 (positive) with respect to true positive, true negative, false positive, false negative metrics. PRC plots focus on classification of 'positives'.
* All feature values (both categorical and quantitative) are numerically encoded. Scikit-learn does not accept text-based values. However both instance_label and match_label values may be either numeric or text.
* One or more target datasets for analysis are put in the same folder. The path to this folder is a critical pipeline run parameter. If multiple datasets are being analyzed they must have the same class_label, and (if present) the same instance_label and match_label.
* SVM modeling should only be applied when data scaling is applied by the pipeline
* Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is applied by the pipeline.  Otherwise 'use_uniform_FI' should be True.

***
## Unique Characteristics (ordered by appearence in pipeline)
* Pipeline includes reliable default run parameters that can be adjusted for further customization.
* Easily compare ML performance between multiple target datasets (e.g. with different feature subsets)
* Easily conduct an exploratory analysis including: (1) basic dataset characteristics: data dimensions, feature stats, missing value counts, and class balance, (2) detection of categorical vs. quantiative features, (3) feature correlation (with heatmap), and (4) univariate analyses with Chi-Square (categorical features), or Mann-Whitney U-Test (quantitative features).
* Option to manually specify which features to treat as categorical vs. quantitative.
* Option to manually specify features in loaded dataset to ignore in analysis.
* Option to utilize 'matched' cross validation partitioning: Case/control pairs or groups that have been matched based on one or more covariates will be kept together within CV data partitions.
* Imputation is completed using mode imputation for categorical variables first, followed by MICE-based iterative imputation for quantitaive features.
* Data scaling, imputation, and feature selection are all conducted within respective CV partitions to prevent data leakage (i.e. testing data is not seen for any aspect of learning until final model evaluation).
* The scaling, imputation, and feature selection data transformations (based only on the training data) are saved (i.e. 'pickled') so that they can be applied in the same way to testing partitions, and in the future to any replication data.
* Collective feature selection is used: Both mutual information (proficient at detectin univariate associations) and MultiSURF (a Relief-based algorithm proficient at detecting both univariate and epistatic interactions) are run, and features are only removed from consideration if both algorithms fail to detect an informative signal (i.e. score > 0). This ensures that interacting features that may have no univariate association with class are not removed from the data prior to modeling.
* Automatically outputs average feature importance bar-plots from feature importance/feature selection phase.
* Since MultiSURF scales linearly with # of features and quadratically with # of instances, there is an option to select a random instance subset for MultiSURF scoring to reduce computational burden.
* Includes 3 rule-based machine learning algorithms: ExSTraCS, XCS, and eLCS (to run optionally). These 'learning classifier systems' have been demonstrated to be able to detect complex associations while providing human interpretable models in the form of IF:THEN rule-sets. The ExSTraCS algorithm was developed by our research group to specifically handle the challenges of scalability, noise, and detection of epistasis and genetic heterogeneity in biomedical data mining.  
* Utilizes the 'optuna' package to conduct automated Bayesian hyperparameter optimization during modeling (and optionally outputs plots summarizing the sweep).
* We have sought to specify a comprehensive range of relevant hyperparameter options for all included ML algorithms.
* All ML algorithms that have a build in strategy to gather model feature importance estimates use them by default (i.e. LR,DT,RF,XGB,LGB,GB,eLCS,XCS,ExSTraCS).
* All other algorithms (NB,SVM,ANN,k-NN) estimate feature importance using permutation feature importance.
* The pipeline includes the option to apply permutation feature importance estimation uniformly (i.e. for all algorithms) by setting the 'use_uniform_FI' parameter to 'True'.
* All models are evaluated, reporting 16 classification metrics: Accuracy, Balanced Accuracy, F1 Score, Sensitivity(Recall), Specificity, Precision (PPV), True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN), Negative Predictive Value (NPV), Likeliehood Ratio + (LR+), Likeliehood Ratio - (LR-), ROC AUC, PRC AUC, and PRC APS.
* All models are saved as 'pickle' files so that they can be loaded and reapplied in the future.
* Outputs ROC and PRC plots for each ML modeling algorithm displaying individual n-fold CV runs and average the average curve.
* Outputs boxplots for each classification metric comparing ML modeling performance (across n-fold CV).
* Outputs boxplots of feature importance estimation for each ML modeling algorithm (across n-fold CV).
* Outputs our proposed 'composite feature importance plots' to examine feature importance estimate consistency (or lack of consistency) across all ML models (i.e. all algorithms)
* Outputs summary ROC and PRC plots comparing average curves across all ML algorithms.
* Collects run-time information on each phase of the pipeline and for the training of each ML algorithm model.
* For each dataset, Kruskall-Wallis and subsequent pairwise Mann-Whitney U-Tests evaluates statistical significance of ML algorithm modeling performance differences for all metrics.
* The same statistical tests (Kruskall-Wallis and Mann-Whitney U-Test) are conducted comparing datasets using the best performing modeling algorithm (for a given metric and dataset).
* A formatted PDF report is automatically generated giving a snapshot of all key pipeline results.
* A script is included to apply all trained (and 'pickled') models to an external replication dataset to further evaluate model generalizability. This script (1) conducts an exploratory analysis of the new dataset, (2) uses the same scaling, imputation, and feature subsets determined from n-fold cv training, yielding 'n' versions of the replication dataset to be applied to the respective models, (3) applies and evaluates all models with these respective versions of the replication data, (4) outputs the same set of aforementioned boxplots, ROC, and PRC plots, and (5) automatically generates a new, formatted PDF report summarizing these applied results.

# Installation
To use AutoMLPipe-BC, download this GitHub repository to your local working directory.

***
## Prerequisites
To be able to run AutoMLPipe-BC you will need Python 3, Anaconda (recommended rather than individually installing all individual packages), and a handful of other Python packages that are not included within Anaconda. Anaconda is a distribution of Python and R programming languages for scientific computing, that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS. We recommend installing the most recent stable version of Anaconda (https://docs.anaconda.com/anaconda/install/) within your computing environment (make sure to install a version appropriate for your operating system). Anaconda also includes jupyter notebook.

As an alternative to installing Anaconda, you will need to install Python 3, as well as all python packages used by AutoMLPipe-BC, e.g. pandas, and numpy (not listed in detail here). At the time of development we had installed 'Anaconda3-2020.07-Linux-x86_64.sh', using Python 3.8.3.

In addition to the above you will also need to install the following packages in your computing environment: skrebate, xgboost, lightgbm, scikit-eLCS, scikit-XCS, scikit-ExSTraCS, optuna, plotly, orca, and kaleido.  Installation commands are given below (along with the version used at time of posting):

### Feature Selection Packages
* scikit-learn compatible version of ReBATE, a suite of Relief-based feature selection algorithms (v.0.7). There is currently a PyPi issue requiring that the newest version (i.e. 0.7) be explicitly installed.
```
pip install skrebate==0.7
```

### ML Modeling Packages
* XGboost (v.1.2.0)
```
pip install xgboost
```
* LightGBM (v.3.0.0)
```
pip install lightgbm
```
* scikit-learn compatible version of eLCS, an educational learning classifier system (v.1.2.2)
```
pip install scikit-eLCS
```
* scikit-learn compatible version of the learning classifier system XCS designed exclusively for supervised learning (v.1.0.6)
```
pip install scikit-XCS
```
* scikit-learn compatible version of the learning classifier system ExSTraCS (v.1.0.7)
```
pip install scikit-ExSTraCS
```

### Other Required Packages
* FPDF, a simple PDF generation for Python (v.1.7.2)
```
pip install fpdf
```
* Optuna, a hyperparameter optimization framework (v.2.0.0)
```
pip install optuna
```
Plotly, an open-source, interactive data visualization library. Used by optuna to generate hyperparameter sweep visualizations (v.4.9.0)
```
pip install plotly
```
Kaleido a package for static image export for web-based visualization (v.0.0.3.post1)
```
pip install kaleido
```

### Possibly needed package
Orca, a python library for task organization was previously needed by plotly, however Kaleido should be sufficient. If Kaleido alone is not working try installing the following:
```
pip install orca
```
If the above installation does not work, try:
```
conda install -c plotly plotly-orca
```
See https://pypi.org/project/plotly/ for details or updates for installing these plotly dependencies.

# Usage
Here we give an overview of the codebase and how to run AutoMLPipe-BC in different contexts.
***
## Code Orientation
The base code for AutoMLPipe-BC is organized into a series of script phases designed to best optimize the parallelization of a given analysis. These loosely correspond with the pipeline schematic above. These phases are designed to be run in order. Phases 1-7 make up the core automated pipeline, with Phase 7 and beyond being run optionally based on user needs. In general this pipeline will run more slowly when a larger number of 'target' dataset are being analyzed and when a larger number of CV 'folds' are requested.

* Phase 1: Exploratory Analysis
  * Conducts an initial exploratory analysis of all target datasets to be analyzed and compared
  * Conducts basic data cleaning
  * Conducts k-fold cross validation (CV) partitioning to generate k training and k testing datasets
  * [Code]: ExploratoryAnalysisMain.py and ExploratoryAnalysisJob.py
  * [Runtime]: Typically fast, with the exception of generating feature correlation heatmaps in datasets with a large number of features

* Phase 2: Data Preprocessing
  * Conducts feature transformations (i.e. data scaling) on all CV training datasets individually
  * Conducts imputation of missing data values (missing data is not allowed by most scikit-learn modeling packages) on all CV training datasets individually
  * Generates updated training and testing CV datasets
  * [Code]: DataPreprocessingMain.py and DataPreprocessingJob.py
  * [Runtime]: Typically fast, with the exception of imputing larger datasets with many missing values

* Phase 3: Feature Importance Evaluation
  * Conducts feature importance estimations on all CV training datasets individually
  * Generates updated training and testing CV datasets
  * [Code]: FeatureImportanceMain.py and FeatureImportanceJob.py
  * [Runtime]: Typically reasonably fast, takes more time to run MultiSURF as the number of training instances approaches the default for 'instance_subset', or this parameter set higher in larger datasets

* Phase 4: Feature Selection
  * Applies 'collective' feature selection within all CV training datasets individually
  * Features removed from a given training dataset are also removed from corresponding testing dataset
  * Generates updated training and testing CV datasets
  * [Code]: FeatureSelectionMain.py and FeatureSelectionJob.py
  * [Runtime]: Fast

* Phase 5: Machine Learning Modeling
  * Conducts hyperparameter sweep for all ML modeling algorithms individually on all CV training datasets
  * Conducts 'final' modeling for all ML algorithms individually on all CV training datasets using 'optimal' hyperparameters found in previous step
  * Calculates and saves all evaluation metrics for all 'final' models
  * [Code]: ModelMain.py and ModelJob.py
  * [Runtime]: Slowest phase, can be sped up by reducing the set of ML methods selected to run, or deactivating ML methods that run slowly on large datasets

* Phase 6: Statistics Summary
  * Combines all results to generate summary statistics files, generate results plots, and conduct non-parametric statistical significance analyses comparing ML model performance across CV runs
  * [Code]: StatsMain.py and StatsJob.py
  * [Runtime]: Moderately fast

* Phase 7: [Optional] Compare Datasets
  * NOTE: Only can be run if the AutoMLPipe-BC was run on more than dataset
  * Conducts non-parametric statistical significance analyses comparing separate original 'target' datasets analyzed by pipeline
  * [Code]: DataCompareMain.py and DataCompareJob.py
  * [Runtime]: Fast

* Phase 8: [Optional] Copy Key Files
  * Makes a copy of key results files and puts them in a folder called 'KeyFileCopy'
  * [Code]: KeyFileCopyMain.py and KeyFileCopyJob.py
  * [Runtime]: Fast

* Phase 9: [Optional] Generate PDF Training Summary Report
  * Generates a pre-formatted PDF including all pipeline run parameters, basic dataset information, and key exploratory analyses, ML modeling results, statistical comparisons, and runtime.
  * [Code]: PDF_ReportTrainMain.py and PDF_ReportTrainJob.py
  * [Runtime]: Moderately fast

* Phase 10: [Optional] Apply Models to Replication Data
  * Applies all previously trained models for a single 'target' dataset to one or more new 'replication' dataset that has all features found in the original 'target' datasets
  * Conducts exploratory analysis on new 'replication' dataset(s)
  * Applies scaling, imputation, and feature selection (unique to each CV partition from model training) to new 'replication' dataset(s) in preparation for model application
  * Evaluates performance of all models the prepared 'replication' dataset(s)
  * Generates summary statistics files, results plots, and conducts non-parametric statistical significance analyses comparing ML model performance across replications CV data transformations
  * NOTE: feature importance evaluation and 'target' dataset statistical comparisons are irrelevant to this phase
  * [Code]: ApplyModelMain.py and ApplyModelJob.py
  * [Runtime]: Moderately fast

* Phase 11: [Optional] Generate PDF 'Apply Replication' Summary Report
  * Generates a pre-formatted PDF including all pipeline run parameters, basic dataset information, and key exploratory analyses, ML modeling results, and statistics.
  * [Code]: PDF_ReportApplyMain.py and PDF_ReportApplyJob.py
  * [Runtime]: Moderately fast

***
## Run From Jupyter Notebook
Here we detail how to run AutoMLPipe-BC within the provided jupyter notebook. This is likely the easiest approach for those newer to python, or for those who wish to explore, or easily test the code. However depending on the size of the target dataset(s) and the pipeline settings, this can take a long time to run locally. The included notebook is set up to run on included example datasets (HCC data taken from the UCI repository). NOTE: The user will still need to update the local folder/file paths in this notebook to be able for it to correctly run.
* First, ensure all prerequisite packages are installed in your environment and dataset assumptions (above) are satisfied.
* Open jupyter notebook (https://jupyter.readthedocs.io/en/latest/running.html). We recommend opening the 'anaconda prompt' which comes with your anaconda installation.  Once opened, type the command 'jupyter notebook' which will open as a webpage. Navigate to your working directory and open the included jupyter notebook file: 'AutoMLPipe-BC-Notebook.ipynb'.
* Towards the beginning of the notebook in the section 'Mandatory Parameters to Update', make sure to revise your dataset-specific information (especially your local path information for files/folders)
* If you have a replication dataset to analyze, scroll down to the section 'Apply Models to Replication Data' and revise the dataset-specific information in 'Mandatory Parameters to Update', just below.
* Check any other notebook cells specifying 'Run Parameters' for any of the pipeline phases and update these settings as needed.
* Now that the code as been adapted to your desired dataset/analysis, click 'Kernel' on the Jupyter notebook GUI, and select 'Restart & Run All' to run the script.  
* To run the included example dataset with the pre-specified notebook run parameters, should only take a matter of minutes.
* However it may take several hours or more to run this notebook in other contexts. Runtime is primarily increased by selecting additional ML modeling algorithms, picking a larger number of CV partitions, increasing 'n_trials' and 'timeout' which controls hyperparameter optimization, or increasing 'instance_subset' which controls the maximum number of instances used to run Relief-based feature selection (note: these algorithms scale quadratically with number of training instances).

***
## Run From Command Line (Local or Cluster Parallelization)
The primary way to run AutoMLPipe-BC is via the command line, one phase at a time (running the next phase only after the previous one has completed). As indicated above, each phase can run locally (not parallelized) or parallelized using a Linux based computing cluster. With a little tweaking of the 'Main' scripts this code could also be parallelized with cloud computing. We welcome help in extending the code for that purpose.

### Local Run Example
Below we give an example of the set of all commands needed to run AutoMLPipe-BC in it's entirety using mostly default run parameters. In this example we specify instance and class label run parameters to emphasize the importance setting these values correctly.
```
python ExploratoryAnalysisMain.py --data-path /mydatapath/TestData --output-path /myoutputpath/output --experiment-name hcc_test --instance-label InstanceID --class-label Class --run-parallel False

python DataPreprocessingMain.py --output-path /myoutputpath/output --experiment-name hcc_test --run-parallel False

python FeatureImportanceMain.py --output-path /myoutputpath/output --experiment-name hcc_test --run-parallel False

python FeatureSelectionMain.py --output-path /myoutputpath/output --experiment-name hcc_test --run-parallel False

python ModelMain.py --output-path /myoutputpath/output --experiment-name hcc_test --run-parallel False

python StatsMain.py --output-path /myoutputpath/output --experiment-name hcc_test --run-parallel False

python DataCompareMain.py --output-path /myoutputpath/output --experiment-name hcc_test --run-parallel False

python KeyFileCopyMain.py --data-path /mydatapath/TestData --output-path /myoutputpath/output --experiment-name hcc_test --run-parallel False

python PDF_ReportTrainMain.py --output-path /myoutputpath/output --experiment-name hcc_test --run-parallel False

python ApplyModelMain.py --output-path /myoutputpath/output --experiment-name hcc_test --rep-data-path /myrepdatapath/TestRep  --data-path /mydatapath/TestData/hcc-data_example.csv --run-parallel False

python PDF_ReportApplyMain.py --output-path /myoutputpath/output --experiment-name hcc_test --rep-data-path /myrepdatapath/TestRep  --data-path /mydatapath/TestData/hcc-data_example.csv --run-parallel False
```

### Computing Cluster Run (Parallelized) Example
Below we give the same set of AutoMLPipe-BC run command, however in each, the run parameter --run-parallel is left to its default value of 'True'.
```
python ExploratoryAnalysisMain.py --data-path /mydatapath/TestData --output-path /myoutputpath/output --experiment-name hcc_test --instance-label InstanceID --class-label Class

python DataPreprocessingMain.py --output-path /myoutputpath/output --experiment-name hcc_test

python FeatureImportanceMain.py --output-path /myoutputpath/output --experiment-name hcc_test

python FeatureSelectionMain.py --output-path /myoutputpath/output --experiment-name hcc_test

python ModelMain.py --output-path /myoutputpath/output --experiment-name hcc_test

python StatsMain.py --output-path /myoutputpath/output --experiment-name hcc_test

python DataCompareMain.py --output-path /myoutputpath/output --experiment-name hcc_test

python KeyFileCopyMain.py --data-path /mydatapath/TestData --output-path /myoutputpath/output --experiment-name hcc_test

python PDF_ReportTrainMain.py --output-path /myoutputpath/output --experiment-name hcc_test

python ApplyModelMain.py --output-path /myoutputpath/output --experiment-name hcc_test --rep-data-path /myrepdatapath/TestRep  --data-path /mydatapath/TestData/hcc-data_example.csv

python PDF_ReportApplyMain.py --output-path /myoutputpath/output --experiment-name hcc_test --rep-data-path /myrepdatapath/TestRep  --data-path /mydatapath/TestData/hcc-data_example.csv
```

### Checking Phase Completion
After running any of Phases 1-6 a 'phase-complete' file is automatically generated for each job run locally or in parallel.  Users can confirm that all jobs for that phase have been completed by running the phase command again, this time with the argument '-c'. Any incomplete jobs will be listed, or an indication of successful completion will be returned.

For example, after running ModelMain.py, the following command can be given to check whether all jobs have been completed.
```
python ModelMain.py --output-path /myoutputpath/output --experiment-name hcc_test -c
```

## Phase Details (Run Parameters and Additional Examples)
Here we review the run parameters available for each of the 11 phases and provide some additional examples of each. Run parameters that are necessary to set are marked as 'MANDATORY' under 'default'. The additional examples illustrate how to flexibly adapt AutoMLPipe-BC to user needs.

### Phase 1: Exploratory Analysis

| Argument &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Description | Default |
|:-------- |:---------------------  | ----------- |
| --data-path | path to directory containing datasets | MANDATORY |
| --out-path | path to output directory | MANDATORY |
| --exp-name | name of experiment output folder (no spaces) | MANDATORY |
| --class-label | outcome label of all datasets | Class |
| --inst-label | instance label of all datasets (if present) | None |
| --fi | path to .csv file with feature labels to be ignored in analysis | None |
| --cf | path to .csv file with feature labels specified to be treated as categorical | None |
| --cv | number of CV partitions | 10 |
| --part | 'S', or 'R', or 'M', for stratified, random, or matched, respectively | S |
| --match-label | only applies when M selected for partition-method; indicates column with matched instance ids | None |
| --cat-cutoff | number of unique values after which a variable is considered to be quantitative vs categorical | 10 |
| --sig | significance cutoff used throughout pipeline | 0.05 |
| --export-ea | run and export basic exploratory analysis files, i.e. unique value counts, missingness counts, class balance barplot | True |
| --export-fc | run and export feature correlation analysis (yields correlation heatmap) | True |
| --export-up | export univariate analysis plots (note: univariate analysis still output by default) | False |
| --rand-state | "Dont Panic" - sets a specific random seed for reproducible results | 42 |
| --run-parallel | if run parallel | True |
| --queue | specify name of parallel computing queue (uses our research groups queue by default) | i2c2_normal |
| --res-mem | reserved memory for the job (in Gigabytes) | 4 |
| --max-mem | maximum memory before the job is automatically terminated | 15 |
| -c | Boolean: Specify whether to check for existence of all output files | Stores False |



* Phase 1: Exploratory Analysis

* Phase 2: Data Preprocessing

* Phase 3: Feature Importance Evaluation

* Phase 4: Feature Selection

* Phase 5: Machine Learning Modeling

* Phase 6: Statistics Summary

* Phase 7: [Optional] Compare Datasets

* Phase 8: [Optional] Copy Key Files

* Phase 9: [Optional] Generate PDF Training Summary Report

* Phase 10: [Optional] Apply Models to Replication Data

* Phase 11: [Optional] Generate PDF 'Apply Replication' Summary Report
