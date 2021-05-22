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
* Target datasets for analysis are in comma-separated format (.txt or .csv)
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

***
## Code Orientation
The base code for AutoMLPipe-BC is organized into a series of scripts designed to best optimize the parallelization of a given analysis. These loosely correspond with the pipeline schematic above.

* Phase 1: Exploratory Analysis (Code: ExploratoryAnalysisMain.py and ExploratoryAnalysisJob.py)
  * Conducts an initial exploratory analysis of all target datasets to be analyzed and compared
  * Conducts basic data cleaning
  * Conducts k-fold cross validation (CV) partitioning to generate k training and k testing datasets

* Phase 2: Data Preprocessing (Code: DataPreprocessingMain.py and DataPreprocessingJob.py)



***
## Jupyter Notebook
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
# Repository Orientation
Included in this repository is the following:
* The ML pipeline jupyter notebook, used to run the analysis - 'Supervised_Classification_ML_Pipeline.ipynb'
* An example/test dataset taken from the UCI repository - 'hcc-data_example.txt'
* A python script used by part 1 of the notebook - 'data_processing_methods.py'
* A python script used by part 2 of the notebook - 'feature_selection_methods.py'
* A python script used by part 3 of the notebook - 'modeling_methods.py'
* A folder containing python code for the ExSTraCS ML algorithm - 'exstracs_2.0.2.1_noclassmutate_lynch'
* A schematic summarizing the ML analysis pipeline - 'ML pipeline schematic2.png'

***
# Notebook Organization
## Part 1: Exploratory analysis, data cleaning, and creating n-fold CV partitioned datasets
- Instances missing a class value are excluded
- The user can indicate other columns that should be excluded from the analysis
- The user can turn on/off the option to apply standard scaling to the data prior to CV partitioning or imputation
    - We use no scaling by default. This is because most methods should work properly without it, and in applying the model downstream, it is difficult to properly scale new data so that models may be re-applied later.
    - ANN modeling is sensitive to feature scaling, thus without it, performance not be as good. However this is only one of many challenges in getting ANN to perform well.
- The user can turn on/off the option to impute missing values following CV partitioning
- The user can turn on/off the option for the code to automatically attempt to discriminate nominal from ordinal features
- The user can choose the number of CV partitions as well as the strategy for CV partitioning (i.e.  random (R), stratified (S), and matched (M)
- CV training and testing datasets are saved as .txt files so that the same partitions may be analyzed external to this code

## Part 2: Feature selection
- The user can turn on/off the option to filter out the lowest scoring features in the data (i.e. to conduct not just feature importance evaluation but feature selection)
- Feature importance evaluation and feature selection are conducted within each respective CV training partition
- The pipeline reports feature importance estimates via two feature selection algorithms:
    - Mutual Information: Proficient at detecting univariate associations
    - MultiSURF: Proficient at detecting univariate associations, 2-way epistatic interactions, and heterogeneous associations

- When selected by the user, feature selection conservatively keeps any feature identified as 'potentially relevant' (i.e. score > 0) by either algorithm
- Since MultiSURF scales quadratically with the number of training instances, there is an option to utilize a random subset of instances when running this algorithm to save computational time.

## Part 3: Machine learning modeling
- Seven ML modeling algorithms have been implemented in this pipeline:
    - Logistic Regression (scikit learn)
    - Decision Tree (scikit learn)
    - Random Forest (scikit learn)
    - Naïve Bayes (scikit learn)
    - XGBoost (separate python package)
    - LightGBM (separate python package)
    - SVM (scikit learn)
    - ANN (scikit learn)
    - ExSTraCS (v2.0.2.1) - a Learning Classifier System (LCS) algorithm manually configured to run in this notebook
- User can select any subset of these methods to run
- ML modeling is conducted within each respective CV training partition on the respective feature subset selected within the given CV partition
- ML modeling begins with a hyperparameter sweep conducted with a grid search of hard coded run parameter options (user can edit as needed)
- Balanced accuracy is applied as the evaluation metric for the hyperparameter sweep

## Part 4: ML feature importance vizualization
Performs normalization and transformation of feature importances scores for all algorithms and generates our proposed 'compound feature importance plots'.
