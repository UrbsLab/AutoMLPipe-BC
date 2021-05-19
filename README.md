# Summary
An automated, rigorous, and largely scikit-learn based machine learning (ML) analysis pipeline for binary classification. Adopts current best practices to avoid bias, optimize performance, ensure replicatability, capture complex associations (e.g. interactions and heterogeneity), and enhance interpretability. Includes (1) exploratory analysis, (2) data cleaning, (3) partitioning, (4) scaling, (5) imputation, (6) filter-based feature selection, (7) collective feature selection, (8) modeling with 'optuna' hyperparameter optimization across 13 implemented ML algorithms (including three rule-based machine learning algorithms: ExSTraCS, XCS, and eLCS), (9) testing evaluations with 16 classification metrics, model feature importance estimation, (10) automatically saves all results, models, and publication-ready plots (including proposed composite feature importance plots), (11) non-parametric statistical comparisons across ML algorithms and analyzed datasets, and (12) automatically generated PDF summary reports.

# Overview
This AutoML tool empowers anyone with a basic understanding of python to easily run a comprehensive and customizable machine learning analysis. Unlike most other AutoML tools, AutoMLPipe-BC was designed as a framework to rigorously apply and compare a variety of ML modeling algorithms and collectively learn from them as opposed to simply identifying a best performing model and/or series of pipeline steps. It's design focused on automating (1) application of best practices in data science and ML for binary classification, (2) avoiding potential sources of bias (e.g. by conducting data transformations, imputation, and feature selection withing distinct CV partitions), (3) providing transparency in the modeling and evaluation of models, (4) the detection and characterization of complex patterns of association (e.g. interactions and heterogeneity), (5) publication-ready plots/figures, and (6) PDF summary report generation for quick evaluation.

The following 13 ML modeling algorithms are currently included as options: 1. Logistic Regression (LR), 2. Naive Bayes, 3. Decision Tree (DT), 4. Random Forest (RF), 5. XGBoost (XGB), 6. LGBoost (LGB), 7. Support Vector Machine (SVM), 8. Artificial Neural Network (ANN), 9. K-Nearest Neighbors (k-NN), 10. Gradient Boosting (GB), 11. Eductional Learning Classifier System (eLCS), 12. X Classifier System (XCS), and 13. Extended Supervised Tracking and Classifying System (ExSTraCS).

This pipeline does NOT: (1) conduct feature engineering, or feature construction, (2) conduct feature encoding (e.g. apply one-hot-encoding to categorical features, or numerically encode text-based feature values), (3) account for bias in data collection, or (4) conduct anything beyond simple data cleaning (i.e. it only removes instances with no class label, or where all features are missing). These elements should be conducted externally at the discression of the user.

***
## Schematic of AutoMLPipe-BC
![alttext](https://github.com/UrbsLab/AutoMLPipe-BC/blob/main/ML_pipe_schematic.png?raw=true)

***
## Implementation
AutoMLPipe-BC is coded in Python 3 relying heavily on pandas and scikit-learn as well as a variety of other python packages. 

***
## Modes of Use
This multi-phase pipeline has been set up in a way that it can be easily run in one of three ways:
* A series of scripts that are run as parallelized jobs within a linux-based computing cluster (see https://github.com/UrbsLab/I2C2-Documentation for a description of the computing cluster for which this functionality was designed).
* A series of scripts (not parallelized) running on a local PC from the command line.
* As an editable Jupyter Notebook that can be run all at once utilizing the associated code from the scripts above.

***
## Suggested Use
* To easily conduct a rigorous, customizable ML analysis of one or more datasets using a variety of ML algorithms. 
* As the basis to create a new expanded, adapted, or modified ML analysis pipeline.
* As an educational example of how to program many of the most commontly used ML analysis proceedures, and generate a variety of standard and novel plots.

***
## Assumptions For Use (Data and Run Preparation)
* Target datasets for analysis are in comma-separated format (.txt or .csv)
* Data columns include features, class label, and optionally instance (i.e. row) labels, or match labels (if matched cross validation will be used)
* Binary class values are encoded as 0 (e.g. negative), and 1 (positive) with respect to true positive, true negative, false positive, false negative metrics. PRC plots focus on classification of 'positives'.
* All feature values (both categorical and quantitative) are numerically encoded. Scikit-learn does not accept text-based values. However both instance_label and match_label values may be either numeric or text.
* One or more target datasets for analysis are put in the same folder. The path to this folder is a critical pipeline run parameter. If multiple datasets are being analyzed they must have the same class_label, and (if present) the same instance_label and match_label.
* SVM modeling should only be applied when data scaling is applied by the pipeline
* Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is applied by the pipeline.  Otherwise 'use_uniform_FI' should be True.

***
## Unique Features
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

# Usage

***
# Prerequisites for Use
## Environment Requirements
In order to run this pipeline as a Jupyter Notebook you must have the proper environment set up on your computer. Python 3 as well as a number of Python packages are required.  Most of these requirements are satisfied by installing the most recent version of anaconda (https://docs.anaconda.com/anaconda/install/). We used Anaconda3 with python version 3.7.7 during this pipeline development. In addition to the packages included in anaconda, the following packages will need to be installed separately (or possibly updated, if you have an older version installed). We recommend installing them within the 'anaconda prompt' that installs with anaconda:

* scikit-rebate (To install: pip install skrebate)
* xgboost (To install: pip install xgboost)
* lightgbm (To install: pip install lightgbm)
* optuna (To install: pip install optuna)

Additionally, while currently commented out in the file (modeling_methods.py) if you want the optuna hypterparameter sweep figures to appear within the jupyter notebook (via the command 'fig.show()' ) you will need to run the following installation commands.  This should only be required if you edit the python file to uncomment this line for any or all of the ML modeling algorithms.

* pip install -U plotly>=4.0.0
* conda install -c plotly plotly-orca

Lastly, in order to include the stand-alone algorithm 'ExSTraCS' we needed to call this from the command line within this Jupyter Notebook.  As a result, the part of this notebook running ExSTraCS will only run properly if the path to the working directory used to run this notebook includes no spaces.  In other words if your path includes a folder called 'My Folder' vs. 'My_Folder' you will likely get a run error for ExSTraCS (at least on a Windows machine). Thus, make sure to check that wherever you are running this notebook from, that the entire path to the working directory does note include any spaces.

***
## Dataset Requirements
This notebook loads a single dataset to be run through the entire pipeline. Here we summarize the requirements for this dataset:
* Ensure your data is in a single file: (If you have a pre-partitioned training/testing dataset, you should combine them into a single dataset before running this notebook)
* Any dataset specific cleaning, feature transformation, or feature engineering that may be needed in order to maximize ML performance should be conducted by the user separately or added to the beginning of this notebook.
* The dataset should be in tab-delimited .txt format to run this notebook (as is).  Commented-out code to load a comma separated file (.csv) and excel file (.xlsx) is included in the notebook as an alternative.
* Missing data values should be empty or indicated with an 'NA'.
* Dataset includes a header with column names. This should include a column for the binary class label and (optionally) a column for the instance ID, as well as columns for other 'features', e.g. independend variables.
* The class labels should be 0 for the major class (i.e. the most frequent class), and 1 for the minor class.  This is important for generation of the precision/recall curve (PRC) plots.
* This dataset is saved in the working directory containing the jupyter notebook file, and all other files in this repository.
* All variables in the dataset have been numerically encoded (otherwise additional data preprocessing may be needed)

***
# Usage
* First, ensure all of the environment and dataset requirments above are satisfied.
* Next, save this repository to the desired 'working directory' on your pc (make sure there are no 'spaces' in the path to this directory!)
* Open the jupyter notebook file (https://jupyter.readthedocs.io/en/latest/running.html). We found that the most reliable way to do this and ensure your run environment is correct is to open the 'anaconda prompt' which comes with your anaconda installation.  Once opened type the command 'jupyter notebook'.  Then navigate to your working directory and click on the notebook file: 'Supervised_Classification_ML_Pipeline.ipynb'.
* Towards the beginning of the notebook in the section 'Set Dataset Pipeline Variables (Mandatory)', make sure to update your dataset-specific information (e.g. dataset name, outcome label, and instance label (if applicable)
* In the next notebook cell, 'Set Other Pipeline Variables (Optional)', you can 'optionally' set other analysis pipeline settings (e.g. number of cross validation partitions, what algorithms to include, etc)
* Next, in the next cell, 'ML Modeling Hyperparamters (Optional)' you can adjust the hyperparameters of all ML modeling algorithms to be explored in the respective hyperparameter sweeps. You can also adjust the overall optuna settings controlling the basics of how the hyperparameter sweeps are conducted. Note that 'adding' any other hyperparameters that have not been included in this section for a given ML modeler, will require updates to the code in the file 'modeling_methods.py'. We believe that we have included all critical run parameters for each ML algorithm so this should not be an issue for most users.
* Now that the code as been adapted to your desired dataset/analysis, click 'Kernel' on the Jupyter notebook GUI, and select 'Restart & Run All' to run the script.  
* Note that due to all that is involved in running this notebook, it may take several hours or more to complete running all analyses. Runtime can be shortened by picking a subset of ML algorithms, picking a smaller number of CV partitions, reducing 'n_trials' and 'hype_cv' which controls hyperparameter optimization, or reducing 'instanceSubset' which controls the maximum number of instances used to run Relief-based feature selection (note: these algorithms scale quadratically with number of training instances).

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
