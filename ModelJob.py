import sys
import time
import random
import pandas as pd
import numpy as np
import os
import pickle
import copy
import math
#Model Packages:
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from skeLCS import eLCS
from skXCS import XCS
from skExSTraCS import ExSTraCS
#Evalutation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import metrics
#Other packages
from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import optuna #hyperparameter optimization

def job(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric):
    #Get hyperparameter grid
    param_grid = hyperparameters(random_state,do_lcs_sweep,nu,iterations,N)[algorithm]

    runModel(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric,param_grid)


def runModel(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric,param_grid):
    job_start_time = time.time()
    random.seed(random_state)
    np.random.seed(random_state)

    trainX,trainY,testX,testY = dataPrep(train_file_path,instance_label,class_label,test_file_path)

    #Run model
    abbrev = {'naive_bayes':'NB','logistic_regression':'LR','decision_tree':'DT','random_forest':'RF','gradient_boosting':'GB','XGB':'XGB','LGB':'LGB','SVM':'SVM','ANN':'ANN','k_neighbors':'KN','eLCS':'eLCS','XCS':'XCS','ExSTraCS':'ExSTraCS'}
    if algorithm == 'naive_bayes':
        ret = run_NB_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'logistic_regression':
        ret = run_LR_full(trainX,trainY,testX,testY, random_state, cvCount,param_grid,n_trials,timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'decision_tree':
        ret = run_DT_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,export_hyper_sweep_plots,full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'random_forest':
        ret = run_RF_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'gradient_boosting':
        ret = run_GB_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'XGB':
        ret = run_XGB_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,export_hyper_sweep_plots,full_path,training_subsample,use_uniform_FI,primary_metric)
    elif algorithm == 'LGB':
        ret = run_LGB_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'SVM':
        ret = run_SVM_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,training_subsample,use_uniform_FI,primary_metric)
    elif algorithm == 'ANN':
        ret = run_ANN_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,training_subsample,use_uniform_FI,primary_metric)
    elif algorithm == 'k_neighbors':
        ret = run_KN_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,export_hyper_sweep_plots, full_path,training_subsample,use_uniform_FI,primary_metric)
    elif algorithm == 'eLCS':
        ret = run_eLCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'XCS':
        ret = run_XCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,export_hyper_sweep_plots, full_path,use_uniform_FI,primary_metric)
    elif algorithm == 'ExSTraCS':
        ret = run_ExSTraCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,export_hyper_sweep_plots, full_path,filter_poor_features,instance_label,class_label,use_uniform_FI,primary_metric)

    pickle.dump(ret, open(full_path + '/training/' + abbrev[algorithm] + '_CV_' + str(cvCount) + "_metrics", 'wb'))

    saveRuntime(full_path,job_start_time,abbrev,algorithm,cvCount)

    # Print completion
    print(full_path.split('/')[-1] + " CV" + str(cvCount) + " phase 5 "+abbrev[algorithm]+" training complete")
    experiment_path = '/'.join(full_path.split('/')[:-1])
    job_file = open(experiment_path + '/jobsCompleted/job_model_' + full_path.split('/')[-1] + '_' + str(cvCount) +'_' +abbrev[algorithm]+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def dataPrep(train_file_path,instance_label,class_label,test_file_path):
    #Get X and Y
    train = pd.read_csv(train_file_path)
    if instance_label != 'None':
        train = train.drop(instance_label,axis=1)
    trainX = train.drop(class_label,axis=1).values
    trainY = train[class_label].values

    test = pd.read_csv(test_file_path)
    if instance_label != 'None':
        test = test.drop(instance_label,axis=1)
    testX = test.drop(class_label,axis=1).values
    testY = test[class_label].values
    return trainX,trainY,testX,testY

def saveRuntime(full_path,job_start_time,abbrev,algorithm,cvCount):
    # Save Runtime
    runtime_file = open(full_path + '/runtime/runtime_'+abbrev[algorithm]+'_CV'+str(cvCount)+'.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

#Hyperparameter optimization with optuna
def hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric):
    cv = StratifiedKFold(n_splits=hype_cv, shuffle=True, random_state=randSeed)
    model = clone(est).set_params(**params)
    #Flexibly handle whether random seed is given as 'random_seed' or 'seed' - scikit learn uses 'random_seed'
    for a in ['random_state','seed']:
        if hasattr(model,a):
            setattr(model,a,randSeed)
    performance = np.mean(cross_val_score(model,x_train,y_train,cv=cv,scoring=scoring_metric ))
    return performance

#Naive Bayes #############################################################################################################################
def run_NB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    #No hyperparameters to optimize.

    #Train model using 'best' hyperparameters - Uses default 3-fold internal CV (training/validation splits)
    clf = GaussianNB()
    model = clf.fit(x_train, y_train)

    #Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/NB_'+str(i), 'wb'))

    #Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    #Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    #Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#Logistic Regression ###################################################################################################################
def objective_LR(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'penalty' : trial.suggest_categorical('penalty',param_grid['penalty']),
			  'dual' : trial.suggest_categorical('dual', param_grid['dual']),
			  'C' : trial.suggest_loguniform('C', param_grid['C'][0], param_grid['C'][1]),
			  'solver' : trial.suggest_categorical('solver',param_grid['solver']),
			  'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight']),
			  'max_iter' : trial.suggest_loguniform('max_iter',param_grid['max_iter'][0], param_grid['max_iter'][1]),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_LR_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    est = LogisticRegression()

    if not isSingle:
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_LR(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/LR_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        est = LogisticRegression()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/LR_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/LR_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/LR_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1] #reversed list orders
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = pow(math.e,model.coef_[0])

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#Decision Tree #####################################################################################################################################
def objective_DT(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'criterion' : trial.suggest_categorical('criterion',param_grid['criterion']),
                'splitter' : trial.suggest_categorical('splitter', param_grid['splitter']),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'max_features' : trial.suggest_categorical('max_features',param_grid['max_features']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight']),
                'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_DT_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    est = tree.DecisionTreeClassifier()

    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_DT(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/DT_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        est = tree.DecisionTreeClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/DT_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/DT_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/DT_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#Random Forest ######################################################################################################################################
def objective_RF(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'criterion' : trial.suggest_categorical('criterion',param_grid['criterion']),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'max_features' : trial.suggest_categorical('max_features',param_grid['max_features']),
                'bootstrap' : trial.suggest_categorical('bootstrap',param_grid['bootstrap']),
                'oob_score' : trial.suggest_categorical('oob_score',param_grid['oob_score']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight']),
                'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_RF_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    est = RandomForestClassifier()

    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_RF(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/RF_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        est = RandomForestClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/RF_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/RF_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/RF_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#Gradient Boosting Classifier #####################################################################################################################
def objective_GB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'loss': trial.suggest_categorical('loss', param_grid['loss']),
                'learning_rate': trial.suggest_loguniform('learning_rate', param_grid['learning_rate'][0],param_grid['learning_rate'][1]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0],param_grid['min_samples_leaf'][1]),
                'min_samples_split': trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0],param_grid['min_samples_split'][1]),
                'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_GB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    # Run Hyperparameter sweep
    est = GradientBoostingClassifier()

    if not isSingle:
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_GB(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/GB_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        est = GradientBoostingClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/GB_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/GB_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/GB_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_

    return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi

#XGBoost ###################################################################################################################################################
def objective_XGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    posInst = sum(y_train)
    negInst = len(y_train) - posInst
    classWeight = negInst/float(posInst)
    params = {'booster' : trial.suggest_categorical('booster',param_grid['booster']),
                'objective' : trial.suggest_categorical('objective',param_grid['objective']),
                'verbosity' : trial.suggest_categorical('verbosity',param_grid['verbosity']),
                'reg_lambda' : trial.suggest_loguniform('reg_lambda', param_grid['reg_lambda'][0], param_grid['reg_lambda'][1]),
                'alpha' : trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
                'eta' : trial.suggest_loguniform('eta', param_grid['eta'][0], param_grid['eta'][1]),
                'gamma' : trial.suggest_loguniform('gamma', param_grid['gamma'][0], param_grid['gamma'][1]),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'grow_policy' : trial.suggest_categorical('grow_policy',param_grid['grow_policy']),
                'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'subsample' : trial.suggest_uniform('subsample', param_grid['subsample'][0], param_grid['subsample'][1]),
                'min_child_weight' : trial.suggest_loguniform('min_child_weight', param_grid['min_child_weight'][0], param_grid['min_child_weight'][1]),
                'colsample_bytree' : trial.suggest_uniform('colsample_bytree', param_grid['colsample_bytree'][0], param_grid['colsample_bytree'][1]),
                'scale_pos_weight' : trial.suggest_categorical('scale_pos_weight', [1.0, classWeight]),
                'nthread' : trial.suggest_categorical('nthread',param_grid['nthread']),
                'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_XGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For XGB training sample reduced to '+str(x_train.shape[0])+' instances')

    est = xgb.XGBClassifier()

    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_XGB(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/XGB_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        est = xgb.XGBClassifier()
        clf = clone(est).set_params(**best_trial.params)
        export_best_params(full_path + '/training/XGB_bestparams' + str(i) + '.csv', best_trial.params)
        setattr(clf, 'random_state', randSeed)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/XGB_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/XGB_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#LGBoost #########################################################################################################################################
def objective_LGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    posInst = sum(y_train)
    negInst = len(y_train) - posInst
    classWeight = negInst / float(posInst)
    params = {'objective': trial.suggest_categorical('objective', param_grid['objective']),
              'metric': trial.suggest_categorical('metric', param_grid['metric']),
              'verbosity': trial.suggest_categorical('verbosity', param_grid['verbosity']),
              'boosting_type': trial.suggest_categorical('boosting_type', param_grid['boosting_type']),
              'num_leaves': trial.suggest_int('num_leaves', param_grid['num_leaves'][0], param_grid['num_leaves'][1]),
              'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
              'lambda_l1': trial.suggest_loguniform('lambda_l1', param_grid['lambda_l1'][0],param_grid['lambda_l1'][1]),
              'lambda_l2': trial.suggest_loguniform('lambda_l2', param_grid['lambda_l2'][0],param_grid['lambda_l2'][1]),
              'feature_fraction': trial.suggest_uniform('feature_fraction', param_grid['feature_fraction'][0],param_grid['feature_fraction'][1]),
              'bagging_fraction': trial.suggest_uniform('bagging_fraction', param_grid['bagging_fraction'][0],param_grid['bagging_fraction'][1]),
              'bagging_freq': trial.suggest_int('bagging_freq', param_grid['bagging_freq'][0],param_grid['bagging_freq'][1]),
              'min_child_samples': trial.suggest_int('min_child_samples', param_grid['min_child_samples'][0],param_grid['min_child_samples'][1]),
              'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0],param_grid['n_estimators'][1]),
              'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, classWeight]),
              'num_threads' : trial.suggest_categorical('num_threads',param_grid['num_threads']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_LGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    # Run Hyperparameter sweep
    est = lgb.LGBMClassifier()

    if not isSingle:
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_LGB(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/LGB_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        est = lgb.LGBMClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/LGB_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/LGB_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/LGB_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.feature_importances_

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#Support Vector Machines #####################################################################################################################################
def objective_SVM(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'kernel': trial.suggest_categorical('kernel', param_grid['kernel']),
              'C': trial.suggest_loguniform('C', param_grid['C'][0], param_grid['C'][1]),
              'gamma': trial.suggest_categorical('gamma', param_grid['gamma']),
              'degree': trial.suggest_int('degree', param_grid['degree'][0], param_grid['degree'][1]),
              'probability': trial.suggest_categorical('probability', param_grid['probability']),
              'class_weight': trial.suggest_categorical('class_weight', param_grid['class_weight']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_SVM_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For SVM, training sample reduced to '+str(x_train.shape[0])+' instances')

    # Run Hyperparameter sweep
    est = SVC()

    if not isSingle:
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_SVM(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/SVM_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        est = SVC()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/SVM_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/SVM_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/SVM_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates (SVM can only automatically obtain feature importance estimates (coef_) for linear kernel)
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#Artificial Neural Networks #######################################################################################################################
def objective_ANN(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'activation': trial.suggest_categorical('activation', param_grid['activation']),
              'learning_rate': trial.suggest_categorical('learning_rate', param_grid['learning_rate']),
              'momentum': trial.suggest_uniform('momentum', param_grid['momentum'][0], param_grid['momentum'][1]),
              'solver': trial.suggest_categorical('solver', param_grid['solver']),
              'batch_size': trial.suggest_categorical('batch_size', param_grid['batch_size']),
              'alpha': trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
              'max_iter': trial.suggest_categorical('max_iter', param_grid['max_iter']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}

    n_layers = trial.suggest_int('n_layers', param_grid['n_layers'][0], param_grid['n_layers'][1])
    layers = []
    for i in range(n_layers):
        layers.append(
            trial.suggest_int('n_units_l{}'.format(i), param_grid['layer_size'][0], param_grid['layer_size'][1]))
        params['hidden_layer_sizes'] = tuple(layers)

    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_ANN_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For ANN, training sample reduced to '+str(x_train.shape[0])+' instances')

    est = MLPClassifier()
    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_ANN(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/ANN_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Handle special parameter requirement for ANN
        layers = []
        for j in range(best_trial.params['n_layers']):
            layer_name = 'n_units_l' + str(j)
            layers.append(best_trial.params[layer_name])
            del best_trial.params[layer_name]

        best_trial.params['hidden_layer_sizes'] = tuple(layers)
        del best_trial.params['n_layers']

        # Train model using 'best' hyperparameters
        est = MLPClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/ANN_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/ANN_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/ANN_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#K-Neighbors Classifier ####################################################################################################################################
def objective_KN(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', param_grid['n_neighbors'][0], param_grid['n_neighbors'][1]),
        'weights': trial.suggest_categorical('weights', param_grid['weights']),
        'p': trial.suggest_int('p', param_grid['p'][0], param_grid['p'][1]),
        'metric': trial.suggest_categorical('metric', param_grid['metric'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_KN_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,training_subsample,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    if training_subsample > 0 and training_subsample < x_train.shape[0]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(training_subsample), random_state=randSeed)
        for train_index, _ in sss.split(x_train,y_train):
            x_train = x_train[train_index]
            y_train = y_train[train_index]
        print('For KN, training sample reduced to '+str(x_train.shape[0])+' instances')

    # Run Hyperparameter sweep
    est = KNeighborsClassifier()

    if not isSingle:
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_KN(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/KN_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        est = KNeighborsClassifier()
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/KN_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/KN_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/KN_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
    fi = results.importances_mean

    return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi

#eLCS (educational learning Classifier System) ##################################################################################################################################
def objective_eLCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
        'N': trial.suggest_categorical('N', param_grid['N']),
        'nu': trial.suggest_categorical('nu', param_grid['nu']),
        'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_eLCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    est = eLCS()
    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_eLCS(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/eLCS_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/eLCS_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/eLCS_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/eLCS_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.get_final_attribute_specificity_list()

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#XCS ('X' Learning classifier system) ############################################################################################################################################
def objective_XCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
        'N': trial.suggest_categorical('N', param_grid['N']),
        'nu': trial.suggest_categorical('nu', param_grid['nu']),
        'random_state' : trial.suggest_categorical('random_state',param_grid['random_state'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_XCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,use_uniform_FI,primary_metric):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    est = XCS()
    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_XCS(trial, est, x_train, y_train, randSeed, 3, param_grid, primary_metric),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/XCS_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path + '/training/XCS_bestparams' + str(i) + '.csv', best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/XCS_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/XCS_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.get_final_attribute_specificity_list()

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

#ExSTraCS (Extended supervised tracking and classifying system) #############################################################################
def objective_ExSTraCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
              'N': trial.suggest_categorical('N', param_grid['N']), 'nu': trial.suggest_categorical('nu', param_grid['nu']),
              'random_state' : trial.suggest_categorical('random_state',param_grid['random_state']),
              'expert_knowledge':param_grid['expert_knowledge'],
              'rule_compaction':trial.suggest_categorical('rule_compaction', param_grid['rule_compaction'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def get_FI_subset_ExSTraCS(full_path,i,instance_label,class_label,filter_poor_features):
    """ For ExSTraCS, gets the MultiSURF (or MI if MS not availabile) FI scores for the feature subset being analyzed here in modeling"""
    scores = [] #to be filled in, in filted dataset order.
    data_name = full_path.split('/')[-1]

    if os.path.exists(full_path + "/multisurf/pickledForPhase4/"):  # If MultiSURF was done previously:
        algorithmlabel = 'multisurf'
    elif os.path.exists(full_path + "/mutualinformation/pickledForPhase4/"):  # If MI was done previously and MS wasn't:
        algorithmlabel = 'mutualinformation'
    else:
        scores = []
        return scores

    if eval(filter_poor_features):
        #Load current data ordered_feature_names
        header = pd.read_csv(full_path+'/CVDatasets/'+data_name+'_CV_'+str(i)+'_Test.csv').columns.values.tolist()
        if instance_label != 'None':
            header.remove(instance_label)
        header.remove(class_label)

        #Load orignal dataset multisurf scores
        scoreInfo = full_path+ "/"+algorithmlabel+"/pickledForPhase4/"+str(i)
        file = open(scoreInfo, 'rb')
        rawData = pickle.load(file)
        file.close()
        scoreDict = rawData[1]

        #Generate filtered multisurf score list with same order as working datasets
        for each in header:
            scores.append(scoreDict[each])
    else:
        #Load orignal dataset multisurf scores
        scoreInfo = full_path+ "/"+algorithmlabel+"/pickledForPhase4/"+str(i)
        file = open(scoreInfo, 'rb')
        rawData = pickle.load(file)
        file.close()
        scores = rawData[0]

    return scores

def run_ExSTraCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path,filter_poor_features,instance_label,class_label,use_uniform_FI,primary_metric):
    #Grab feature importance weights from multiSURF, used by ExSTraCS
    scores = get_FI_subset_ExSTraCS(full_path,i,instance_label,class_label,filter_poor_features)
    param_grid['expert_knowledge'] = scores

    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1 and key != 'expert_knowledge':
            isSingle = False

    est = ExSTraCS()

    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_ExSTraCS(trial, est, x_train, y_train, randSeed, 3, param_grid,primary_metric), n_trials=n_trials, timeout=timeout,catch=(ValueError,))

        if eval(do_plot):
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/ExSTraCS_ParamOptimization_'+str(i)+'.png')

        print('Best trial:')
        best_trial = study.best_trial
        print('  Value: ', best_trial.value)
        print('  Params: ')
        for key, value in best_trial.params.items():
            print('    {}: {}'.format(key, value))

        # Train model using 'best' hyperparameters
        clf = est.set_params(**best_trial.params)
        export_best_params(full_path+'/training/ExSTraCS_bestparams'+str(i)+'.csv',best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            if key == 'expert_knowledge':
                params[key] = value
            else:
                params[key] = value[0]
        clf = est.set_params(**params)
        export_best_params(full_path + '/training/ExSTraCS_usedparams' + str(i) + '.csv', params)

    print(clf)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/ExSTraCS_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)
    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    if eval(use_uniform_FI):
        results = permutation_importance(model, x_train, y_train, n_repeats=10,random_state=randSeed, scoring=primary_metric)
        fi = results.importances_mean
    else:
        fi = clf.get_final_attribute_specificity_list()

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]


def export_best_params(file_name,param_grid):
    best_params_copy = param_grid
    for best in best_params_copy:
        best_params_copy[best] = [best_params_copy[best]]
    df = pd.DataFrame.from_dict(best_params_copy)
    df.to_csv(file_name, index=False)

def classEval(y_true, y_pred):
    # calculate and store evaluation metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    ac = accuracy_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)
    re = recall_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # calculate specificity
    if tn == 0 and fp == 0:
        sp = 0
    else:
        sp = tn / float(tn + fp)

    # calculate NPV
    if tn == 0 and fn == 0:
        npv = 0
    else:
        npv = tn/float(tn+fn)

    # calculate lrp
    if sp == 1:
        lrp = 0
    else:
        lrp = re/float(1-sp)

    # calculate lrm
    if sp == 0:
        lrm = 0
    else:
        lrm = (1-re)/float(sp)

    return [bac, ac, f1, re, sp, pr, tp, tn, fp, fn, npv, lrp, lrm]

def hyperparameters(random_state,do_lcs_sweep,nu,iterations,N):
    param_grid = {}
    #EDITABLE CODE###############################################################
    # Naive Bayes - Has no hyperparameters

    # Logistic Regression - can take a longer while in larger instance spaces
    param_grid_LR = {'penalty': ['l2', 'l1'],'C': [1e-5, 1e5],'dual': [True, False],
                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                     'class_weight': [None, 'balanced'],'max_iter': [10, 1000],
                     'random_state':[random_state]}

    # Decision Tree
    param_grid_DT = {'criterion': ['gini', 'entropy'],'splitter': ['best', 'random'],'max_depth': [1, 30],
                     'min_samples_split': [2, 50],'min_samples_leaf': [1, 50],'max_features': [None, 'auto', 'log2'],
                     'class_weight': [None, 'balanced'],
                     'random_state':[random_state]}

    # Random Forest
    param_grid_RF = {'n_estimators': [10, 1000],'criterion': ['gini', 'entropy'],'max_depth': [1, 30],
                     'min_samples_split': [2, 50],'min_samples_leaf': [1, 50],'max_features': [None, 'auto', 'log2'],
                     'bootstrap': [True],'oob_score': [False, True],'class_weight': [None, 'balanced'],
                     'random_state':[random_state]}
    # GB
    param_grid_GB = {'n_estimators': [10, 1000],'loss': ['deviance', 'exponential'], 'learning_rate': [.0001, 0.3], 'min_samples_leaf': [1, 50],
                     'min_samples_split': [2, 50], 'max_depth': [1, 30],'random_state':[random_state]}

    # XG Boost - not great for large instance spaces (limited completion). note: class weight balance is included as option internally
    param_grid_XGB = {'booster': ['gbtree'],'objective': ['binary:logistic'],'verbosity': [0],'reg_lambda': [1e-8, 1.0],
                      'alpha': [1e-8, 1.0],'eta': [1e-8, 1.0],'gamma': [1e-8, 1.0],'max_depth': [1, 30],
                      'grow_policy': ['depthwise', 'lossguide'],'n_estimators': [10, 1000],'min_samples_split': [2, 50],
                      'min_samples_leaf': [1, 50],'subsample': [0.5, 1.0],'min_child_weight': [0.1, 10],
                      'colsample_bytree': [0.1, 1.0],'nthread':[1],'random_state':[random_state]}

    # LG Boost - note: class weight balance is included as option internally (still takes a while on large instance spaces)
    param_grid_LGB = {'objective': ['binary'],'metric': ['binary_logloss'],'verbosity': [-1],'boosting_type': ['gbdt'],
                      'num_leaves': [2, 256],'max_depth': [1, 30],'lambda_l1': [1e-8, 10.0],'lambda_l2': [1e-8, 10.0],
                      'feature_fraction': [0.4, 1.0],'bagging_fraction': [0.4, 1.0],'bagging_freq': [1, 7],
                      'min_child_samples': [5, 100],'n_estimators': [10, 1000],'num_threads':[1],'random_state':[random_state]}

    # SVM - not approppriate for large instance spaces
    param_grid_SVM = {'kernel': ['linear', 'poly', 'rbf'],'C': [0.1, 1000],'gamma': ['scale'],'degree': [1, 6],
                      'probability': [True],'class_weight': [None, 'balanced'],'random_state':[random_state]}

    # ANN - bad for large instances spaces
    param_grid_ANN = {'n_layers': [1, 3],'layer_size': [1, 100],'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'],'momentum': [.1, .9],
                      'solver': ['sgd', 'adam'],'batch_size': ['auto'],'alpha': [0.0001, 0.05],'max_iter': [200],'random_state':[random_state]}

    # KN - not appropriate for large instance spaces
    param_grid_KN = {'n_neighbors': [1, 100], 'weights': ['uniform', 'distance'], 'p': [1, 5],
                     'metric': ['euclidean', 'minkowski']}

    if eval(do_lcs_sweep):
        # eLCS
        param_grid_eLCS = {'learning_iterations': [100000,200000,500000],'N': [1000,2000,5000],'nu': [1,10],
                           'random_state':[random_state]}
        # XCS
        param_grid_XCS = {'learning_iterations': [100000,200000,500000],'N': [1000,2000,5000],'nu': [1,10],
                          'random_state':[random_state]}
        # ExSTraCS
        param_grid_ExSTraCS = {'learning_iterations': [100000,200000,500000],'N': [1000,2000,5000],'nu': [1,10],
                               'random_state':[random_state],'rule_compaction':[None]}
    else:
        # eLCS
        param_grid_eLCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu], 'random_state': [random_state]}
        # XCS
        param_grid_XCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu], 'random_state': [random_state]}
        # ExSTraCS
        param_grid_ExSTraCS = {'learning_iterations': [iterations], 'N': [N], 'nu': [nu], 'random_state': [random_state], 'rule_compaction': ['None']} #'QRF'

    ####################################################################################################################
    param_grid['naive_bayes'] = {}
    param_grid['logistic_regression'] = param_grid_LR
    param_grid['decision_tree'] = param_grid_DT
    param_grid['random_forest'] = param_grid_RF
    param_grid['gradient_boosting'] = param_grid_GB
    param_grid['XGB'] = param_grid_XGB
    param_grid['LGB'] = param_grid_LGB
    param_grid['SVM'] = param_grid_SVM
    param_grid['ANN'] = param_grid_ANN
    param_grid['k_neighbors'] = param_grid_KN
    param_grid['eLCS'] = param_grid_eLCS
    param_grid['XCS'] = param_grid_XCS
    param_grid['ExSTraCS'] = param_grid_ExSTraCS
    return param_grid

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),sys.argv[8],sys.argv[9],sys.argv[10],int(sys.argv[11]),int(sys.argv[12]),sys.argv[13],sys.argv[14],int(sys.argv[15]),int(sys.argv[16]),int(sys.argv[17]),int(sys.argv[18]),sys.argv[19],sys.argv[20])
