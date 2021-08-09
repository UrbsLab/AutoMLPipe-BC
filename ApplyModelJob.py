"""
File: ApplyModelJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 10 of AutoMLPipe-BC - This 'Job' script is called by ApplyModelMain.py. It conducts exploratory analysis on the new replication dataset then
applies and evaluates all trained models on one or more previously unseen hold-out or replication study dataset(s). It also genertes new evaluation figure.
It does not deal with model feature importance estimation as this is a part of model training interpretation only.  This script is run once for each replication
dataset in rep_data_path.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import copy
import pandas as pd
import os
import csv
import sys
import ExploratoryAnalysisJob
import StatsJob
import ModelJob
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#Evalutation metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import metrics
from scipy import interp,stats
from statistics import mean,stdev

def job(datasetFilename,full_path,class_label,instance_label,categorical_cutoff,sig_cutoff,cv_partitions,scale_data,impute_data,do_LR,do_DT,do_RF,do_NB,do_XGB,do_LGB,do_SVM,do_ANN,do_ExSTraCS,do_eLCS,do_XCS,do_GB,do_KN,primary_metric,data_path,match_label,plot_ROC,plot_PRC,plot_metric_boxplots,export_feature_correlations,jupyterRun):
    train_name = full_path.split('/')[-1] #original training data name
    apply_name = datasetFilename.split('/')[-1].split('.')[0]
    #Load Replication Dataset
    repData = pd.read_csv(datasetFilename, na_values='NA', sep = ",")
    rep_feature_list = list(repData.columns.values)
    rep_feature_list.remove(class_label)
    if match_label != 'None':
        rep_feature_list.remove(match_label)
    if instance_label != 'None':
        rep_feature_list.remove(instance_label)
    #Load original training dataset (could include 'match label')
    trainData = pd.read_csv(data_path, na_values='NA', sep = ",")
    all_train_feature_list = list(trainData.columns.values)
    all_train_feature_list.remove(class_label)
    if match_label != 'None':
        all_train_feature_list.remove(match_label)
    if instance_label != 'None':
        all_train_feature_list.remove(instance_label)
    #Confirm that all features in original training data appear in replication datasets
    if(set(all_train_feature_list).issubset(set(rep_feature_list))):
        pass
    else:
        print('Error: One or more features in training dataset did not appear in replication dataset!')
        fail = 5/0
    #Grab and order replication data columns to match training data columns
    repData = repData[trainData.columns]
    #Create Folder hierarchy
    if not os.path.exists(full_path+"/applymodel/"+apply_name+'/'+'exploratory'):
        os.mkdir(full_path+"/applymodel/"+apply_name+'/'+'exploratory')
    if not os.path.exists(full_path+"/applymodel/"+apply_name+'/'+'training'):
        os.mkdir(full_path+"/applymodel/"+apply_name+'/'+'training')
    if not os.path.exists(full_path+"/applymodel/"+apply_name+'/'+'training'+'/'+'results'):
        os.mkdir(full_path+"/applymodel/"+apply_name+'/'+'training'+'/'+'results')
    #Load previously identified list of categorical variables and create an index list to identify respective columns
    file = open(full_path + '/exploratory/categorical_variables','rb')
    categorical_variables = pickle.load(file)
    #ExploratoryAnalysis - basic data cleaning
    repData = ExploratoryAnalysisJob.removeRowsColumns(repData,class_label,[])
    #Export basic exploratory analysis files
    ExploratoryAnalysisJob.describeData(repData,full_path,'applymodel/'+apply_name) #Arguments changed to send to correct locations describeData(data,experiment_path,dataset_name)
    ExploratoryAnalysisJob.countsSummary(repData,class_label,full_path,'applymodel/'+apply_name,instance_label,match_label,categorical_variables,jupyterRun)
    ExploratoryAnalysisJob.missingnessCounts(repData,full_path,'applymodel/'+apply_name,jupyterRun)
    #Create features-only version of dataset for some operations
    if instance_label == "None" and match_label == "None":
        x_repData = repData.drop([class_label],axis=1) #exclude class column
    elif not instance_label == "None" and match_label == "None":
        x_repData = repData.drop([class_label,instance_label],axis=1) #exclude class column
    elif instance_label == "None" and not match_label == "None":
        x_repData = repData.drop([class_label,match_label],axis=1) #exclude class column
    else:
        x_repData = repData.drop([class_label,instance_label,match_label],axis=1) #exclude class column
    #Export feature correlation plot if user specified
    if eval(export_feature_correlations):
        ExploratoryAnalysisJob.featureCorrelationPlot(x_repData,full_path,'applymodel/'+apply_name,jupyterRun)
    del x_repData #memory cleanup
    #Rep Data Preparation for each Training Partion Model set (rep data will potentially be scaled, imputed and feature selected in the same was as was done for each corresponding CV training partition)
    masterList = [] #Will hold all evalDict's, one for each cv dataset.
    for cvCount in range(0,cv_partitions):
        #Get corresponding training CV dataset
        cv_train_path = full_path+"/CVDatasets/"+train_name+'_CV_'+str(cvCount)+'_Train.csv'
        cv_train_data = pd.read_csv(cv_train_path, na_values='NA', sep = ",")
        #Get List of features in cv dataset (if feature selection took place this may only include a subset of original training data features)
        train_feature_list = list(cv_train_data.columns.values)
        train_feature_list.remove(class_label)
        if instance_label != 'None':
            train_feature_list.remove(instance_label)
        #Working copy of original dataframe - a new version will be created for each CV partition to be applied to each corresponding set of models
        cvRepData = repData.copy()
        #Impute dataframe based on training imputation
        if eval(impute_data):
            cvRepData = imputeRepData(full_path,cvCount,instance_label,class_label,cvRepData,all_train_feature_list)
        #Scale dataframe based on training scaling
        if eval(scale_data):
            cvRepData = scaleRepData(full_path,cvCount,instance_label,class_label,cvRepData,all_train_feature_list)
        #Conduct feature selection based on training selection (Filters out any features not in the final cv training dataset)
        cvRepData = cvRepData[cv_train_data.columns]
        del cv_train_data #memory cleanup
        #Prep data for evaluation
        if instance_label != 'None':
            cvRepData = cvRepData.drop(instance_label,axis=1)
        cvRepDataX = cvRepData.drop(class_label,axis=1).values
        cvRepDataY = cvRepData[class_label].values
        #For all previously trained algorithms, load pickled model and apply prepared replication dataset
        evalDict = {}
        algorithms = []
        if eval(do_LR):
            algorithm = 'LR'
            algorithms.append('Logistic Regression')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_DT):
            algorithm = 'DT'
            algorithms.append('Decision Tree')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_RF):
            algorithm = 'RF'
            algorithms.append('Random Forest')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_NB):
            algorithm = 'NB'
            algorithms.append('Naive Bayes')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_XGB):
            algorithm = 'XGB'
            algorithms.append('XGB')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_LGB):
            algorithm = 'LGB'
            algorithms.append('LGB')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_ANN):
            algorithm = 'ANN'
            algorithms.append('ANN')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_SVM):
            algorithm = 'SVM'
            algorithms.append('SVM')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_eLCS):
            algorithm = 'eLCS'
            algorithms.append('eLCS')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_XCS):
            algorithm = 'XCS'
            algorithms.append('XCS')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_ExSTraCS):
            algorithm = 'ExSTraCS'
            algorithms.append('ExSTraCS')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_GB):
            algorithm = 'GB'
            algorithms.append('Gradient Boosting')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        if eval(do_KN):
            algorithm = 'KN'
            algorithms.append('K Neighbors')
            evalDict[algorithm] = evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount)
        masterList.append(evalDict) #update master list with evalDict for this CV model
    #Summarize top_results
    abbrev = {'Logistic Regression':'LR','Decision Tree':'DT','Random Forest':'RF','Naive Bayes':'NB','XGB':'XGB','LGB':'LGB','ANN':'ANN','SVM':'SVM','ExSTraCS':'ExSTraCS','eLCS':'eLCS','XCS':'XCS','Gradient Boosting':'GB','K Neighbors':'KN'}
    colors = {'Logistic Regression':'black','Decision Tree':'yellow','Random Forest':'orange','Naive Bayes':'grey','XGB':'purple','LGB':'aqua','ANN':'red','SVM':'blue','ExSTraCS':'lightcoral','eLCS':'firebrick','XCS':'deepskyblue','Gradient Boosting':'bisque','K Neighbors':'seagreen'}
    result_table,metric_dict = primaryStats(algorithms,cv_partitions,full_path,apply_name,instance_label,class_label,abbrev,colors,plot_ROC,plot_PRC,jupyterRun,masterList,repData)
    StatsJob.doPlotROC(result_table,colors,full_path+'/applymodel/'+apply_name,jupyterRun)
    doPlotPRC(result_table,colors,full_path,apply_name,instance_label,class_label,jupyterRun,repData)
    metrics = list(metric_dict[algorithms[0]].keys())
    StatsJob.saveMetricMeans(full_path+'/applymodel/'+apply_name,metrics,metric_dict)
    StatsJob.saveMetricStd(full_path+'/applymodel/'+apply_name,metrics,metric_dict)
    if eval(plot_metric_boxplots):
        StatsJob.metricBoxplots(full_path+'/applymodel/'+apply_name,metrics,algorithms,metric_dict,jupyterRun)
    #Save Kruskal Wallis, Mann Whitney, and Wilcoxon Rank Sum Stats
    if len(algorithms) > 1:
        kruskal_summary = StatsJob.kruskalWallis(full_path+'/applymodel/'+apply_name,metrics,algorithms,metric_dict,sig_cutoff)
        StatsJob.mannWhitneyU(full_path+'/applymodel/'+apply_name,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff)
        StatsJob.wilcoxonRank(full_path+'/applymodel/'+apply_name,metrics,algorithms,metric_dict,kruskal_summary,sig_cutoff)

def scaleRepData(full_path,cvCount,instance_label,class_label,cvRepData,all_train_feature_list):
    scaleInfo = full_path+'/exploratory/scale_impute/scaler_cv'+str(cvCount) #Corresponding pickle file name with scalingInfo
    infile = open(scaleInfo,'rb')
    scaler = pickle.load(infile)
    infile.close()
    #Scale target replication data
    if instance_label == None or instance_label == 'None':
        x_rep = cvRepData.drop([class_label], axis=1)
    else:
        x_rep = cvRepData.drop([class_label, instance_label], axis=1)
        inst_rep = cvRepData[instance_label]  # pull out instance labels in case they include text
    y_rep = cvRepData[class_label]
    # Scale features (x)
    x_rep_scaled = pd.DataFrame(scaler.transform(x_rep), columns=x_rep.columns)
    # Recombine x and y
    if instance_label == None or instance_label == 'None':
        scale_rep_df = pd.concat([pd.DataFrame(y_rep, columns=[class_label]), pd.DataFrame(x_rep_scaled, columns=all_train_feature_list)],axis=1, sort=False)
    else:
        scale_rep_df = pd.concat([pd.DataFrame(y_rep, columns=[class_label]), pd.DataFrame(inst_rep, columns=[instance_label]),pd.DataFrame(x_rep_scaled, columns=all_train_feature_list)], axis=1, sort=False)
    return scale_rep_df

def imputeRepData(full_path,cvCount,instance_label,class_label,cvRepData,all_train_feature_list):
    imputeOridinalInfo = full_path+'/exploratory/scale_impute/ordinal_imputer_cv'+str(cvCount) #Corresponding pickle file name with scalingInfo
    infile = open(imputeOridinalInfo,'rb')
    imputer = pickle.load(infile)
    infile.close()
    imputeCatInfo = full_path+'/exploratory/scale_impute/categorical_imputer_cv'+str(cvCount) #Corresponding pickle file name with scalingInfo
    infile = open(imputeCatInfo,'rb')
    mode_dict = pickle.load(infile)
    infile.close()
    cvRepData.shape
    #Impute categorical features (i.e. those included in the mode_dict)
    for c in cvRepData.columns:
        if c in mode_dict: #was the given feature identified as and treated as categorical during training?
            cvRepData[c].fillna(mode_dict[c], inplace=True)
    #Preprare data for scikit imputation
    if instance_label == None or instance_label == 'None':
        x_rep = cvRepData.drop([class_label], axis=1).values
    else:
        x_rep = cvRepData.drop([class_label, instance_label], axis=1).values
        inst_rep = cvRepData[instance_label].values  # pull out instance labels in case they include text
    y_rep = cvRepData[class_label].values
    x_rep_impute = imputer.transform(x_rep)
    # Recombine x and y
    if instance_label == None or instance_label == 'None':
        impute_rep_df = pd.concat([pd.DataFrame(y_rep, columns=[class_label]), pd.DataFrame(x_rep_impute, columns=all_train_feature_list)],axis=1, sort=False)
    else:
        impute_rep_df = pd.concat([pd.DataFrame(y_rep, columns=[class_label]), pd.DataFrame(inst_rep, columns=[instance_label]),pd.DataFrame(x_rep_impute, columns=all_train_feature_list)], axis=1, sort=False)
    return impute_rep_df

def evalModel(full_path,algorithm,cvRepDataX,cvRepDataY,cvCount):
    modelInfo = full_path+'/training/pickledModels/'+algorithm+'_'+str(cvCount) #Corresponding pickle file name with scalingInfo
    infile = open(modelInfo,'rb')
    model = pickle.load(infile)
    infile.close()
    # Prediction evaluation
    y_pred = model.predict(cvRepDataX)
    metricList = ModelJob.classEval(cvRepDataY, y_pred)
    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(cvRepDataX)
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(cvRepDataY, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(cvRepDataY, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1] #reversed list orders
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(cvRepDataY, probas_[:, 1])
    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, probas_]

def primaryStats(algorithms,cv_partitions,full_path,apply_name,instance_label,class_label,abbrev,colors,plot_ROC,plot_PRC,jupyterRun,masterList,repData):
    #Main Ops
    result_table = []
    metric_dict = {}
    for algorithm in algorithms:
        alg_result_table = []
        # Define evaluation stats variable lists
        s_bac = []
        s_ac = []
        s_f1 = []
        s_re = []
        s_sp = []
        s_pr = []
        s_tp = []
        s_tn = []
        s_fp = []
        s_fn = []
        s_npv = []
        s_lrp = []
        s_lrm = []
        # Define ROC plot variable lists
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)
        # Define PRC plot variable lists
        precs = []
        praucs = []
        aveprecs = []
        #Gather statistics over all CV partitions
        for cvCount in range(0,cv_partitions):
            results = masterList[cvCount][abbrev[algorithm]] #grabs evalDict for a specific algorithm entry (with data values)
            metricList = results[0]
            fpr = results[1]
            tpr = results[2]
            roc_auc = results[3]
            prec = results[4]
            recall = results[5]
            prec_rec_auc = results[6]
            ave_prec = results[7]
            s_bac.append(metricList[0])
            s_ac.append(metricList[1])
            s_f1.append(metricList[2])
            s_re.append(metricList[3])
            s_sp.append(metricList[4])
            s_pr.append(metricList[5])
            s_tp.append(metricList[6])
            s_tn.append(metricList[7])
            s_fp.append(metricList[8])
            s_fn.append(metricList[9])
            s_npv.append(metricList[10])
            s_lrp.append(metricList[11])
            s_lrm.append(metricList[12])
            alg_result_table.append([fpr, tpr, roc_auc, recall, prec, prec_rec_auc, ave_prec])
            # Update ROC plot variable lists
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)
            # Update PRC plot variable lists
            precs.append(interp(mean_recall, recall, prec))
            praucs.append(prec_rec_auc)
            aveprecs.append(ave_prec)

        #CV ROC plot ------------------------------------------------------------------
        if jupyterRun:
            print(algorithm)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        if eval(plot_ROC):
            # Plot individual CV ROC line
            plt.rcParams["figure.figsize"] = (6,6)
            for i in range(cv_partitions):
                plt.plot(alg_result_table[i][0], alg_result_table[i][1], lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.3f)' % (i, alg_result_table[i][2]))
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='No-Skill', alpha=.8)
            std_auc = np.std(aucs)
            std_tpr = np.std(tprs, axis=0)
            plt.plot(mean_fpr, mean_tpr, color=colors[algorithm],label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),lw=2, alpha=.8)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            #plt.title(algorithm + ' : ROC over CV Partitions')
            plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
            plt.savefig(full_path+"/applymodel/"+apply_name+'/training/results/'+abbrev[algorithm]+"_ROC.png", bbox_inches="tight")
            if eval(jupyterRun):
                plt.show()
            else:
                plt.close('all')

        #CV PRC plot ---------------------------------------------------------------
        mean_prec = np.mean(precs, axis=0)
        mean_pr_auc = np.mean(praucs)
        if eval(plot_PRC):
            for i in range(cv_partitions):
                plt.plot(alg_result_table[i][3], alg_result_table[i][4], lw=1, alpha=0.3, label='PRC fold %d (AUC = %0.3f)' % (i, alg_result_table[i][5]))
            repClass = repData[class_label].values
            noskill = len(repClass[repClass == 1]) / len(repClass)  # Fraction of cases
            plt.plot([0, 1], [noskill, noskill], color='orange', linestyle='--', label='No-Skill', alpha=.8)
            std_pr_auc = np.std(praucs)
            std_prec = np.std(precs, axis=0)
            plt.plot(mean_recall, mean_prec, color=colors[algorithm],label=r'Mean PRC (AUC = %0.3f $\pm$ %0.3f)' % (mean_pr_auc, std_pr_auc),lw=2, alpha=.8)
            precs_upper = np.minimum(mean_prec + std_prec, 1)
            precs_lower = np.maximum(mean_prec - std_prec, 0)
            plt.fill_between(mean_fpr, precs_lower, precs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('Recall (Sensitivity)')
            plt.ylabel('Precision (PPV)')
            #plt.title(algorithm + ' : PRC over CV Partitions')
            plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
            plt.savefig(full_path+"/applymodel/"+apply_name+'/training/results/'+abbrev[algorithm]+"_PRC.png", bbox_inches="tight")
            if eval(jupyterRun):
                plt.show()
            else:
                plt.close('all')
        #Save Average Algorithm Stats
        results = {'Balanced Accuracy': s_bac, 'Accuracy': s_ac, 'F1_Score': s_f1, 'Sensitivity (Recall)': s_re, 'Specificity': s_sp,'Precision (PPV)': s_pr, 'TP': s_tp, 'TN': s_tn, 'FP': s_fp, 'FN': s_fn, 'NPV': s_npv, 'LR+': s_lrp, 'LR-': s_lrm, 'ROC_AUC': aucs,'PRC_AUC': praucs, 'PRC_APS': aveprecs}
        dr = pd.DataFrame(results)
        filepath = full_path+"/applymodel/"+apply_name+'/training/results/'+abbrev[algorithm]+"_performance.csv"
        dr.to_csv(filepath, header=True, index=False)
        metric_dict[algorithm] = results
        #Store ave metrics for creating global ROC and PRC plots later
        mean_ave_prec = np.mean(aveprecs)
        result_dict = {'algorithm':algorithm,'fpr':mean_fpr, 'tpr':mean_tpr, 'auc':mean_auc, 'prec':mean_prec, 'pr_auc':mean_pr_auc, 'ave_prec':mean_ave_prec}
        result_table.append(result_dict)
    result_table = pd.DataFrame.from_dict(result_table)
    result_table.set_index('algorithm',inplace=True)
    return result_table,metric_dict

def doPlotPRC(result_table,colors,full_path,apply_name,instance_label,class_label,jupyterRun,repData):
    #Plot summarizing average PRC across algorithms
    count = 0
    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],result_table.loc[i]['prec'], color=colors[i],label="{}, AUC={:.3f}, APS={:.3f}".format(i, result_table.loc[i]['pr_auc'],result_table.loc[i]['ave_prec']))
        count += 1
    repClass = repData[class_label].values
    noskill = len(repClass[repClass == 1]) / len(repClass)  # Fraction of cases
    plt.plot([0, 1], [noskill, noskill], color='orange', linestyle='--',label='No-Skill', alpha=.8)
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall (Sensitivity)", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision (PPV)", fontsize=15)
    #plt.title('Comparing Algorithms: Testing Data with CV', fontweight='bold', fontsize=15)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01,1))
    #plt.legend(prop={'size': 13}, loc='best')
    plt.savefig(full_path+'/applymodel/'+apply_name+'/training/results/Summary_PRC.png', bbox_inches="tight")
    if eval(jupyterRun):
        plt.show()
    else:
        plt.close('all')

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]),float(sys.argv[6]),int(sys.argv[7]),sys.argv[8],sys.argv[9],sys.argv[10],sys.argv[11],sys.argv[12],sys.argv[13],sys.argv[14],sys.argv[15],sys.argv[16],sys.argv[17],sys.argv[18],sys.argv[19],sys.argv[20],sys.argv[21],sys.argv[22],sys.argv[23],sys.argv[24],sys.argv[25],sys.argv[26],sys.argv[27],sys.argv[28],sys.argv[29],sys.argv[30])
