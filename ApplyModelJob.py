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
import exploratoryanalysisjob

def job(datasetFilename,full_path,class_label,instance_label,categorical_cutoff,sig_cutoff,cv_partitions,scale_data,impute_data,do_LR,do_DT,do_RF,do_NB,do_XGB,do_LGB,do_SVM,do_ANN,do_ExSTraCS,do_eLCS,do_XCS,do_GB,do_KN,primary_metric):
    jupyterRun = 'False'
    data_name = full_path.split('/')[-1]

    #Load Replication Datasets
    repData = pd.read_csv(datasetFilename, na_values='NA', sep = ",")

    #Create Folder hierarchy
    if not os.path.exists(full_path+"/applymodel/"+data_name+'/'+'exploratory'):
        os.mkdir(full_path+"/applymodel/"+data_name+'/'+'exploratory')
    if not os.path.exists(full_path+"/applymodel/"+data_name+'/'+'CVDatasets'):
        os.mkdir(full_path+"/applymodel/"+data_name+'/'+'CVDatasets')
    if not os.path.exists(full_path+"/applymodel/"+data_name+'/'+'training'):
        os.mkdir(full_path+"/applymodel/"+data_name+'/'+'training')
    if not os.path.exists(full_path+"/applymodel/"+data_name+'/'+'training'+'/'+'pickledModels'):
        os.mkdir(full_path+"/applymodel/"+data_name+'/'+'training'+'/'+'pickledModels')
    if not os.path.exists(full_path+"/applymodel/"+data_name+'/'+'training'+'/'+'results'):
        os.mkdir(full_path+"/applymodel/"+data_name+'/'+'training'+'/'+'results')
    if not os.path.exists(full_path+"/applymodel/"+data_name+'/'+'training'+'/'+'performanceBoxplots'):
        os.mkdir(full_path+"/applymodel/"+data_name+'/'+'training'+'/'+'performanceBoxplots')
    if not os.path.exists(full_path+"/applymodel/"+data_name+'/'+'training'+'/'+'KWMW'):
        os.mkdir(full_path+"/applymodel/"+data_name+'/'+'training'+'/'+'KWMW')

#to do(have to fix passing paths below so they turn out right wehn we call them in exploratory since they don't match up)
    #ExploratoryAnalysis
    exploratoryanalysisjob.makeDataFolders(repData,experiment_path,data_name)
    repData = exploratoryanalysisjob.removeRowsColumns(repData,class_label,[])

    categorical_variables = exploratoryanalysisjob.idFeatureTypes(repData,[],instance_label,"None",class_label,categorical_cutoff)
    exploratoryanalysisjob.countsSummary(repData,class_label,experiment_path,data_name,instance_label,match_label,categorical_variables,jupyterRun)

    #Rep Data Preparation for each Training Partion Model set (rep data will potentially be scaled, imputed and feature selected in the same was as was done for each corresponding CV training partition)
    for cvCount in range(0,cv_partitions):
        #Get corresponding training CV dataset
        cv_train_path = full_path+"/CVDatasets/"+data_name+'_CV_'+str(cvCount)+'_Train.csv'
        cv_train_data = pd.read_csv(cv_train_path, na_values='NA', sep = ",")

        #Get List of features in that datasets
        feature_list = list(cv_train_data.columns.values)

        #Working copy of original dataframe
        cvRepData = repData.copy()

        #Scale dataframe based on training scaling
        if eval(scale_data):

            scaleInfo = full_path+'/exploratory/scale_impute/scaler_cv'+str(cvCount)






        if eval(impute_data):



    #grab feature list from that datasets
    #scale new dataset
    #impute new datasets
    #filter features with above list of new datasets




		#-load target rep data
		#-make header list from training datasets cv
		#-point to existing ml pipe experiment folder
		#-has to have same features as original data
		#-create internal folder hierarchy within project (replication folder, further named by datasets)
		#-scale and impute new data according to each CV’s specifications
		#-for each feature pick same feature subset
		#-for all algorithms load pickled model and apply transformed data
		#	-calculate all metrics
		#	-generate all plots
		#	-generate all outputputs


if __name__ == '__main__':
    job(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]),sys.argv[7], sys.argv[8],sys.argv[9],int(sys.argv[10]),sys.argv[11])
