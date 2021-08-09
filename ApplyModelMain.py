"""
File: ApplyModelMain.py
Authors: Ryan J. Urbanowicz, Richard Zhang, Wilson Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 10 of AutoMLPipe-BC (Optional)- This 'Main' script manages Phase 10 run parameters, and submits job to run locally (to run serially) or on a linux computing
             cluster (parallelized). This script runs ApplyModelJob.py which applies and evaluates all trained models on one or more previously unseen hold-out or replication study dataset(s).
             All 'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion anytime after AutoMLPipe-BC Phase 6 (StatsMain.py).
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python ApplyModelMain.py --rep-path /Users/robert/Desktop/RepDatasets --dataset /Users/robert/Desktop/Datasets/targetData1.csv --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python ApplyModelMain.py --rep-path /Users/robert/Desktop/RepDatasets --dataset /Users/robert/Desktop/Datasets/targetData1.csv --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import pandas as pd
import FeatureSelectionJob
import time
import csv
import glob

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--rep-path',dest='rep_data_path',type=str,help='path to directory containing replication or hold-out testing datasets (must have at least all features with same labels as in original training dataset)')
    parser.add_argument('--dataset',dest='data_path',type=str,help='path to target original training dataset')
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--export-fc', dest='export_feature_correlations', type=str, help='run and export feature correlation analysis (yields correlation heatmap)',default="True")
    parser.add_argument('--plot-ROC', dest='plot_ROC', type=str,help='Plot ROC curves individually for each algorithm including all CV results and averages', default='True')
    parser.add_argument('--plot-PRC', dest='plot_PRC', type=str,help='Plot PRC curves individually for each algorithm including all CV results and averages', default='True')
    parser.add_argument('--plot-box', dest='plot_metric_boxplots', type=str,help='Plot box plot summaries comparing algorithms for each metric', default='True')
    parser.add_argument('--top-results', dest='top_results', type=int,help='number of top features to illustrate in figures', default=20)
    parser.add_argument('--match-label', dest='match_label', type=str, help='applies if original training data included column with matched instance ids', default="None")

    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    jupyterRun = 'False'
    job_counter = 0

    #Load variables specified earlier in the pipeline from metadata file
    metadata = pd.read_csv(options.output_path + '/' + options.experiment_name + '/' + 'metadata.csv').values
    experiment_path = options.output_path+'/'+options.experiment_name
    data_name = options.data_path.split('/')[-1].split('.')[0] #Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    categorical_cutoff = metadata[4, 1]
    sig_cutoff = metadata[5,1]
    cv_partitions = metadata[6,1]
    scale_data = metadata[10,1]
    impute_data = metadata[11,1]
    do_NB = metadata[19,1]
    do_LR = metadata[20,1]
    do_DT = metadata[21,1]
    do_RF = metadata[22,1]
    do_GB = metadata[23, 1]
    do_XGB = metadata[24,1]
    do_LGB = metadata[25,1]
    do_SVM = metadata[26,1]
    do_ANN = metadata[27,1]
    do_KN = metadata[28, 1]
    do_eLCS = metadata[29,1]
    do_XCS = metadata[30,1]
    do_ExSTraCS = metadata[31,1]
    primary_metric = metadata[32,1]

    # Argument checks
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1-5) before model application can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1-5) before model application can begin")

    full_path = options.output_path + "/" + options.experiment_name + "/" + data_name #location of folder containing models respective training dataset

    if not os.path.exists(full_path+"/applymodel"):
        os.mkdir(full_path+"/applymodel")

    #Determine file extension of datasets in target folder:
    file_count = 0
    unique_datanames = []
    for datasetFilename in glob.glob(options.rep_data_path+'/*'):
        file_extension = datasetFilename.split('/')[-1].split('.')[-1]
        apply_name = datasetFilename.split('/')[-1].split('.')[0] #Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
        if not os.path.exists(full_path+"/applymodel/"+apply_name):
            os.mkdir(full_path+"/applymodel/"+apply_name)

        if file_extension == 'txt' or file_extension == 'csv':
            if apply_name not in unique_datanames:
                unique_datanames.append(apply_name)
                if eval(options.run_parallel):
                    job_counter += 1
                    submitClusterJob(options.reserved_memory,options.maximum_memory,options.queue,experiment_path,datasetFilename,full_path,class_label,instance_label,categorical_cutoff,sig_cutoff,cv_partitions,scale_data,impute_data,do_LR,do_DT,do_RF,do_NB,do_XGB,do_LGB,do_SVM,do_ANN,do_ExSTraCS,do_eLCS,do_XCS,do_GB,do_KN,primary_metric,options.data_path,options.match_label,options.plot_ROC,options.plot_PRC,options.plot_metric_boxplots,options.export_feature_correlations,jupyterRun)
                else:
                    submitLocalJob(datasetFilename,full_path,class_label,instance_label,categorical_cutoff,sig_cutoff,cv_partitions,scale_data,impute_data,do_LR,do_DT,do_RF,do_NB,do_XGB,do_LGB,do_SVM,do_ANN,do_ExSTraCS,do_eLCS,do_XCS,do_GB,do_KN,primary_metric,options.data_path,options.match_label,options.plot_ROC,options.plot_PRC,options.plot_metric_boxplots,options.export_feature_correlations,jupyterRun)
                file_count += 1

    if file_count == 0: #Check that there was at least 1 dataset
        raise Exception("There must be at least one .txt or .csv dataset in rep_data_path directory")

    print(str(job_counter)+ " jobs submitted in Phase 10")

def submitLocalJob(datasetFilename,full_path,class_label,instance_label,categorical_cutoff,sig_cutoff,cv_partitions,scale_data,impute_data,do_LR,do_DT,do_RF,do_NB,do_XGB,do_LGB,do_SVM,do_ANN,do_ExSTraCS,do_eLCS,do_XCS,do_GB,do_KN,primary_metric,data_path,match_label,plot_ROC,plot_PRC,plot_metric_boxplots,export_feature_correlations,jupyterRun):
    """ Runs ApplyModelJob.py on each dataset in dataset_path locally. These runs will be completed serially rather than in parallel. """
    ApplyModelJob.job(datasetFilename,full_path,class_label,instance_label,categorical_cutoff,sig_cutoff,cv_partitions,scale_data,impute_data,do_LR,do_DT,do_RF,do_NB,do_XGB,do_LGB,do_SVM,do_ANN,do_ExSTraCS,do_eLCS,do_XCS,do_GB,do_KN,primary_metric,data_path,match_label,plot_ROC,plot_PRC,plot_metric_boxplots,export_feature_correlations,jupyterRun)

def submitClusterJob(reserved_memory,maximum_memory,queue,experiment_path,datasetFilename,full_path,class_label,instance_label,categorical_cutoff,sig_cutoff,cv_partitions,scale_data,impute_data,do_LR,do_DT,do_RF,do_NB,do_XGB,do_LGB,do_SVM,do_ANN,do_ExSTraCS,do_eLCS,do_XCS,do_GB,do_KN,primary_metric,data_path,match_label,plot_ROC,plot_PRC,plot_metric_boxplots,export_feature_correlations,jupyterRun):
    """ Runs ApplyModelJob.py on each dataset in rep_data_path. Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    train_name = full_path.split('/')[-1] #original training data name
    apply_name = datasetFilename.split('/')[-1].split('.')[0]

    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/Apply_'+train_name+'_'+apply_name+'_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/Apply_'+train_name+'_'+apply_name+'_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/Apply_'+train_name+'_'+apply_name+'_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ApplyModelJob.py '+datasetFilename+" "+full_path+" "+class_label+" "+instance_label+" "+categorical_cutoff+" "+sig_cutoff+" "+cv_partitions+" "+scale_data+" "+impute_data+" "+do_LR+" "+do_DT+" "+do_RF+" "+do_NB+" "+do_XGB+" "+
                  do_LGB+" "+do_SVM+" "+do_ANN+" "+do_ExSTraCS+" "+do_eLCS+" "+do_XCS+" "+do_GB+" "+do_KN+" "+primary_metric+" "+data_path+" "+match_label+" "+plot_ROC+" "+plot_PRC+" "+plot_metric_boxplots+" "+export_feature_correlations+" "+jupyterRun+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
