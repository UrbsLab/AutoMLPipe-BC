"""
File: StatsMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 6 of AutoMLPipe-BC - This 'Main' script manages Phase 6 run parameters, and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).
             This script runs StatsJob.py which (for a single orginal target dataset) creates summaries of ML classification evaluation statistics (means and standard deviations),
             ROC and PRC plots (comparing CV performance in the same ML algorithm and comparing average performance between ML algorithms), model feature importance averages over CV runs,
             boxplots comparing ML algorithms for each metric, Kruskal Wallis and Mann Whitney statistical comparsions between ML algorithms, model feature importance boxplots for each
             algorithm, and composite feature importance plots summarizing model feature importance across all ML algorithms. This script is run on all cv results for a given original
             target dataset from Phase 1. All 'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of AutoMLPipe-BC Phase 5 (ModelMain.py).
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python StatsMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python StatsMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import time
import pandas as pd
import StatsJob
import glob

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--plot-ROC', dest='plot_ROC', type=str,help='Plot ROC curves individually for each algorithm including all CV results and averages', default='True')
    parser.add_argument('--plot-PRC', dest='plot_PRC', type=str,help='Plot PRC curves individually for each algorithm including all CV results and averages', default='True')
    parser.add_argument('--plot-box', dest='plot_metric_boxplots', type=str,help='Plot box plot summaries comparing algorithms for each metric', default='True')
    parser.add_argument('--plot-FI_box', dest='plot_FI_box', type=str,help='Plot feature importance boxplots for each algorithm', default='True')
    parser.add_argument('--top-results', dest='top_results', type=int,help='number of top features to illustrate in figures', default=20)
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    jupyterRun = 'False'

    # Argument checks
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1) before phase 6 can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

    #Load variables specified earlier in the pipeline from metadata file
    metadata = pd.read_csv(options.output_path + '/' + options.experiment_name + '/' + 'metadata.csv').values
    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    sig_cutoff = metadata[5,1]
    cv_partitions = int(metadata[6,1])
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

    encodedAlgos = ''
    encodedAlgos = encode(do_NB, encodedAlgos)
    encodedAlgos = encode(do_LR,encodedAlgos)
    encodedAlgos = encode(do_DT, encodedAlgos)
    encodedAlgos = encode(do_RF, encodedAlgos)
    encodedAlgos = encode(do_GB, encodedAlgos)
    encodedAlgos = encode(do_XGB, encodedAlgos)
    encodedAlgos = encode(do_LGB, encodedAlgos)
    encodedAlgos = encode(do_SVM, encodedAlgos)
    encodedAlgos = encode(do_ANN, encodedAlgos)
    encodedAlgos = encode(do_KN, encodedAlgos)
    encodedAlgos = encode(do_eLCS, encodedAlgos)
    encodedAlgos = encode(do_XCS, encodedAlgos)
    encodedAlgos = encode(do_ExSTraCS, encodedAlgos)

    if not options.do_check: #Run job submission
        # Iterate through datasets
        dataset_paths = os.listdir(options.output_path + "/" + options.experiment_name)
        dataset_paths.remove('logs')
        dataset_paths.remove('jobs')
        dataset_paths.remove('jobsCompleted')
        dataset_paths.remove('metadata.csv')
        for dataset_directory_path in dataset_paths:
            full_path = options.output_path + "/" + options.experiment_name + "/" + dataset_directory_path
            if eval(options.run_parallel):
                submitClusterJob(full_path,encodedAlgos,options.plot_ROC,options.plot_PRC,options.plot_FI_box,class_label,instance_label,options.output_path+'/'+options.experiment_name,cv_partitions,options.reserved_memory,options.maximum_memory,options.queue,options.plot_metric_boxplots,primary_metric,options.top_results,sig_cutoff,jupyterRun)
            else:
                submitLocalJob(full_path,encodedAlgos,options.plot_ROC,options.plot_PRC,options.plot_FI_box,class_label,instance_label,cv_partitions,options.plot_metric_boxplots,primary_metric,options.top_results,sig_cutoff,jupyterRun)
    else: #run job completion checks
        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.csv' in datasets:
            datasets.remove('metadata.csv')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')

        phase6Jobs = []
        for dataset in datasets:
            phase6Jobs.append('job_stats_'+dataset+'.txt')

        for filename in glob.glob(options.output_path + "/" + options.experiment_name+'/jobsCompleted/job_stats*'):
            ref = filename.split('/')[-1]
            phase6Jobs.remove(ref)
        for job in phase6Jobs:
            print(job)
        if len(phase6Jobs) == 0:
            print("All Phase 6 Jobs Completed")
        else:
            print("Above Phase 6 Jobs Not Completed")
        print()

def submitLocalJob(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,cv_partitions,plot_metric_boxplots,primary_metric,top_results,sig_cutoff,jupyterRun):
    """ Runs StatsJob.py locally, once for each of the original target datasets (all CV datasets analyzed at once). These runs will be completed serially rather than in parallel. """
    StatsJob.job(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,cv_partitions,plot_metric_boxplots,primary_metric,top_results,sig_cutoff,jupyterRun)

def submitClusterJob(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,experiment_path,cv_partitions,reserved_memory,maximum_memory,queue,plot_metric_boxplots,primary_metric,top_results,sig_cutoff,jupyterRun):
    """ Runs StatsJob.py once for each of the original target datasets (all CV datasets analyzed at once). Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/P6_' + job_ref + '_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P6_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P6_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/StatsJob.py '+full_path+" "+encoded_algos+" "+plot_ROC+" "+plot_PRC+" "+plot_FI_box+" "+class_label+" "+instance_label+" "+str(cv_partitions)+" "+str(plot_metric_boxplots)+" "+str(primary_metric)+" "+str(top_results)+" "+str(sig_cutoff)+" "+str(jupyterRun)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

def encode(do_algo,encodedAlgos):
    """ Encodes boolean values identifying which ML algorithms were run and should be included in the stats analysis. """
    if eval(do_algo):
        encodedAlgos += '1'
    else:
        encodedAlgos += '0'
    return encodedAlgos

if __name__ == '__main__':
    sys.exit(main(sys.argv))
