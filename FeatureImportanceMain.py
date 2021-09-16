"""
File: FeatureImportanceMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 3 of AutoMLPipe-BC - This 'Main' script manages Phase 3 run parameters, updates the metadata file (with user specified run parameters across pipeline run)
             and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).  This script runs FeatureImportanceJob.py which conducts the
             filter-based feature importance estimations. All 'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel
             computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of AutoMLPipe-BC Phase 2 (DataPreprocessingMain.py).
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python FeatureImportanceMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python FeatureImportanceMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import glob
import FeatureImportanceJob
import time
import pandas as pd
import csv

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--do-mi', dest='do_mutual_info', type=str, help='do mutual information analysis',default="True")
    parser.add_argument('--do-ms', dest='do_multisurf', type=str, help='do multiSURF analysis',default="True")
    parser.add_argument('--use-turf', dest='use_TURF', type=str, help='use TURF wrapper around MultiSURF', default="False")
    parser.add_argument('--turf-pct', dest='TURF_pct', type=float, help='proportion of instances removed in an iteration (also dictates number of iterations)',default=0.5)
    parser.add_argument('--n-jobs', dest='n_jobs', type=int, help='number of cores dedicated to running algorithm; setting to -1 will use all available cores', default=1)
    parser.add_argument('--inst-sub', dest='instance_subset', type=int, help='sample subset size to use with multiSURF',default=2000)
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    job_counter = 0

    # Argument checks
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1) before phase 3 can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 3 can begin")

    #Load variables specified earlier in the pipeline from metadata file
    metadata = pd.read_csv(options.output_path+'/'+options.experiment_name + '/' + 'metadata.csv').values
    class_label = metadata[0, 1]
    instance_label = metadata[1,1]
    random_state = int(metadata[3, 1])
    categorical_cutoff = int(metadata[4,1])
    cv_partitions = int(metadata[6,1])

    if not options.do_check: #Run job file
        #Iterate through datasets, ignoring common folders
        dataset_paths = os.listdir(options.output_path+"/"+options.experiment_name)
        dataset_paths.remove('logs')
        dataset_paths.remove('jobs')
        dataset_paths.remove('jobsCompleted')
        dataset_paths.remove('metadata.csv')
        for dataset_directory_path in dataset_paths:
            full_path = options.output_path+"/"+options.experiment_name+"/"+dataset_directory_path
            experiment_path = options.output_path+'/'+options.experiment_name

            if eval(options.do_mutual_info):
                if not os.path.exists(full_path+"/mutualinformation"):
                    os.mkdir(full_path+"/mutualinformation")
                for cv_train_path in glob.glob(full_path+"/CVDatasets/*_CV_*Train.csv"):
                    command_text = '/FeatureImportanceJob.py ' + cv_train_path+" "+experiment_path+" "+str(random_state)+" "+class_label+" "+instance_label+" " +str(options.instance_subset)+" mi "+str(options.n_jobs)+' '+str(options.use_TURF)+' '+str(options.TURF_pct)
                    if eval(options.run_parallel):
                        job_counter += 1
                        submitClusterJob(command_text, experiment_path,options.reserved_memory,options.maximum_memory,options.queue)
                    else:
                        submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,options.instance_subset,'mi',options.n_jobs,options.use_TURF,options.TURF_pct)

            if eval(options.do_multisurf):
                if not os.path.exists(full_path+"/multisurf"):
                    os.mkdir(full_path+"/multisurf")
                for cv_train_path in glob.glob(full_path+"/CVDatasets/*_CV_*Train.csv"):
                    command_text = '/FeatureImportanceJob.py ' + cv_train_path+" "+experiment_path+" "+str(random_state)+" "+class_label+" "+instance_label+" " +str(options.instance_subset)+" ms "+str(options.n_jobs)+' '+str(options.use_TURF)+' '+str(options.TURF_pct)
                    if eval(options.run_parallel):
                        job_counter += 1
                        submitClusterJob(command_text, experiment_path,options.reserved_memory,options.maximum_memory,options.queue)
                    else:
                        submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,options.instance_subset,'ms',options.n_jobs,options.use_TURF,options.TURF_pct)

        #Update metadata
        if metadata.shape[0] == 13: #Only update if metadata below hasn't been added before (i.e. in a previous phase 2 run)
            with open(options.output_path + '/' + options.experiment_name + '/' + 'metadata.csv',mode='a', newline="") as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["mutual information",options.do_mutual_info])
                writer.writerow(["MultiSURF", options.do_multisurf])
                writer.writerow(["TURF",options.use_TURF])
                writer.writerow(["TURF cutoff", options.TURF_pct])
                writer.writerow(["MultiSURF instance subset", options.instance_subset])
            file.close()

    else: #Instead of running job, checks whether previously run jobs were successfully completed
        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.csv' in datasets:
            datasets.remove('metadata.csv')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')

        phase3Jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                if eval(options.do_multisurf):
                    phase3Jobs.append('job_multisurf_' + dataset + '_' + str(cv) + '.txt')
                if eval(options.do_mutual_info):
                    phase3Jobs.append('job_mutualinformation_' + dataset + '_' + str(cv) + '.txt')

        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_mu*'):
            ref = filename.split('/')[-1]
            phase3Jobs.remove(ref)
        for job in phase3Jobs:
            print(job)
        if len(phase3Jobs) == 0:
            print("All Phase 3 Jobs Completed")
        else:
            print("Above Phase 3 Jobs Not Completed")

    if not options.do_check:
        print(str(job_counter)+ " jobs submitted in Phase 3")

def submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,instance_subset,algorithm,n_jobs,use_TURF,TURF_pct):
    """ Runs FeatureImportanceJob.py locally on a single CV dataset applying one of the implemented feature importance algorithms. These runs will be completed serially rather than in parallel. """
    FeatureImportanceJob.job(cv_train_path,experiment_path,random_state,class_label,instance_label,instance_subset,algorithm,n_jobs,use_TURF,TURF_pct)

def submitClusterJob(command_text,experiment_path,reserved_memory,maximum_memory,queue):
    """ Runs FeatureImportanceJob.py on a single CV dataset applying one of the implemented feature importance algorithms. Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P3_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P3_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P3_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + command_text+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
