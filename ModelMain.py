"""
File: ModelMain.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 5 of AutoMLPipe-BC - This 'Main' script manages Phase 5 run parameters, updates the metadata file (with user specified run parameters across pipeline run)
             and submits job to run locally (to run serially) or on a linux computing cluster (parallelized).  This script runs ModelJob.py which conducts machine learning
             modeling using respective training datasets. This pipeline currently includes the following 13 ML modeling algorithms for binary classification:
             * Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LGBoost, Support Vector Machine (SVM), Artificial Neural Network (ANN),
             * k Nearest Neighbors (k-NN), Educational Learning Classifier System (eLCS), X Classifier System (XCS), and the Extended Supervised Tracking and Classifying System (ExSTraCS)
             This phase includes hyperparameter optimization of all algorithms (other than naive bayes), model training, model feature importance estimation (using internal algorithm
             estimations, if available, or via permutation feature importance), and performance evaluation on hold out testing data. This script creates a single job for each
             combination of cv dataset (for each original target dataset) and ML modeling algorithm. In addition to an option to check the completion of all jobs, this script also has a
             'resubmit' option that will run any jobs that may have failed from a previous run. All 'Main' scripts in this pipeline have the potential to be extended by users to
             submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of AutoMLPipe-BC Phase 4 (FeatureSelectionMain.py). SVM modeling should only be applied when data scaling is applied by the pipeline
            Logistic Regression' baseline model feature importance estimation is determined by the exponential of the feature's coefficient. This should only be used if data scaling is
            applied by the pipeline. Otherwise 'use_uniform_FI' should be True.
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python ModelMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python ModelMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import pandas as pd
import glob
import ModelJob
import time
import csv
import random

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Sets default run all or none to make algorithm selection from command line simpler
    parser.add_argument('--do-all', dest='do_all', type=str, help='run all modeling algorithms by default (when set False, individual algorithms are activated individually)',default='True')
    #ML modeling algorithms: Defaults available
    parser.add_argument('--do-NB', dest='do_NB', type=str, help='run naive bayes modeling',default='True')
    parser.add_argument('--do-LR', dest='do_LR', type=str, help='run logistic regression modeling',default='True')
    parser.add_argument('--do-DT', dest='do_DT', type=str, help='run decision tree modeling',default='True')
    parser.add_argument('--do-RF', dest='do_RF', type=str, help='run random forest modeling',default='True')
    parser.add_argument('--do-GB', dest='do_GB', type=str, help='run gradient boosting modeling',default='True')
    parser.add_argument('--do-XGB', dest='do_XGB', type=str, help='run XGBoost modeling',default='True')
    parser.add_argument('--do-LGB', dest='do_LGB', type=str, help='run LGBoost modeling',default='True')
    parser.add_argument('--do-SVM', dest='do_SVM', type=str, help='run support vector machine modeling',default='True')
    parser.add_argument('--do-ANN', dest='do_ANN', type=str, help='run artificial neural network modeling',default='True')
    parser.add_argument('--do-KN', dest='do_KN', type=str, help='run k-neighbors classifier modeling',default='True')
    parser.add_argument('--do-eLCS', dest='do_eLCS', type=str, help='run eLCS modeling (a basic supervised-learning learning classifier system)',default='True')
    parser.add_argument('--do-XCS', dest='do_XCS', type=str, help='run XCS modeling (a supervised-learning-only implementation of the best studied learning classifier system)',default='True')
    parser.add_argument('--do-ExSTraCS', dest='do_ExSTraCS', type=str, help='run ExSTraCS modeling (a learning classifier system designed for biomedical data mining)',default='True')
    #Other Analysis Parameters - Defaults available
    parser.add_argument('--metric', dest='primary_metric', type=str,help='primary scikit-learn specified scoring metric used for hyperparameter optimization and permutation-based model feature importance evaluation', default='balanced_accuracy')
    parser.add_argument('--subsample', dest='training_subsample', type=int, help='for long running algos (XGB,SVM,ANN,KN), option to subsample training set (0 for no subsample)', default=0)
    parser.add_argument('--use-uniformFI', dest='use_uniform_FI', type=str, help='overrides use of any available feature importance estimate methods from models, instead using permutation_importance uniformly',default='False')
    #Hyperparameter sweep options - Defaults available
    parser.add_argument('--n-trials', dest='n_trials', type=int,help='# of bayesian hyperparameter optimization trials using optuna', default=100)
    parser.add_argument('--timeout', dest='timeout', type=int,help='seconds until hyperparameter sweep stops running new trials (Note: it may run longer to finish last trial started)', default=900) #900 sec = 15 minutes default
    parser.add_argument('--export-hyper-sweep', dest='export_hyper_sweep_plots', type=str, help='export optuna-generated hyperparameter sweep plots', default='False')
    #LCS specific parameters - Defaults available
    parser.add_argument('--do-LCS-sweep', dest='do_lcs_sweep', type=str, help='do LCS hyperparam tuning or use below params',default='False')
    parser.add_argument('--nu', dest='nu', type=int, help='fixed LCS nu param', default=1)
    parser.add_argument('--iter', dest='iterations', type=int, help='fixed LCS # learning iterations param', default=200000)
    parser.add_argument('--N', dest='N', type=int, help='fixed LCS rule population maximum size param', default=2000)
    parser.add_argument('--lcs-timeout', dest='lcs_timeout', type=int, help='seconds until hyperparameter sweep stops for LCS algorithms', default=1200)
    #Lostistical arguments - Defaults available
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')
    parser.add_argument('-r','--do-resubmit',dest='do_resubmit', help='Boolean: Rerun any jobs that did not complete (or failed) in an earlier run.', action='store_true')

    options = parser.parse_args(argv[1:])
    job_counter = 0
    #Code to allow more flexible specification of which ML algorithms to run (i.e. all minus specific algorithms or none plus specific algorithms)
    if eval(options.do_all):
        do_NB = True
        do_LR = True
        do_DT = True
        do_RF = True
        do_GB = True
        do_XGB = True
        do_LGB = True
        do_SVM = True
        do_ANN = True
        do_KN = True
        do_eLCS = True
        do_XCS = True
        do_ExSTraCS = True
        algorithms = ['naive_bayes','logistic_regression','decision_tree','random_forest','gradient_boosting','XGB','LGB','SVM','ANN','k_neighbors','eLCS','XCS','ExSTraCS']
        if options.do_NB == 'False':
            do_NB = False
            algorithms.remove('naive_bayes')
        if options.do_LR == 'False':
            do_LR = False
            algorithms.remove("logistic_regression")
        if options.do_DT == 'False':
            do_DT = False
            algorithms.remove("decision_tree")
        if options.do_RF == 'False':
            do_RF = False
            algorithms.remove('random_forest')
        if options.do_GB == 'False':
            do_GB = False
            algorithms.remove('gradient_boosting')
        if options.do_XGB == 'False':
            do_XGB = False
            algorithms.remove('XGB')
        if options.do_LGB == 'False':
            do_LGB = False
            algorithms.remove('LGB')
        if options.do_SVM == 'False':
            do_SVM = False
            algorithms.remove('SVM')
        if options.do_ANN == 'False':
            do_ANN = False
            algorithms.remove('ANN')
        if options.do_KN == 'False':
            do_KN = False
            algorithms.remove('k_neighbors')
        if options.do_eLCS == 'False':
            do_eLCS = False
            algorithms.remove('eLCS')
        if options.do_XCS == 'False':
            do_XCS = False
            algorithms.remove('XCS')
        if options.do_ExSTraCS == 'False':
            do_ExSTraCS = False
            algorithms.remove('ExSTraCS')
    else:
        do_NB = False
        do_LR = False
        do_DT = False
        do_RF = False
        do_GB = False
        do_XGB = False
        do_LGB = False
        do_SVM = False
        do_ANN = False
        do_KN = False
        do_eLCS = False
        do_XCS = False
        do_ExSTraCS = False
        algorithms = []
        if options.do_NB == 'True':
            do_NB = True
            algorithms.append('naive_bayes')
        if options.do_LR == 'True':
            do_LR = True
            algorithms.append("logistic_regression")
        if options.do_DT == 'True':
            do_DT = True
            algorithms.append("decision_tree")
        if options.do_RF == 'True':
            do_RF = True
            algorithms.append('random_forest')
        if options.do_GB == 'True':
            do_GB = True
            algorithms.append('gradient_boosting')
        if options.do_XGB == 'True':
            do_XGB = True
            algorithms.append('XGB')
        if options.do_LGB == 'True':
            do_LGB = True
            algorithms.append('LGB')
        if options.do_SVM == 'True':
            do_SVM = True
            algorithms.append('SVM')
        if options.do_ANN == 'True':
            do_ANN = True
            algorithms.append('ANN')
        if options.do_KN == 'True':
            do_KN = True
            algorithms.append('k_neighbors')
        if options.do_eLCS == 'True':
            do_eLCS = True
            algorithms.append('eLCS')
        if options.do_XCS == 'True':
            do_XCS = True
            algorithms.append('XCS')
        if options.do_ExSTraCS == 'True':
            do_ExSTraCS = True
            algorithms.append('ExSTraCS')

    # Argument checks
    if not os.path.exists(options.output_path):
        raise Exception("Output path must exist (from phase 1) before phase 5 can begin")
    if not os.path.exists(options.output_path + '/' + options.experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 5 can begin")

    #Load variables specified earlier in the pipeline from metadata file
    metadata = pd.read_csv(options.output_path + '/' + options.experiment_name + '/' + 'metadata.csv').values
    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    random_state = int(metadata[3,1])
    cv_partitions = int(metadata[6,1])
    filter_poor_features = metadata[16,1]

    if options.do_resubmit: #Attempts to resolve optuna hyperparameter optimization hangup (i.e. when it runs indefinitely for a given random seed attempt)
        random_state = random.randint(1,1000)

    if not options.do_check and not options.do_resubmit: #Run job submission
        dataset_paths = os.listdir(options.output_path + "/" + options.experiment_name)
        dataset_paths.remove('logs')
        dataset_paths.remove('jobs')
        dataset_paths.remove('jobsCompleted')
        dataset_paths.remove('metadata.csv')
        for dataset_directory_path in dataset_paths:
            full_path = options.output_path + "/" + options.experiment_name + "/" + dataset_directory_path
            if not os.path.exists(full_path+'/training'):
                os.mkdir(full_path+'/training')
            if not os.path.exists(full_path+'/training/pickledModels'):
                os.mkdir(full_path+'/training/pickledModels')
            for cvCount in range(cv_partitions):
                train_file_path = full_path+'/CVDatasets/'+dataset_directory_path+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path = full_path + '/CVDatasets/' + dataset_directory_path + "_CV_" + str(cvCount) + "_Test.csv"
                for algorithm in algorithms:
                    if eval(options.run_parallel):
                        job_counter += 1
                        submitClusterJob(algorithm,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.lcs_timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,options.output_path+'/'+options.experiment_name,cvCount,filter_poor_features,options.reserved_memory,options.maximum_memory,options.do_lcs_sweep,options.nu,options.iterations,options.N,options.training_subsample,options.queue,options.use_uniform_FI,options.primary_metric)
                    else:
                        submitLocalJob(algorithm,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.lcs_timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,options.do_lcs_sweep,options.nu,options.iterations,options.N,options.training_subsample,options.use_uniform_FI,options.primary_metric)

        # Update metadata
        if metadata.shape[0] == 20: #Only update if metadata below hasn't been added before
            with open(options.output_path + '/' + options.experiment_name + '/' + 'metadata.csv', mode='a', newline="") as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["NB", str(do_NB)])
                writer.writerow(["LR", str(do_LR)])
                writer.writerow(["DT", str(do_DT)])
                writer.writerow(["RF", str(do_RF)])
                writer.writerow(["GB",str(do_GB)])
                writer.writerow(["XGB", str(do_XGB)])
                writer.writerow(["LGB", str(do_LGB)])
                writer.writerow(["SVM", str(do_SVM)])
                writer.writerow(["ANN", str(do_ANN)])
                writer.writerow(["KN", str(do_KN)])
                writer.writerow(["eLCS", str(do_eLCS)])
                writer.writerow(["XCS",str(do_XCS)])
                writer.writerow(["ExSTraCS",str(do_ExSTraCS)])
                writer.writerow(["primary metric",options.primary_metric])
                writer.writerow(["training subsample for KN,ANN,SVM,and XGB",options.training_subsample])
                writer.writerow(["uniform feature importance estimation (models)",options.use_uniform_FI])
                writer.writerow(["hypersweep number of trials",options.n_trials])
                writer.writerow(["hypersweep timeout",options.timeout])
                writer.writerow(['do LCS sweep',options.do_lcs_sweep])
                writer.writerow(['nu', options.nu])
                writer.writerow(['training iterations', options.iterations])
                writer.writerow(['N (rule population size)', options.N])
                writer.writerow(["LCS hypersweep timeout",options.lcs_timeout])
            file.close()

    elif options.do_check and not options.do_resubmit: #run job completion checks
        abbrev = {'naive_bayes':'NB','logistic_regression':'LR','decision_tree':'DT','random_forest':'RF','gradient_boosting':'GB','XGB':'XGB','LGB':'LGB','ANN':'ANN','SVM':'SVM','k_neighbors':'KN','eLCS':'eLCS','XCS':'XCS','ExSTraCS':'ExSTraCS'}

        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.csv' in datasets:
            datasets.remove('metadata.csv')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')

        phase5Jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                for algorithm in algorithms:
                    phase5Jobs.append('job_model_' + dataset + '_' + str(cv) +'_' +abbrev[algorithm]+'.txt')

        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_model*'):
            ref = filename.split('/')[-1]
            phase5Jobs.remove(ref)
        for job in phase5Jobs:
            print(job)
        if len(phase5Jobs) == 0:
            print("All Phase 5 Jobs Completed")
        else:
            print("Above Phase 5 Jobs Not Completed")
        print()

    elif options.do_resubmit and not options.do_check: #resubmit any jobs that didn't finish in previous run (mix of job check and job submit)
        abbrev = {'naive_bayes':'NB','logistic_regression':'LR','decision_tree':'DT','random_forest':'RF','gradient_boosting':'GB','XGB':'XGB','LGB':'LGB','ANN':'ANN','SVM':'SVM','k_neighbors':'KN','eLCS':'eLCS','XCS':'XCS','ExSTraCS':'ExSTraCS'}

        datasets = os.listdir(options.output_path + "/" + options.experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.csv' in datasets:
            datasets.remove('metadata.csv')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')

        #start by making list of finished jobs instead of all jobs then step through loop
        phase5completed = []
        for filename in glob.glob(options.output_path + "/" + options.experiment_name + '/jobsCompleted/job_model*'):
            ref = filename.split('/')[-1]
            phase5completed.append(ref)

        for dataset in datasets:
            for cv in range(cv_partitions):
                for algorithm in algorithms:
                    targetFile = 'job_model_' + dataset + '_' + str(cv) +'_' +abbrev[algorithm]+'.txt'
                    if targetFile not in phase5completed: #target for a re-submit
                        full_path = options.output_path + "/" + options.experiment_name + "/" + dataset
                        train_file_path = full_path+'/CVDatasets/'+dataset+"_CV_"+str(cv)+"_Train.csv"
                        test_file_path = full_path + '/CVDatasets/' + dataset + "_CV_" + str(cv) + "_Test.csv"
                        if eval(options.run_parallel):
                            job_counter += 1
                            submitClusterJob(algorithm,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.lcs_timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,options.output_path+'/'+options.experiment_name,cv,filter_poor_features,options.reserved_memory,options.maximum_memory,options.do_lcs_sweep,options.nu,options.iterations,options.N,options.training_subsample,options.queue,options.use_uniform_FI,options.primary_metric)
                        else:
                            submitLocalJob(algorithm,train_file_path,test_file_path,full_path,options.n_trials,options.timeout,options.lcs_timeout,options.export_hyper_sweep_plots,instance_label,class_label,random_state,cv,filter_poor_features,options.do_lcs_sweep,options.nu,options.iterations,options.N,options.training_subsample,options.use_uniform_FI,options.primary_metric)
    else:
        print("Run options in conflict. Do not request to run check and resubmit at the same time.")

    if not options.do_check:
        print(str(job_counter)+ " jobs submitted in Phase 5")

def submitLocalJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric):
    """ Runs ModelJob.py locally, once for each combination of cv dataset (for each original target dataset) and ML modeling algorithm. These runs will be completed serially rather than in parallel. """
    ModelJob.job(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features,do_lcs_sweep,nu,iterations,N,training_subsample,use_uniform_FI,primary_metric)

def submitClusterJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,experiment_path,cvCount,filter_poor_features,reserved_memory,maximum_memory,do_lcs_sweep,nu,iterations,N,training_subsample,queue,use_uniform_FI,primary_metric):
    """ Runs ModelJob.py once for each combination of cv dataset (for each original target dataset) and ML modeling algorithm. Runs in parallel on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P5_'+str(algorithm)+'_'+str(cvCount)+'_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P5_'+str(algorithm)+'_'+str(cvCount)+'_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P5_'+str(algorithm)+'_'+str(cvCount)+'_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ModelJob.py '+algorithm+" "+train_file_path+" "+test_file_path+" "+full_path+" "+
                  str(n_trials)+" "+str(timeout)+" "+str(lcs_timeout)+" "+export_hyper_sweep_plots+" "+instance_label+" "+class_label+" "+
                  str(random_state)+" "+str(cvCount)+" "+str(filter_poor_features)+" "+str(do_lcs_sweep)+" "+str(nu)+" "+str(iterations)+" "+str(N)+" "+str(training_subsample)+" "+str(use_uniform_FI)+" "+str(primary_metric)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
