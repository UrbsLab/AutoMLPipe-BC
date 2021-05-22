
import sys
import os
import argparse
import glob
import ExploratoryAnalysisJob
import time
import csv

'''Phase 1 of Machine Learning Analysis Pipeline:
Sample Run Command:
python ExploratoryAnalysisMain.py --data-path /Users/robert/Desktop/Datasets --output-path /Users/robert/Desktop/outputs --experiment-name test1

Local Command:
python ExploratoryAnalysisMain.py --data-path /Users/robert/Desktop/Datasets --output-path /Users/robert/Desktop/outputs --experiment-name test1 --run-parallel False --cv 3
OR
python ExploratoryAnalysisMain.py --instance-label Instance --data-path /Users/robert/Desktop/Datasets --output-path /Users/robert/Desktop/outputs --experiment-name randomtest2 --run-parallel False --cv 3


WARNINGS:
-no spaces are allowed in filenames!! - this will lead to 'invalid literal' export_exploratory_analysis
-all datasets in target folder being analyzed must have the same class and instance labels (if any for the latter)
'''


def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--data-path',dest='data_path',type=str,help='path to directory containing datasets')
    parser.add_argument('--out-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')
    #Defaults available (but critical to check)
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets', default="Class")
    parser.add_argument('--inst-label', dest='instance_label', type=str, help='instance label of all datasets (if present)', default="")
    parser.add_argument('--fi', dest='ignore_features_path',type=str, help='path to .csv file with feature labels to be ignored in analysis (e.g. /home/ryanurb/code/AutoMLPipe-BC/droppedFeatures.csv))', default="None")
    parser.add_argument('--cf', dest='categorical_feature_path',type=str, help='path to .csv file with feature labels specified to be treated as categorical where possible', default="None")
    #Defaults available (but less critical to check)
    parser.add_argument('--cv',dest='cv_partitions',type=int,help='number of CV partitions',default=10)
    parser.add_argument('--part',dest='partition_method',type=str,help="'S', or 'R', or 'M', for stratified, random, or matched, respectively",default="S")
    parser.add_argument('--match-label', dest='match_label', type=str, help='only applies when M selected for partition-method; indicates column with matched instance ids', default="")
    parser.add_argument('--cat-cutoff', dest='categorical_cutoff', type=int,help='number of unique values after which a variable is considered to be quantitative vs categorical', default=10)
    parser.add_argument('--sig', dest='sig_cutoff', type=float, help='significance cutoff used throughout pipeline',default=0.05)
    parser.add_argument('--export-ea', dest='export_exploratory_analysis', type=str, help='run and export basic exploratory analysis files, i.e. unique value counts, missingness counts, class balance barplot',default="True")
    parser.add_argument('--export-fc', dest='export_feature_correlations', type=str, help='run and export feature correlation analysis (yields correlation heatmap)',default="True")
    parser.add_argument('--export-up', dest='export_univariate_plots', type=str, help='export univariate analysis plots (note: univariate analysis still output by default)',default="False")
    parser.add_argument('--rand-state', dest='random_state', type=int, help='"Dont Panic" - sets a specific random seed for reproducible results',default=42)
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name

    class_label = options.class_label
    if options.instance_label == '':
        instance_label = 'None'
    else:
        instance_label = options.instance_label

    ignore_features_path = options.ignore_features_path
    categorical_feature_path = options.categorical_feature_path


    cv_partitions = options.cv_partitions
    partition_method = options.partition_method
    if options.match_label == '':
        match_label = 'None'
    else:
        match_label = options.match_label
    categorical_cutoff = options.categorical_cutoff
    sig_cutoff = options.sig_cutoff
    export_exploratory_analysis = options.export_exploratory_analysis
    export_feature_correlations = options.export_feature_correlations
    export_univariate_plots = options.export_univariate_plots
    random_state = options.random_state
    run_parallel = options.run_parallel
    queue = options.queue
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory
    do_check = options.do_check

    jupyterRun = 'False' #controls whether plots are shown or closed depending on whether jupyter notebook is used to run code or not
    if not do_check:
        makeDirTree(data_path,output_path,experiment_name,jupyterRun) #check file/path names and create directory tree for output

        #Determine file extension of datasets in target folder:
        file_count = 0
        unique_datanames = []
        for dataset_path in glob.glob(data_path+'/*'):
            file_extension = dataset_path.split('/')[-1].split('.')[-1]
            data_name = dataset_path.split('/')[-1].split('.')[0] #Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
            if file_extension == 'txt' or file_extension == 'csv':
                if data_name not in unique_datanames:
                    unique_datanames.append(data_name)
                    if eval(run_parallel):
                        submitClusterJob(dataset_path,output_path+'/'+experiment_name,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,reserved_memory,maximum_memory,queue,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun)
                    else:
                        submitLocalJob(dataset_path,output_path+'/'+experiment_name,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun)
                    file_count += 1

        if file_count == 0: #Check that there was at least 1 dataset
            raise Exception("There must be at least one .txt or .csv dataset in data_path directory")

        # Save metadata to file
        with open(output_path+'/'+experiment_name+'/'+'metadata.csv',mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["DATA LABEL", "VALUE"])
            writer.writerow(["class label",class_label])
            writer.writerow(["instance label", instance_label])
            writer.writerow(["match label", match_label])
            writer.writerow(["random state",random_state])
            writer.writerow(["categorical cutoff",categorical_cutoff])
            writer.writerow(["statistical significance cutoff",sig_cutoff])
            writer.writerow(["cv partitions",cv_partitions])
            writer.writerow(["partition method",partition_method])
            writer.writerow(["ignored features",ignore_features_path])
            writer.writerow(["specified categorical variables",categorical_feature_path])
        file.close()

    else: #run job checks
        datasets = os.listdir(output_path + "/" + experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.csv' in datasets:
            datasets.remove('metadata.csv')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')

        phase1Jobs = []
        for dataset in datasets:
            phase1Jobs.append('job_exploratory_'+dataset+'.txt')

        for filename in glob.glob(output_path + "/" + experiment_name+'/jobsCompleted/job_exploratory*'):
            ref = filename.split('/')[-1]
            phase1Jobs.remove(ref)
        for job in phase1Jobs:
            print(job)
        if len(phase1Jobs) == 0:
            print("All Phase 1 Jobs Completed")
        else:
            print("Above Phase 1 Jobs Not Completed")
        print()

def makeDirTree(data_path,output_path,experiment_name,jupyterRun):
    #Check to make sure data_path exists and experiment name is valid & unique
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    if os.path.exists(output_path+'/'+experiment_name):
        raise Exception("Experiment Name must be unique")

    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_':
            raise Exception('Experiment Name must be alphanumeric')

    #Create output folder if it doesn't already exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #Create Experiment folder, with log and job folders
    os.mkdir(output_path+'/'+experiment_name)
    os.mkdir(output_path+'/'+ experiment_name+'/jobsCompleted')
    if not eval(jupyterRun):
        os.mkdir(output_path+'/'+experiment_name+'/jobs')
        os.mkdir(output_path+'/'+experiment_name+'/logs')


def submitLocalJob(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun):
    ExploratoryAnalysisJob.job(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun)

def submitClusterJob(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,reserved_memory,maximum_memory,queue,ignore_features_path,categorical_feature_path,sig_cutoff,jupyterRun):
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P1_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P1_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P1_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ExploratoryAnalysisJob.py '+dataset_path+" "+experiment_path+" "+str(cv_partitions)+" "+partition_method+" "+str(categorical_cutoff)+" "+export_exploratory_analysis+
                  " "+export_feature_correlations+" "+export_univariate_plots+" "+class_label+" "+instance_label+" "+match_label+" "+str(random_state)+" "+str(ignore_features_path)+" "+str(categorical_feature_path)+" "+str(sig_cutoff)+" "+str(jupyterRun)+'\n')
    sh_file.close()
    os.system('bsub < '+job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
