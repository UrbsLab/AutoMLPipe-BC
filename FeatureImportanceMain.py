
import argparse
import os
import sys
import glob
import FeatureImportanceJob
import time
import pandas as pd
import csv

'''Phase 3 of Machine Learning Analysis Pipeline:
Sample Run Command:
python FeatureImportanceMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1

Local Command:
python FeatureImportanceMain.py --output-path /Users/robert/Desktop/outputs --experiment-name randomtest2 --run-parallel False
'''

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

    output_path = options.output_path
    experiment_name = options.experiment_name
    do_mutual_info = options.do_mutual_info
    do_multisurf = options.do_multisurf
    use_TURF = options.use_TURF
    TURF_pct = options.TURF_pct
    n_jobs = options.n_jobs
    instance_subset = options.instance_subset
    run_parallel = options.run_parallel
    queue = options.queue
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory
    do_check = options.do_check

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 3 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 3 can begin")

    metadata = pd.read_csv(output_path+'/'+experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1,1]
    random_state = int(metadata[3, 1])
    categorical_cutoff = int(metadata[4,1])
    cv_partitions = int(metadata[6,1])

    if not do_check:
        #Iterate through datasets, ignoring common folders
        dataset_paths = os.listdir(output_path+"/"+experiment_name)
        dataset_paths.remove('logs')
        dataset_paths.remove('jobs')
        dataset_paths.remove('jobsCompleted')
        dataset_paths.remove('metadata.csv')
        for dataset_directory_path in dataset_paths:
            full_path = output_path+"/"+experiment_name+"/"+dataset_directory_path
            experiment_path = output_path+'/'+experiment_name

            if eval(do_mutual_info):
                if not os.path.exists(full_path+"/mutualinformation"):
                    os.mkdir(full_path+"/mutualinformation")
                for cv_train_path in glob.glob(full_path+"/CVDatasets/*_CV_*Train.csv"):
                    command_text = '/FeatureImportanceJob.py ' + cv_train_path+" "+experiment_path+" "+str(random_state)+" "+class_label+" "+instance_label+" " +str(instance_subset)+" mi "+str(n_jobs)+' '+str(use_TURF)+' '+str(TURF_pct)
                    if eval(run_parallel):
                        submitClusterJob(command_text, experiment_path,reserved_memory,maximum_memory,queue)
                    else:
                        submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,instance_subset,'mi',n_jobs,use_TURF,TURF_pct)

            if eval(do_multisurf):
                if not os.path.exists(full_path+"/multisurf"):
                    os.mkdir(full_path+"/multisurf")
                for cv_train_path in glob.glob(full_path+"/CVDatasets/*_CV_*Train.csv"):
                    command_text = '/FeatureImportanceJob.py ' + cv_train_path+" "+experiment_path+" "+str(random_state)+" "+class_label+" "+instance_label+" " +str(instance_subset)+" ms "+str(n_jobs)+' '+str(use_TURF)+' '+str(TURF_pct)
                    if eval(run_parallel):
                        submitClusterJob(command_text, experiment_path,reserved_memory,maximum_memory,queue)
                    else:
                        submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,instance_subset,'ms',n_jobs,use_TURF,TURF_pct)

        #Update metadata
        if metadata.shape[0] == 12: #Only update if metadata below hasn't been added before (i.e. in a previous phase 2 run)
            with open(output_path + '/' + experiment_name + '/' + 'metadata.csv',mode='a', newline="") as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["mutual information",do_mutual_info])
                writer.writerow(["MultiSURF", do_multisurf])
                writer.writerow(["TURF",use_TURF])
                writer.writerow(["TURF cutoff", TURF_pct])
                writer.writerow(["MultiSURF instance subset", instance_subset])
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

        phase3Jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                if eval(do_multisurf):
                    phase3Jobs.append('job_multisurf_' + dataset + '_' + str(cv) + '.txt')
                if eval(do_mutual_info):
                    phase3Jobs.append('job_mutualinformation_' + dataset + '_' + str(cv) + '.txt')

        for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_mu*'):
            ref = filename.split('/')[-1]
            phase3Jobs.remove(ref)
        for job in phase3Jobs:
            print(job)
        if len(phase3Jobs) == 0:
            print("All Phase 3 Jobs Completed")
        else:
            print("Above Phase 3 Jobs Not Completed")
        print()

def submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,instance_subset,algorithm,n_jobs,use_TURF,TURF_pct):
    FeatureImportanceJob.job(cv_train_path,experiment_path,random_state,class_label,instance_label,instance_subset,algorithm,n_jobs,use_TURF,TURF_pct)

def submitClusterJob(command_text,experiment_path,reserved_memory,maximum_memory,queue):
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
