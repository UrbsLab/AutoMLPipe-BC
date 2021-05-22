
import argparse
import os
import sys
import pandas as pd
import FeatureSelectionJob
import time
import csv
import glob

'''Phase 4 of Machine Learning Analysis Pipeline:
Sample Run Command:
python FeatureSelectionMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1

Local Command:
python FeatureSelectionMain.py --output-path /Users/robert/Desktop/outputs --experiment-name randomtest2 --run-parallel False
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--out-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--max-feat', dest='max_features_to_keep', type=int,help='max features to keep. None if no max', default=2000)
    parser.add_argument('--filter-feat', dest='filter_poor_features', type=str, help='filter out the worst performing features prior to modeling',default='True')
    parser.add_argument('--top-results', dest='top_results', type=int,help='number of top features to illustrate in figures', default=20)
    parser.add_argument('--export-scores', dest='export_scores', type=str,help='export figure summarizing average feature importance scores over cv partitions', default='True')
    parser.add_argument('--over-cv', dest='overwrite_cv',type=str,help='overwrites working cv datasets with new feature subset datasets',default="True")
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    max_features_to_keep = options.max_features_to_keep
    filter_poor_features = options.filter_poor_features
    top_results = options.top_results
    export_scores = options.export_scores
    overwrite_cv = options.overwrite_cv
    run_parallel = options.run_parallel
    queue = options.queue
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory
    do_check = options.do_check

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    cv_partitions = int(metadata[6,1])
    do_mutual_info = metadata[12,1]
    do_multisurf = metadata[13,1]
    jupyterRun = 'False'

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 4 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 4 can begin")

    if not do_check:
        dataset_paths = os.listdir(output_path + "/" + experiment_name)
        dataset_paths.remove('logs')
        dataset_paths.remove('jobs')
        dataset_paths.remove('jobsCompleted')
        dataset_paths.remove('metadata.csv')
        for dataset_directory_path in dataset_paths:
            full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
            if eval(run_parallel):
                submitClusterJob(full_path,output_path+'/'+experiment_name,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,reserved_memory,maximum_memory,queue,jupyterRun)
            else:
                submitLocalJob(full_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,jupyterRun)

        #Update metadata
        if metadata.shape[0] == 17: #Only update if metadata below hasn't been added before
            with open(output_path + '/' + experiment_name + '/' + 'metadata.csv',mode='a', newline="") as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["max features to keep",max_features_to_keep])
                writer.writerow(["filter poor features", filter_poor_features])
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

        phase4Jobs = []
        for dataset in datasets:
            phase4Jobs.append('job_featureselection_' + dataset + '.txt')

        for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_featureselection*'):
            ref = filename.split('/')[-1]
            phase4Jobs.remove(ref)
        for job in phase4Jobs:
            print(job)
        if len(phase4Jobs) == 0:
            print("All Phase 4 Jobs Completed")
        else:
            print("Above Phase 4 Jobs Not Completed")
        print()

def submitLocalJob(full_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,jupyterRun):
    FeatureSelectionJob.job(full_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,jupyterRun)

def submitClusterJob(full_path,experiment_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,reserved_memory,maximum_memory,queue,jupyterRun):
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P4_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P4_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P4_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/FeatureSelectionJob.py '+full_path+" "+do_mutual_info+" "+do_multisurf+" "+
                  str(max_features_to_keep)+" "+filter_poor_features+" "+str(top_results)+" "+export_scores+" "+class_label+" "+instance_label+" "+str(cv_partitions)+" "+overwrite_cv+" "+jupyterRun+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
