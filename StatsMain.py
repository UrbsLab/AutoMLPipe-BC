
import argparse
import os
import sys
import time
import pandas as pd
import StatsJob
import glob

'''Phase 6 of Machine Learning Analysis Pipeline:
Sample Run Command:
python StatsMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1

Local Command:
python StatsMain.py --output-path /Users/robert/Desktop/outputs --experiment-name randomtest2 --run-parallel False
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--plot-ROC', dest='plot_ROC', type=str,help='Plot ROC curves individually for each algorithm including all CV results and averages', default='True')
    parser.add_argument('--plot-PRC', dest='plot_PRC', type=str,help='Plot PRC curves individually for each algorithm including all CV results and averages', default='True')
    parser.add_argument('--plot-box', dest='plot_metric_boxplots', type=str,help='Plot box plot summaries comparing algorithms for each metric', default='True')
    parser.add_argument('--plot-FI_box', dest='plot_FI_box', type=str,help='Plot feature importance boxplots for each algorithm', default='True')
    parser.add_argument('--top-results', dest='top_results', type=int,help='number of top features to illustrate in figures', default=20)
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of LPC queue',default="i2c2_normal") #specific to our research institution and computing cluster
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    plot_ROC = options.plot_ROC
    plot_PRC = options.plot_PRC
    plot_metric_boxplots = options.plot_metric_boxplots
    plot_FI_box = options.plot_FI_box
    top_results = options.top_results

    run_parallel = options.run_parallel
    queue = options.queue
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory
    do_check = options.do_check

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 6 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 6 can begin")

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    cv_partitions = int(metadata[5,1])
    sig_cutoff = metadata[4,1]

    do_LR = metadata[17,1]
    do_DT = metadata[18,1]
    do_RF = metadata[19,1]
    do_NB = metadata[20,1]
    do_XGB = metadata[21,1]
    do_LGB = metadata[22,1]
    do_SVM = metadata[23,1]
    do_ANN = metadata[24,1]
    do_ExSTraCS = metadata[25,1]
    do_eLCS = metadata[26,1]
    do_XCS = metadata[27,1]
    do_GB = metadata[28, 1]
    do_KN = metadata[29, 1]
    primary_metric = metadata[30,1]

    encodedAlgos = ''
    encodedAlgos = encode(do_LR,encodedAlgos)
    encodedAlgos = encode(do_DT, encodedAlgos)
    encodedAlgos = encode(do_RF, encodedAlgos)
    encodedAlgos = encode(do_NB, encodedAlgos)
    encodedAlgos = encode(do_XGB, encodedAlgos)
    encodedAlgos = encode(do_LGB, encodedAlgos)
    encodedAlgos = encode(do_ANN, encodedAlgos)
    encodedAlgos = encode(do_SVM, encodedAlgos)
    encodedAlgos = encode(do_ExSTraCS, encodedAlgos)
    encodedAlgos = encode(do_eLCS, encodedAlgos)
    encodedAlgos = encode(do_XCS, encodedAlgos)
    encodedAlgos = encode(do_GB, encodedAlgos)
    encodedAlgos = encode(do_KN, encodedAlgos)

    if not do_check:
        # Iterate through datasets
        dataset_paths = os.listdir(output_path + "/" + experiment_name)
        dataset_paths.remove('logs')
        dataset_paths.remove('jobs')
        dataset_paths.remove('jobsCompleted')
        dataset_paths.remove('metadata.csv')
        for dataset_directory_path in dataset_paths:
            full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
            if eval(run_parallel):
                submitClusterJob(full_path,encodedAlgos,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,output_path+'/'+experiment_name,cv_partitions,reserved_memory,maximum_memory,queue,plot_metric_boxplots,primary_metric,top_results,sig_cutoff)
            else:
                submitLocalJob(full_path,encodedAlgos,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,cv_partitions,plot_metric_boxplots,primary_metric,top_results,sig_cutoff)
    else:
        datasets = os.listdir(output_path + "/" + experiment_name)
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

        for filename in glob.glob(output_path + "/" + experiment_name+'/jobsCompleted/job_stats*'):
            ref = filename.split('/')[-1]
            phase6Jobs.remove(ref)
        for job in phase6Jobs:
            print(job)
        if len(phase6Jobs) == 0:
            print("All Phase 6 Jobs Completed")
        else:
            print("Above Phase 6 Jobs Not Completed")
        print()

def submitLocalJob(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,cv_partitions,plot_metric_boxplots,primary_metric,top_results,sig_cutoff):
    StatsJob.job(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,cv_partitions,plot_metric_boxplots,primary_metric,top_results,sig_cutoff)

def submitClusterJob(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI_box,class_label,instance_label,experiment_path,cv_partitions,reserved_memory,maximum_memory,queue,plot_metric_boxplots,primary_metric,top_results,sig_cutoff):
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
    sh_file.write('python '+this_file_path+'/StatsJob.py '+full_path+" "+encoded_algos+" "+plot_ROC+" "+plot_PRC+" "+plot_FI_box+" "+class_label+" "+instance_label+" "+str(cv_partitions)+" "+str(plot_metric_boxplots)+" "+str(primary_metric)+" "+str(top_results)+" "+str(sig_cutoff)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

def encode(do_algo,encodedAlgos):
    if eval(do_algo):
        encodedAlgos += '1'
    else:
        encodedAlgos += '0'
    return encodedAlgos

if __name__ == '__main__':
    sys.exit(main(sys.argv))
