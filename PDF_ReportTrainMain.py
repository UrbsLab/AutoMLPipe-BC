"""
File: PDF_ReportTrainMain.py
Authors: Ryan J. Urbanowicz, Richard Zhang, Wilson Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 9 of AutoMLPipe-BC (Optional)- This 'Main' script manages Phase 9 run parameters, and submits job to run locally (to run serially) or on a linux computing
             cluster (parallelized). This script runs PDF_ReportTrainJob.py which generates a formatted PDF summary report of key pipeline results.
             All 'Main' scripts in this pipeline have the potential to be extended by users to submit jobs to other parallel computing frameworks (e.g. cloud computing).
Warnings: Designed to be run following the completion of either AutoMLPipe-BC Phase 6 (StatsMain.py), Phase 7 (DataCompareMain.py), and or Phase 8 (KeyFileCopyMain.py).
Sample Run Command (Linux cluster parallelized with all default run parameters):
    python PDF_ReportTrainMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1
Sample Run Command (Local/serial with with all default run parameters):
    python PDF_ReportTrainMain.py --out-path /Users/robert/Desktop/outputs --exp-name myexperiment1 --run-parallel False
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import os
import re
import sys
import argparse
import time

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--out-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--exp-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of parallel computing queue (uses our research groups queue by default)',default="i2c2_normal")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)

    options = parser.parse_args(argv[1:])

    experiment_path = options.output_path+'/'+options.experiment_name

    if eval(options.run_parallel):
        submitClusterJob(experiment_path,options.reserved_memory,options.maximum_memory,options.queue)
    else:
        submitLocalJob(experiment_path)

def submitLocalJob(experiment_path):
    """ Runs PDF_ReportTrainJob.py locally, once. """
    KeyFileCopyJob.job(experiment_path)

def submitClusterJob(experiment_path,reserved_memory,maximum_memory,queue):
    """ Runs PDF_ReportTrainJob.py once. Runs on a linux-based computing cluster that uses an IBM Spectrum LSF for job scheduling."""
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/PDF_Train_' + job_ref + '_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q '+queue+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/PDF_Train_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/PDF_Train_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/PDF_ReportTrainJob.py ' + experiment_path + '\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
