#######################################
## Richard Zhang, Wilson Zhang, & Ryan Urbanowicz    ##
## March 30, 2021                    ##
## ML Pipeline Report Generator V. 1 ##
## Requirements: pip install fpdf
#######################################

import os
import re
import sys
import argparse
import time

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--output-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')
    #Lostistical arguments
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='if run parallel',default="True")
    parser.add_argument('--queue',dest='queue',type=str,help='specify name of LPC queue',default="i2c2_normal") #specific to our research institution and computing cluster
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name

    run_parallel = options.run_parallel
    queue = options.queue
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory

    experiment_path = output_path+'/'+experiment_name

    if eval(run_parallel):
        submitClusterJob(experiment_path,reserved_memory,maximum_memory,queue)
    else:
        submitLocalJob(experiment_path)

def submitLocalJob(experiment_path):
    KeyFileCopyJob.job(experiment_path)

def submitClusterJob(experiment_path,reserved_memory,maximum_memory,queue):
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
