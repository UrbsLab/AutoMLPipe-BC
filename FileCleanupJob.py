"""
File: FileCleanupJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 12 of AutoMLPipe-BC - This 'Job' script is called by FileCleanupMain.py which gathers key results files and copies them into a new
folder that can be more easily transfered and takes up less storage space. Includes metadata file, Dataset comparisons, along with results and basic
exploratory analysis files for each dataset analyzed. This runs once for the entire pipeline analysis.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
from distutils.dir_util import copy_tree
import sys
import os
import glob
import shutil

def job(experiment_path,data_path):
    """ Run all elements of key file copy once for the entire analysis pipeline: Deletes all unnecessary/temporary files and directories from pipeline output folder."""

    #Copy Dataset comparisons if present
    if os.path.exists(experiment_path+'/DatasetComparisons'):
        #Make corresponding data folder
        os.mkdir(experiment_path+'/KeyFileCopy'+'/DatasetComparisons')
        copy_tree(experiment_path+'/DatasetComparisons', experiment_path+'/KeyFileCopy'+'/DatasetComparisons')
    #Create dataset name folders
    for datasetFilename in glob.glob(data_path+'/*'):
        dataset_name = datasetFilename.split('/')[-1].split('.')[0]
        if not os.path.exists(experiment_path+'/KeyFileCopy'+ '/' + dataset_name):
            os.mkdir(experiment_path+'/KeyFileCopy'+ '/' + dataset_name)
            os.mkdir(experiment_path+'/KeyFileCopy'+ '/' + dataset_name+'/results')
            #copy respective results folder
            copy_tree(experiment_path+ '/' + dataset_name+'/training'+'/results/', experiment_path+'/KeyFileCopy'+ '/' + dataset_name+'/results/')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'ClassCounts.png', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'ClassCounts.png')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'ClassCounts.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'ClassCounts.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'DataCounts.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'DataCounts.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'DataMissingness.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'DataMissingness.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'DescribeDataset.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'DescribeDataset.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'DtypesDataset.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'DtypesDataset.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'NumUniqueDataset.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'NumUniqueDataset.csv')
    #Copy metafile
    shutil.copy(experiment_path+ '/metadata.csv',experiment_path+'/KeyFileCopy'+ '/metadata.csv')

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2])
