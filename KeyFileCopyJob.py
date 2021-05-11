from distutils.dir_util import copy_tree
import sys
import os
import glob
import shutil

def job(experiment_path,data_path):

    #Create copied file summary folder
    os.mkdir(experiment_path+'/KeyFileCopy')

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
            #Copy class balance
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'ClassCounts.png', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'ClassCounts.png')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'ClassCounts.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'ClassCounts.csv')

            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'DataCounts.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'DataCounts.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'DescribeDataset.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'DescribeDataset.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'DtypesDataset.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'DtypesDataset.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'FeatureMissingness.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'FeatureMissingness.csv')
            shutil.copy(experiment_path+ '/' + dataset_name+'/exploratory/'+'NumUniqueDataset.csv', experiment_path+'/KeyFileCopy'+ '/' + dataset_name +'/' +'NumUniqueDataset.csv')
    #Copy metafile
    shutil.copy(experiment_path+ '/metadata.csv',experiment_path+'/KeyFileCopy'+ '/metadata.csv')

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2])
