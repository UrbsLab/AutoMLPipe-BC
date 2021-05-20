import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import copy
import pandas as pd
import os
import csv
import sys

def job(full_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv,jupyterRun):
    job_start_time = time.time()
    dataset_name = full_path.split('/')[-1]
    selected_feature_lists = {}
    meta_feature_ranks = {}
    algorithms = []

    #Mutual Information
    if eval(do_mutual_info):
        algorithms.append('Mutual Information')
        selected_feature_lists,meta_feature_ranks = reportAveFS("Mutual Information","mutualinformation",cv_partitions,top_results,full_path,selected_feature_lists,meta_feature_ranks,export_scores,jupyterRun)

    #MultiSURF
    if eval(do_multisurf):
        algorithms.append('MultiSURF')
        selected_feature_lists,meta_feature_ranks = reportAveFS("MultiSURF","multisurf",cv_partitions,top_results,full_path,selected_feature_lists,meta_feature_ranks,export_scores,jupyterRun)

    if len(algorithms) != 0:
        #Feature Selection
        if eval(filter_poor_features):
            #Identify top feature subset
            cv_selected_list = selectFeatures(algorithms,cv_partitions,selected_feature_lists,max_features_to_keep,meta_feature_ranks)

            #Generate new datasets with selected feature subsets
            genFilteredDatasets(cv_selected_list,class_label,instance_label,cv_partitions,full_path+'/CVDatasets',dataset_name,overwrite_cv)

    saveRuntime(full_path,job_start_time)

    # Print completion
    print(dataset_name + " phase 4 complete")
    experiment_path = '/'.join(full_path.split('/')[:-1])
    job_file = open(experiment_path + '/jobsCompleted/job_featureselection_' + dataset_name + '.txt', 'w')
    job_file.write('complete')
    job_file.close()

def saveRuntime(full_path,job_start_time):
    # Save Runtime
    runtime_file = open(full_path + '/runtime/runtime_featureselection.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

def reportAveFS(algorithm,algorithmlabel,cv_partitions,top_results,full_path,selected_feature_lists,meta_feature_ranks,export_scores,jupyterRun):
    #Calculate score sums
    counter = 0
    cv_keep_list = []
    feature_name_ranks = []
    for i in range(0,cv_partitions):
        scoreInfo = full_path+"/"+algorithmlabel+"/pickledForPhase4/"+str(i)
        file = open(scoreInfo, 'rb')
        rawData = pickle.load(file)
        file.close()

        scoreDict = rawData[1]
        score_sorted_features = rawData[2]
        feature_name_ranks.append(score_sorted_features)

        if counter == 0:
            scoreSum = copy.deepcopy(scoreDict)
        else:
            for each in rawData[1]:
                scoreSum[each] += scoreDict[each]
        counter += 1

        keep_list = []
        for each in scoreDict:
            if scoreDict[each] > 0:
                keep_list.append(each)
        cv_keep_list.append(keep_list)
    selected_feature_lists[algorithm] = cv_keep_list
    meta_feature_ranks[algorithm] = feature_name_ranks

    #Generate barplot of average scores
    if eval(export_scores):
        # Make the sum of scores an average
        for v in scoreSum:
            scoreSum[v] = scoreSum[v] / float(cv_partitions)

        # Sort averages (decreasing order and print top 'n' and plot top 'n'
        f_names = []
        f_scores = []
        for each in scoreSum:
            f_names.append(each)
            f_scores.append(scoreSum[each])

        names_scores = {'Names': f_names, 'Scores': f_scores}
        ns = pd.DataFrame(names_scores)
        ns = ns.sort_values(by='Scores', ascending=False)

        # Select top 'n' to report and plot
        ns = ns.head(top_results)

        # Visualize sorted feature scores
        ns['Scores'].plot(kind='barh', figsize=(6, 12))
        plt.ylabel('Features')
        plt.xlabel(str(algorithm) + ' Score')
        plt.yticks(np.arange(len(ns['Names'])), ns['Names'])
        plt.title('Sorted ' + str(algorithm) + ' Scores')
        plt.savefig((full_path+"/"+algorithmlabel+"/TopAverageScores.png"), bbox_inches="tight")
        if eval(jupyterRun):
            plt.show()
        else:
            plt.close('all')

    return selected_feature_lists,meta_feature_ranks

def selectFeatures(algorithms, cv_partitions, selectedFeatureLists, maxFeaturesToKeep, metaFeatureRanks):
    cv_Selected_List = []  # list of selected features for each cv
    numAlgorithms = len(algorithms)
    if numAlgorithms > 1:  # 'Interesting' features determined by union of feature selection results (from different algorithms)
        for i in range(cv_partitions):
            unionList = selectedFeatureLists[algorithms[0]][i]  # grab first algorithm's lists
            # Determine union
            for j in range(1, numAlgorithms):  # number of union comparisons
                unionList = list(set(unionList) | set(selectedFeatureLists[algorithms[j]][i]))

            if len(unionList) > maxFeaturesToKeep:  # Apply further filtering if more than max features remains
                # Create score list dictionary with indexes in union list
                newFeatureList = []
                k = 0
                while len(newFeatureList) < maxFeaturesToKeep:
                    for each in metaFeatureRanks:
                        targetFeature = metaFeatureRanks[each][i][k]
                        if not targetFeature in newFeatureList:
                            newFeatureList.append(targetFeature)
                        if len(newFeatureList) < maxFeaturesToKeep:
                            break
                    k += 1
                unionList = newFeatureList
            unionList.sort()  # Added to ensure script random seed reproducibility
            cv_Selected_List.append(unionList)

    else:  # Only one algorithm applied
        for i in range(cv_partitions):
            featureList = selectedFeatureLists[algorithms[0]][i]  # grab first algorithm's lists

            if len(featureList) > maxFeaturesToKeep:  # Apply further filtering if more than max features remains
                # Create score list dictionary with indexes in union list
                newFeatureList = []
                k = 0
                while len(newFeatureList) < maxFeaturesToKeep:
                    targetFeature = metaFeatureRanks[algorithms[0]][i][k]
                    newFeatureList.append(targetFeature)
                    k += 1
                featureList = newFeatureList
            cv_Selected_List.append(featureList)

    return cv_Selected_List

def genFilteredDatasets(cv_selected_list,class_label,instance_label,cv_partitions,path_to_csv,dataset_name,overwrite_cv):
    #create lists to hold training and testing set dataframes.
    trainList = []
    testList = []

    for i in range(cv_partitions):
        #Load training partition
        trainSet = pd.read_csv(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv", na_values='NA', sep = ",")
        trainList.append(trainSet)

        #Load testing partition
        testSet = pd.read_csv(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Test.csv", na_values='NA', sep = ",")
        testList.append(testSet)

        #Training datasets
        labelList = [class_label]
        if instance_label != 'None':
            labelList.append(instance_label)
        labelList = labelList + cv_selected_list[i]

        td_train = trainList[i][labelList]
        td_test = testList[i][labelList]

        if eval(overwrite_cv):
            #Remove old CV files
            os.remove(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv")
            os.remove(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Test.csv")
        else:
            #Rename old CV files
            os.rename(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv",path_to_csv+'/'+dataset_name+'_CVPre_' + str(i) +"_Train.csv")
            os.rename(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Test.csv",path_to_csv+'/'+dataset_name+'_CVPre_' + str(i) +"_Test.csv")

        #Write new CV files
        with open(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv",mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(td_train.columns.values.tolist())
            for row in td_train.values:
                writer.writerow(row)
        file.close()

        with open(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Test.csv",mode='w', newline="") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(td_test.columns.values.tolist())
            for row in td_test.values:
                writer.writerow(row)
        file.close()

if __name__ == '__main__':
    job(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]),sys.argv[7], sys.argv[8],sys.argv[9],int(sys.argv[10]),sys.argv[11],sys.argv[12])
