
import argparse
import os
import sys
import pandas as pd
import numpy as np
import glob

'''
python RandomSeedValidator.py --o /Users/robert/Desktop/outputs --e1 randomtest1 --e2 randomtest2 --cv 3
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--e1', dest='e1', type=str, help='path to first completed experiment')
    parser.add_argument('--e2', dest='e2', type=str, help='path to second completed experiment')
    parser.add_argument('--cv', dest='cv_count', type=int, help='# CVs used in both experiments')

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment1 = options.e1
    experiment2 = options.e2
    cv_count = options.cv_count

    e1_path = output_path + '/' + experiment1
    e2_path = output_path + '/' + experiment2

    algorithms = ['LR','DT','RF','NB','XGB','LGB','SVM','ANN','ExSTraCS','eLCS','XCS','GB','KN']

    dataset_paths = os.listdir(e1_path)
    dataset_paths.remove('logs')
    dataset_paths.remove('jobs')
    dataset_paths.remove('jobsCompleted')
    dataset_paths.remove('metadata.csv')
    for dataset_name in dataset_paths:
        print('Starting Analysis for '+dataset_name)
        d1_path = e1_path + "/" + dataset_name
        d2_path = e2_path + "/" + dataset_name

        # Check CV Datasets (no cleaning, post scaling/imputation, post feature selection) #############################
        cv_equal = True
        for cv in range(cv_count):
            train_path1 = d1_path + '/CVDatasets/'+dataset_name+'_CV_'+str(cv)+'_Train.csv'
            test_path1 = d1_path + '/CVDatasets/' + dataset_name + '_CV_' + str(cv) + '_Test.csv'
            train_path1a = d1_path + '/CVDatasets/' + dataset_name + '_CVOnly_' + str(cv) + '_Train.csv'
            test_path1a = d1_path + '/CVDatasets/' + dataset_name + '_CVOnly_' + str(cv) + '_Test.csv'
            train_path1b = d1_path + '/CVDatasets/' + dataset_name + '_CVPre_' + str(cv) + '_Train.csv'
            test_path1b = d1_path + '/CVDatasets/' + dataset_name + '_CVPre_' + str(cv) + '_Test.csv'
            list1 = [train_path1,test_path1,train_path1a,test_path1a,train_path1b,test_path1b]

            train_path2 = d2_path + '/CVDatasets/' + dataset_name + '_CV_' + str(cv) + '_Train.csv'
            test_path2 = d2_path + '/CVDatasets/' + dataset_name + '_CV_' + str(cv) + '_Test.csv'
            train_path2a = d2_path + '/CVDatasets/' + dataset_name + '_CVOnly_' + str(cv) + '_Train.csv'
            test_path2a = d2_path + '/CVDatasets/' + dataset_name + '_CVOnly_' + str(cv) + '_Test.csv'
            train_path2b = d2_path + '/CVDatasets/' + dataset_name + '_CVPre_' + str(cv) + '_Train.csv'
            test_path2b = d2_path + '/CVDatasets/' + dataset_name + '_CVPre_' + str(cv) + '_Test.csv'
            list2 = [train_path2, test_path2, train_path2a, test_path2a, train_path2b, test_path2b]

            for d in range(len(list1)):
                c1 = pd.read_csv(list1[d]).values
                c2 = pd.read_csv(list2[d]).values
                if not np.array_equal(c1,c2):
                    cv_equal = False
        if cv_equal:
            print('CV Datasets are equal\n')
        else:
            print('CV Datasets are not equal\n')


        # Check FI Scores from MI and MS ###############################################################################
        fi_equal = True
        for cv in range(cv_count):
            ms_path1 = d1_path + '/multisurf/scores_cv_'+str(cv)+'.csv'
            mi_path1 = d1_path + '/mutualinformation/scores_cv_' + str(cv) + '.csv'
            ms_path2 = d2_path + '/multisurf/scores_cv_' + str(cv) + '.csv'
            mi_path2 = d2_path + '/mutualinformation/scores_cv_' + str(cv) + '.csv'
            list1 = [ms_path1,mi_path1]
            list2 = [ms_path2,mi_path2]

            for d in range(len(list1)):
                c1 = pd.read_csv(list1[d]).values
                c2 = pd.read_csv(list2[d]).values
                if not np.allclose(c1,c2):
                    fi_equal = False

        if fi_equal:
            print('Feature Importance Scores are equal\n')
        else:
            print('Feature Importance Scores are not equal\n')

        # Check Hyperparameter Optimization Results ####################################################################
        hyper_equal = True
        for cv in range(cv_count):
            list1 = []
            list2 = []
            for algo in algorithms:
                if os.path.exists(d1_path+'/training/'+str(algo)+'_usedparams'+str(cv)+'.csv'):
                    bestparams_path1 = d1_path+'/training/'+str(algo)+'_usedparams'+str(cv)+'.csv'
                    bestparams_path2 = d2_path + '/training/' + str(algo) + '_usedparams' + str(cv) + '.csv'
                    list1.append(bestparams_path1)
                    list2.append(bestparams_path2)
                elif os.path.exists(d1_path+'/training/'+str(algo)+'_bestparams'+str(cv)+'.csv'):
                    bestparams_path1 = d1_path + '/training/' + str(algo) + '_bestparams' + str(cv) + '.csv'
                    bestparams_path2 = d2_path + '/training/' + str(algo) + '_bestparams' + str(cv) + '.csv'
                    list1.append(bestparams_path1)
                    list2.append(bestparams_path2)

            for d in range(len(list1)):
                c1 = pd.read_csv(list1[d])
                c2 = pd.read_csv(list2[d])
                if not c1.equals(c2):
                    hyper_equal = False

        if hyper_equal:
            print('Hyperparameter Optimization Chosen Params are equal\n')
        else:
            print('Hyperparameter Optimization Chosen Params are not equal\n')

        # Check Algorithm Performances #################################################################################
        perform_equal = True
        list1 = []
        list2 = []
        for algo in algorithms:
            if os.path.exists(d1_path + '/training/results/' + str(algo) + '_performance.csv') and os.path.exists(d2_path + '/training/results/' + str(algo) + '_performance.csv'):
                print('Checking '+algo)
                perform_path1 = d1_path + '/training/results/' + str(algo) + '_performance.csv'
                perform_path2 = d2_path + '/training/results/' + str(algo) + '_performance.csv'
                list1.append(perform_path1)
                list2.append(perform_path2)

        for d in range(len(list1)):
            c1 = pd.read_csv(list1[d]).values
            c2 = pd.read_csv(list2[d]).values
            if not np.array_equal(c1, c2):
                perform_equal = False

        if perform_equal:
            print('Model Performances are equal\n')
        else:
            print('Model Performances are not equal\n')

        # Check Algorithm FIs ##########################################################################################
        modelfi_equal = True
        list1 = []
        list2 = []
        for algo in algorithms:
            if os.path.exists(d1_path + '/training/results/FI/' + str(algo) + '_FI.csv') and os.path.exists(d2_path + '/training/results/FI/' + str(algo) + '_FI.csv'):
                print('Checking ' + algo)
                perform_path1 = d1_path + '/training/results/FI/' + str(algo) + '_FI.csv'
                perform_path2 = d2_path + '/training/results/FI/' + str(algo) + '_FI.csv'
                list1.append(perform_path1)
                list2.append(perform_path2)

        for d in range(len(list1)):
            c1 = pd.read_csv(list1[d]).values
            c2 = pd.read_csv(list2[d]).values
            if not np.array_equal(c1, c2):
                modelfi_equal = False

        if modelfi_equal:
            print('Model FIs are equal\n')
        else:
            print('Model FIs are not equal\n')

        # Check if max # of DIVE clusters is the same ##################################################################
        at_count1 = 0
        for _ in glob.glob(d1_path + '/viz-outputs/root/Composite/at/atclusters/*_clusters'):
            at_count1 += 1
        at_count2 = 0
        for _ in glob.glob(d2_path + '/viz-outputs/root/Composite/at/atclusters/*_clusters'):
            at_count2 += 1

        rule_count1 = 0
        for _ in glob.glob(d1_path + '/viz-outputs/root/Composite/rulepop/ruleclusters/*_clusters'):
            rule_count1 += 1
        rule_count2 = 0
        for _ in glob.glob(d2_path + '/viz-outputs/root/Composite/rulepop/ruleclusters/*_clusters'):
            rule_count2 += 1

        if at_count1 == at_count2:
            print('AT cluster count is the same')
        else:
            print('AT cluster count is not the same')

        if rule_count1 == rule_count2:
            print('Rule cluster count is the same')
        else:
            print('Rule cluster count is not the same')

        # Check DIVE optimal clusters match ############################################################################
        at_count1 = None
        for fp in glob.glob(d1_path + '/viz-outputs/root/Composite/at/*optimalClusters.png'):
            at_count1 = fp.split('/')[-1].split('o')[0]
        at_count2 = None
        for fp in glob.glob(d2_path + '/viz-outputs/root/Composite/at/*optimalClusters.png'):
            at_count2 = fp.split('/')[-1].split('o')[0]

        rule_count1 = None
        for fp in glob.glob(d1_path + '/viz-outputs/root/Composite/rulepop/*optimalClusters.png'):
            rule_count1 = fp.split('/')[-1].split('o')[0]
        rule_count2 = None
        for fp in glob.glob(d2_path + '/viz-outputs/root/Composite/rulepop/*optimalClusters.png'):
            rule_count2 = fp.split('/')[-1].split('o')[0]

        if at_count1 == at_count2:
            print('AT optimal cluster count is the same')
        else:
            print('AT optimal cluster count is not the same')

        if rule_count1 == rule_count2:
            print('Rule optimal cluster count is the same')
        else:
            print('Rule optimal cluster count is not the same')

if __name__ == '__main__':
    sys.exit(main(sys.argv))