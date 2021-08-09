"""
File: DataCompareJob.py
Authors: Ryan J. Urbanowicz, Robert Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 7 of AutoMLPipe-BC - This 'Job' script is called by DataCompareMain.py which runs non-parametric statistical analysis
comparing ML algorithm performance between all target datasets included in the original Phase 1 data folder, for each evaluation metric.
Also compares the best overall model for each target dataset, for each evaluation metric. This runs once for the entire pipeline analysis.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import os
import sys
import glob
import pandas as pd
from scipy import stats
import copy

def job(experiment_path,sig_cutoff):
    """ Run all elements of data comparison once for the entire analysis pipeline: runs non-parametric statistical analysis
    comparing ML algorithm performance between all target datasets included in the original Phase 1 data folder, for each
    evaluation metric. Also compares the best overall model for each target dataset, for each evaluation metric.
"""
    # Get dataset paths for all completed dataset analyses in experiment folder
    datasets = os.listdir(experiment_path)
    datasets.remove('metadata.csv')
    datasets.remove('jobsCompleted')
    try:
        datasets.remove('logs')
        datasets.remove('jobs')
    except:
        pass
    try:
        datasets.remove('DatasetComparisons') #If it has been run previously (overwrite)
    except:
        pass
    datasets = sorted(datasets) #ensures consistent ordering of datasets and assignment of temporary identifier
    dataset_directory_paths = []
    for dataset in datasets:
        full_path = experiment_path + "/" + dataset
        dataset_directory_paths.append(full_path)
    # Get ML modeling algorithms that were applied in analysis pipeline
    algorithms = []
    name_to_abbrev = {'naive_bayes': 'NB','logistic_regression': 'LR', 'decision_tree': 'DT', 'random_forest': 'RF','gradient_boosting':'GB',
                      'XGB': 'XGB', 'LGB': 'LGB', 'SVM': 'SVM', 'ANN': 'ANN','k_neighbors':'KN', 'eLCS': 'eLCS',
                      'XCS': 'XCS', 'ExSTraCS': 'ExSTraCS'}
    abbrev_to_name = dict([(value, key) for key, value in name_to_abbrev.items()])
    for filepath in glob.glob(dataset_directory_paths[0] + '/training/pickledModels/*'):
        filepath = str(filepath).replace('\\','/')
        algo_name = abbrev_to_name[filepath.split('/')[-1].split('_')[0]]
        if not algo_name in algorithms:
            algorithms.append(algo_name)
    # Get all mean evaluation metric data for all algorithms.
    data = pd.read_csv(dataset_directory_paths[0] + '/training/results/Summary_performance_mean.csv', sep=',')
    metrics = data.columns.values.tolist()[1:]
    # Create directory to store dataset statistical comparisons
    if not os.path.exists(experiment_path+'/DatasetComparisons'):
        os.mkdir(experiment_path+'/DatasetComparisons')
    #Run Kruscall Wallis test (for each algorithm) to determine if there was a significant difference in any metric performance between all analyzed datasets
    kruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)
    #Run MannWhitney U test (for each algorithm) to determine if there was a significant difference between any pair of datasets for any metric. Runs for all pairs even if kruscall wallis not significant for given metric.
    mannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)
    #Run Wilcoxon Rank Sum test (for each algorithm) to determine if there was a significant difference between any pair of datasets for any metric. Runs for all pairs even if kruscall wallis not significant for given metric.
    wilcoxonRank(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)
    #Run Kruscall Wallist test for each metric comparing all datasets using the best performing algorithm (based on given metric).
    global_data = bestKruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)
    #Run MannWhitney U test for each metric comparing pairs of datsets using the best performing algorithm (based on given metric).
    bestMannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data)
    #Run Wilcoxon Rank sum test for each metric comparing pairs of datsets using the best performing algorithm (based on given metric).
    bestWilcoxonRank(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data)
    print("Phase 7 complete")

def kruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    """ For each algorithm apply non-parametric Kruskal Wallis one-way ANOVA on ranks. Determines if there is a statistically significant difference in performance between original target datasets across CV runs.
    Completed for each standard metric separately."""
    label = ['Statistic', 'P-Value', 'Sig(*)']
    i = 1
    for dataset in datasets:
        label.append('Mean_D' + str(i))
        label.append('Std_D' + str(i))
        i += 1
    for algorithm in algorithms:
        kruskal_summary = pd.DataFrame(index=metrics, columns=label)
        for metric in metrics:
            tempArray = []
            aveList = []
            sdList = []
            for dataset_path in dataset_directory_paths:
                filename = dataset_path+'/training/results/'+name_to_abbrev[algorithm]+'_performance.csv'
                td = pd.read_csv(filename)
                tempArray.append(td[metric])
                aveList.append(td[metric].mean())
                sdList.append(td[metric].std())
            try: #Run Kruscall Wallis
                result = stats.kruskal(*tempArray)
            except:
                result = ['NA',1]
            kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
            kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
            if result[1] < sig_cutoff:
                kruskal_summary.at[metric, 'Sig(*)'] = str('*')
            else:
                kruskal_summary.at[metric, 'Sig(*)'] = str('')
            for j in range(len(aveList)):
                kruskal_summary.at[metric, 'Mean_D' + str(j+1)] = str(round(aveList[j], 6))
                kruskal_summary.at[metric, 'Std_D' + str(j+1)] = str(round(sdList[j], 6))
        #Export analysis summary to .csv file
        kruskal_summary.to_csv(experiment_path+'/DatasetComparisons/KruskalWallis_'+algorithm+'.csv')

def wilcoxonRank(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    """ For each algorithm, apply non-parametric Mann Whitney U-test (pairwise comparisons). Mann Whitney tests individual algorithm pairs of original target datasets (for each metric)
    to determine if there is a statistically significant difference in performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
    for i in range(1,3):
        label.append('Mean_Data' + str(i))
        label.append('Std_Data' + str(i))
    for algorithm in algorithms:
        master_list = []
        for metric in metrics:
            for x in range(0,len(dataset_directory_paths)-1):
                for y in range(x+1,len(dataset_directory_paths)):
                    tempList = []
                    #Grab info on first dataset
                    file1 = dataset_directory_paths[x]+'/training/results/'+name_to_abbrev[algorithm]+'_performance.csv'
                    td1 = pd.read_csv(file1)
                    set1 = td1[metric]
                    ave1 = td1[metric].mean()
                    sd1 = td1[metric].std()
                    #Grab info on second dataset
                    file2 = dataset_directory_paths[y] + '/training/results/' + name_to_abbrev[algorithm] + '_performance.csv'
                    td2 = pd.read_csv(file2)
                    set2 = td2[metric]
                    ave2 = td2[metric].mean()
                    sd2 = td2[metric].std()
                    #handle error when metric values are equal for both algorithms
                    combined = list(copy.deepcopy(set1))
                    combined.extend(list(set2))
                    if all(x == combined[0] for x in combined):  # Check if all nums are equal in sets
                        result = ['NA', 1]
                    else:
                        result = stats.wilcoxon(set1, set2)
                    #Summarize test information in list
                    tempList.append(str(metric))
                    tempList.append('D'+str(x+1))
                    tempList.append('D'+str(y+1))
                    tempList.append(str(round(result[0], 6)))
                    tempList.append(str(round(result[1], 6)))
                    if result[1] < sig_cutoff:
                        tempList.append(str('*'))
                    else:
                        tempList.append(str(''))
                    tempList.append(str(round(ave1, 6)))
                    tempList.append(str(round(sd1, 6)))
                    tempList.append(str(round(ave2, 6)))
                    tempList.append(str(round(sd2, 6)))
                    master_list.append(tempList)
        #Export test results
        df = pd.DataFrame(master_list)
        df.columns = label
        df.to_csv(experiment_path+'/DatasetComparisons/WilcoxonRank_'+algorithm+'.csv',index=False)

def mannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    """ For each algorithm, apply non-parametric Mann Whitney U-test (pairwise comparisons). Mann Whitney tests dataset pairs (for each metric)
    to determine if there is a statistically significant difference in performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
    for i in range(1,3):
        label.append('Mean_Data' + str(i))
        label.append('Std_Data' + str(i))
    for algorithm in algorithms:
        master_list = []
        for metric in metrics:
            for x in range(0,len(dataset_directory_paths)-1):
                for y in range(x+1,len(dataset_directory_paths)):
                    tempList = []
                    #Grab info on first dataset
                    file1 = dataset_directory_paths[x]+'/training/results/'+name_to_abbrev[algorithm]+'_performance.csv'
                    td1 = pd.read_csv(file1)
                    set1 = td1[metric]
                    ave1 = td1[metric].mean()
                    sd1 = td1[metric].std()
                    #Grab info on second dataset
                    file2 = dataset_directory_paths[y] + '/training/results/' + name_to_abbrev[algorithm] + '_performance.csv'
                    td2 = pd.read_csv(file2)
                    set2 = td2[metric]
                    ave2 = td2[metric].mean()
                    sd2 = td2[metric].std()
                    #handle error when metric values are equal for both algorithms
                    combined = list(copy.deepcopy(set1))
                    combined.extend(list(set2))
                    if all(x == combined[0] for x in combined):  # Check if all nums are equal in sets
                        result = ['NA', 1]
                    else:
                        result = stats.mannwhitneyu(set1, set2)
                    #Summarize test information in list
                    tempList.append(str(metric))
                    tempList.append('D'+str(x+1))
                    tempList.append('D'+str(y+1))
                    tempList.append(str(round(result[0], 6)))
                    tempList.append(str(round(result[1], 6)))
                    if result[1] < sig_cutoff:
                        tempList.append(str('*'))
                    else:
                        tempList.append(str(''))
                    tempList.append(str(round(ave1, 6)))
                    tempList.append(str(round(sd1, 6)))
                    tempList.append(str(round(ave2, 6)))
                    tempList.append(str(round(sd2, 6)))
                    master_list.append(tempList)
        #Export test results
        df = pd.DataFrame(master_list)
        df.columns = label
        df.to_csv(experiment_path+'/DatasetComparisons/MannWhitney_'+algorithm+'.csv',index=False)

def bestKruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    """ For best performing algorithm on a given metric and dataset, apply non-parametric Kruskal Wallis one-way ANOVA on ranks.
    Determines if there is a statistically significant difference in performance between original target datasets across CV runs
    on best algorithm for given metric."""
    label = ['Statistic', 'P-Value', 'Sig(*)']
    i = 1
    for dataset in datasets:
        label.append('Best_Alg_D' + str(i))
        label.append('Mean_D' + str(i))
        label.append('Std_D' + str(i))
        i += 1
    kruskal_summary = pd.DataFrame(index=metrics, columns=label)
    global_data = []
    for metric in metrics:
        best_list = []
        best_data = []
        for dataset_path in dataset_directory_paths:
            alg_ave = []
            alg_st = []
            alg_data = []
            for algorithm in algorithms:
                filename = dataset_path+'/training/results/'+name_to_abbrev[algorithm]+'_performance.csv'
                td = pd.read_csv(filename)
                alg_ave.append(td[metric].mean())
                alg_st.append(td[metric].std())
                alg_data.append(td[metric])
            # Find best algorithm for given metric based on average
            best_ave = max(alg_ave)
            best_index = alg_ave.index(best_ave)
            best_sd = alg_st[best_index]
            best_alg = algorithms[best_index]
            best_data.append(alg_data[best_index])
            best_list.append([best_alg, best_ave, best_sd])
        global_data.append([best_data, best_list])
        try:
            result = stats.kruskal(*best_data)
            kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
            kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
            if result[1] < sig_cutoff:
                kruskal_summary.at[metric, 'Sig(*)'] = str('*')
            else:
                kruskal_summary.at[metric, 'Sig(*)'] = str('')
        except ValueError:
            kruskal_summary.at[metric, 'Statistic'] = str(round('NA', 6))
            kruskal_summary.at[metric, 'P-Value'] = str(round('NA', 6))
            kruskal_summary.at[metric, 'Sig(*)'] = str('')
        for j in range(len(best_list)):
            kruskal_summary.at[metric, 'Best_Alg_D' + str(j+1)] = str(best_list[j][0])
            kruskal_summary.at[metric, 'Mean_D' + str(j+1)] = str(round(best_list[j][1], 6))
            kruskal_summary.at[metric, 'Std_D' + str(j+1)] = str(round(best_list[j][2], 6))
    #Export analysis summary to .csv file
    kruskal_summary.to_csv(experiment_path + '/DatasetComparisons/BestCompare_KruskalWallis.csv')
    return global_data

def bestMannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data):
    """ For best performing algorithm on a given metric and dataset, apply non-parametric Mann Whitney U-test (pairwise comparisons). Mann Whitney tests dataset pairs (for each metric)
    to determine if there is a statistically significant difference in performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
    for i in range(1,3):
        label.append('Best_Alg_Data' + str(i))
        label.append('Mean_Data' + str(i))
        label.append('Std_Data' + str(i))
    master_list = []
    j = 0
    for metric in metrics:
        for x in range(0, len(datasets) - 1):
            for y in range(x + 1, len(datasets)):
                tempList = []
                set1 = global_data[j][0][x]
                ave1 = global_data[j][1][x][1]
                sd1 = global_data[j][1][x][2]
                set2 = global_data[j][0][y]
                ave2 = global_data[j][1][y][1]
                sd2 = global_data[j][1][y][2]
                #handle error when metric values are equal for both algorithms
                combined = list(copy.deepcopy(set1))
                combined.extend(list(set2))
                if all(x == combined[0] for x in combined):  # Check if all nums are equal in sets
                    result = [combined[0], 1]
                else:
                    result = stats.mannwhitneyu(set1, set2)
                #Summarize test information in list
                tempList.append(str(metric))
                tempList.append('D'+str(x+1))
                tempList.append('D'+str(y+1))
                tempList.append(str(round(result[0], 6)))
                tempList.append(str(round(result[1], 6)))
                if result[1] < sig_cutoff:
                    tempList.append(str('*'))
                else:
                    tempList.append(str(''))
                tempList.append(global_data[j][1][x][0])
                tempList.append(str(round(ave1, 6)))
                tempList.append(str(round(sd1, 6)))
                tempList.append(global_data[j][1][y][0])
                tempList.append(str(round(ave2, 6)))
                tempList.append(str(round(sd2, 6)))
                master_list.append(tempList)
        j += 1
    #Export analysis summary to .csv file
    df = pd.DataFrame(master_list)
    df.columns = label
    df.to_csv(experiment_path + '/DatasetComparisons/BestCompare_MannWhitney.csv',index=False)

def bestWilcoxonRank(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data):
    """ For best performing algorithm on a given metric and dataset, apply non-parametric Mann Whitney U-test (pairwise comparisons). Mann Whitney tests dataset pairs (for each metric)
    to determine if there is a statistically significant difference in performance across CV runs. Test statistic will be zero if all scores from one set are
    larger than the other."""
    # Best Mann Whitney (Pairwise comparisons)
    label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
    for i in range(1,3):
        label.append('Best_Alg_Data' + str(i))
        label.append('Mean_Data' + str(i))
        label.append('Std_Data' + str(i))
    master_list = []
    j = 0
    for metric in metrics:
        for x in range(0, len(datasets) - 1):
            for y in range(x + 1, len(datasets)):
                tempList = []
                set1 = global_data[j][0][x]
                ave1 = global_data[j][1][x][1]
                sd1 = global_data[j][1][x][2]
                set2 = global_data[j][0][y]
                ave2 = global_data[j][1][y][1]
                sd2 = global_data[j][1][y][2]
                #handle error when metric values are equal for both algorithms
                combined = list(copy.deepcopy(set1))
                combined.extend(list(set2))
                if all(x == combined[0] for x in combined):  # Check if all nums are equal in sets
                    result = [combined[0], 1]
                else:
                    result = stats.wilcoxon(set1, set2)
                #Summarize test information in list
                tempList.append(str(metric))
                tempList.append('D'+str(x+1))
                tempList.append('D'+str(y+1))
                tempList.append(str(round(result[0], 6)))
                tempList.append(str(round(result[1], 6)))
                if result[1] < sig_cutoff:
                    tempList.append(str('*'))
                else:
                    tempList.append(str(''))
                tempList.append(global_data[j][1][x][0])
                tempList.append(str(round(ave1, 6)))
                tempList.append(str(round(sd1, 6)))
                tempList.append(global_data[j][1][y][0])
                tempList.append(str(round(ave2, 6)))
                tempList.append(str(round(sd2, 6)))
                master_list.append(tempList)
        j += 1
    #Export analysis summary to .csv file
    df = pd.DataFrame(master_list)
    df.columns = label
    df.to_csv(experiment_path + '/DatasetComparisons/BestCompare_WilcoxonRank.csv',index=False)

if __name__ == '__main__':
    job(sys.argv[1],float(sys.argv[2]))
