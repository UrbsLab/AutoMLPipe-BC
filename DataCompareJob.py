import os
import sys
import glob
import pandas as pd
from scipy import stats
import copy

def job(experiment_path,sig_cutoff):
    # Get dataset paths
    datasets = os.listdir(experiment_path)

    datasets.remove('logs')
    datasets.remove('jobs')
    datasets.remove('jobsCompleted')
    datasets.remove('metadata.csv')
    try:
        datasets.remove('DatasetComparisons')
    except:
        pass
    datasets = sorted(datasets) #ensures consistent ordering of datasets and assignment of temporary identifier

    dataset_directory_paths = []
    for dataset in datasets:
        full_path = experiment_path + "/" + dataset
        dataset_directory_paths.append(full_path)

    # Get algorithms
    algorithms = []
    name_to_abbrev = {'logistic_regression': 'LR', 'decision_tree': 'DT', 'random_forest': 'RF', 'naive_bayes': 'NB',
                      'XGB': 'XGB', 'LGB': 'LGB', 'ANN': 'ANN', 'SVM': 'SVM', 'ExSTraCS': 'ExSTraCS', 'eLCS': 'eLCS',
                      'XCS': 'XCS','gradient_boosting':'GB','k_neighbors':'KN'}
    abbrev_to_name = dict([(value, key) for key, value in name_to_abbrev.items()])
    for filepath in glob.glob(dataset_directory_paths[0] + '/training/pickledModels/*'):
        algo_name = abbrev_to_name[filepath.split('/')[-1].split('_')[0]]
        if not algo_name in algorithms:
            algorithms.append(algo_name)

    # Get Metrics
    data = pd.read_csv(dataset_directory_paths[0] + '/training/results/Summary_performance_mean.csv', sep=',')
    metrics = data.columns.values.tolist()[1:]

    # Create new directory
    if not os.path.exists(experiment_path+'/DatasetComparisons'):
        os.mkdir(experiment_path+'/DatasetComparisons')

    kruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)

    mannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)

    global_data = bestKruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff)

    bestMannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data)

    print("Phase 7 complete")

def kruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    # Kruscall Wallis (ANOVA-like) comparison between datasets
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
            try:
                result = stats.kruskal(*tempArray)
            except:
                result = [tempArray[0][0],1]
            kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
            kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))

            if result[1] < sig_cutoff:
                kruskal_summary.at[metric, 'Sig(*)'] = str('*')
            else:
                kruskal_summary.at[metric, 'Sig(*)'] = str('')

            for j in range(len(aveList)):
                kruskal_summary.at[metric, 'Mean_D' + str(j+1)] = str(round(aveList[j], 6))
                kruskal_summary.at[metric, 'Std_D' + str(j+1)] = str(round(sdList[j], 6))

        kruskal_summary.to_csv(experiment_path+'/DatasetComparisons/'+algorithm+'_KruskalWallis.csv')

def mannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    # Mann-Whitney U test (Pairwise Comparisons)
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
                    file1 = dataset_directory_paths[x]+'/training/results/'+name_to_abbrev[algorithm]+'_performance.csv'
                    td1 = pd.read_csv(file1)
                    set1 = td1[metric]
                    ave1 = td1[metric].mean()
                    sd1 = td1[metric].std()

                    file2 = dataset_directory_paths[y] + '/training/results/' + name_to_abbrev[algorithm] + '_performance.csv'
                    td2 = pd.read_csv(file2)
                    set2 = td2[metric]
                    ave2 = td2[metric].mean()
                    sd2 = td2[metric].std()

                    combined = list(copy.deepcopy(set1))
                    combined.extend(list(set2))
                    if all(x == combined[0] for x in combined):  # Check if all nums are equal in sets
                        result = [combined[0], 1]
                    else:
                        result = stats.mannwhitneyu(set1, set2)

                    tempList.append(str(metric))
                    #tempList.append(str(datasets[x]))
                    #tempList.append(str(datasets[y]))
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
        df = pd.DataFrame(master_list)
        df.columns = label
        df.to_csv(experiment_path+'/DatasetComparisons/' +algorithm+ '_MannWhitney.csv',index=False)

def bestKruscallWallis(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff):
    #Best Kruskal Wallis results comparison
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

    kruskal_summary.to_csv(experiment_path + '/DatasetComparisons/BestCompare_KruskalWallis.csv')
    return global_data

def bestMannWhitneyU(experiment_path,datasets,algorithms,metrics,dataset_directory_paths,name_to_abbrev,sig_cutoff,global_data):
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

                combined = list(copy.deepcopy(set1))
                combined.extend(list(set2))
                if all(x == combined[0] for x in combined):  # Check if all nums are equal in sets
                    result = [combined[0], 1]
                else:
                    result = stats.mannwhitneyu(set1, set2)

                tempList.append(str(metric))
                #tempList.append(str(datasets[x]))
                #tempList.append(str(datasets[y]))
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

    df = pd.DataFrame(master_list)
    df.columns = label
    df.to_csv(experiment_path + '/DatasetComparisons/BestCompare_MannWhitney.csv',index=False)

if __name__ == '__main__':
    job(sys.argv[1],float(sys.argv[2]))
