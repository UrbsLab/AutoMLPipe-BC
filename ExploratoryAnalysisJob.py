import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as scs
import random
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import csv
import time

'''Phase 1 of Machine Learning Analysis Pipeline:'''
#test comment

def job(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state,ignore_features_path,categorical_feature_path,sig_cutoff):
    job_start_time = time.time()
    random.seed(random_state)
    np.random.seed(random_state)
    jupyterRun = False #controls whether plots are shown or closed depending on whether jupyter notebook is used to run code or not
    topFeatures = 20 #only used with jupyter notebook reporting
    if ignore_features_path == 'None':
        ignore_features = []
    else:
        ignore_features = pd.read_csv(ignore_features_path,sep=',')
        ignore_features = list(ignore_features)

    if categorical_feature_path == 'None':
        categorical_feature_headers = []
    else:
        categorical_feature_headers = pd.read_csv(categorical_feature_path,sep=',')
        categorical_feature_headers = list(categorical_feature_headers)

    dataset_name,dataset_ext = makeFolders(dataset_path,experiment_path)

    data = loadData(dataset_path,dataset_ext)

    makeDataFolders(data,experiment_path,dataset_name)

    data = removeRowsColumns(data,class_label,ignore_features)

    categorical_variables = idFeatureTypes(data,categorical_feature_headers,instance_label,match_label,class_label,categorical_cutoff)

    countsSummary(data,class_label,experiment_path,dataset_name,instance_label,match_label,categorical_variables,jupyterRun)

    if export_exploratory_analysis == "True":
        basicExploratory(data,experiment_path,dataset_name,jupyterRun)

    if export_feature_correlations:
        featureCorrelationPlot(data,class_label,instance_label,match_label,experiment_path,dataset_name,jupyterRun)

    sorted_p_list = univariateAnalysis(data,experiment_path,dataset_name,class_label,instance_label,match_label,categorical_variables,jupyterRun,topFeatures)

    if export_univariate_plots:
        univariatePlots(data,sorted_p_list,class_label,categorical_variables,experiment_path,dataset_name,sig_cutoff)

    reportHeaders(data,experiment_path,dataset_name,class_label,instance_label,match_label,partition_method)

    #Cross Validation
    train_dfs,test_dfs = cv_partitioner(data,cv_partitions,partition_method,class_label,True,match_label,random_state)

    saveCVDatasets(experiment_path,dataset_name,train_dfs,test_dfs)

    saveRuntime(experiment_path,dataset_name,job_start_time)

    #Print completion
    print(dataset_name+" phase 1 complete")
    job_file = open(experiment_path + '/jobsCompleted/job_exploratory_'+dataset_name+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def makeFolders(dataset_path,experiment_path):
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    dataset_ext = dataset_path.split('/')[-1].split('.')[-1]
    if not os.path.exists(experiment_path + '/' + dataset_name):
        os.mkdir(experiment_path + '/' + dataset_name)
    if not os.path.exists(experiment_path + '/' + dataset_name + '/exploratory'):
        os.mkdir(experiment_path + '/' + dataset_name + '/exploratory')
    return dataset_name,dataset_ext

def loadData(dataset_path,dataset_ext):
    if dataset_ext == 'csv':
        data = pd.read_csv(dataset_path,na_values='NA',sep=',')
    else: # txt file
        data = pd.read_csv(dataset_path,na_values='NA',sep='\t')
    return data

def makeDataFolders(data,experiment_path,dataset_name):
    data.describe().to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'DescribeDataset.csv')
    data.dtypes.to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'DtypesDataset.csv',header=['DataType'],index_label='Variable')
    data.nunique().to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'NumUniqueDataset.csv',header=['Count'],index_label='Variable')

def removeRowsColumns(data,class_label,ignore_features):
    #Remove instances with missing outcome values
    data = data.dropna(axis=0,how='any',subset=[class_label])
    data = data.reset_index(drop=True)
    data[class_label] = data[class_label].astype(dtype='int64')

    #Remove columns to be ignored in analysis
    data = data.drop(ignore_features,axis=1)
    return data

def idFeatureTypes(data,categorical_feature_headers,instance_label,match_label,class_label,categorical_cutoff):
    """ Takes a dataframe (of independent variables) with column labels and returns a list of column names identified as
    being categorical based on user defined cutoff. """
    #Identify categorical variables in dataset
    if len(categorical_feature_headers) == 0:
        x_data = data.drop([class_label],axis=1)
        if not instance_label == "None":
            x_data = data.drop([instance_label], axis=1)
        if not match_label == "None":
            x_data = data.drop([match_label], axis=1)

        categorical_variables = []
        for each in x_data:
            if x_data[each].nunique() <= categorical_cutoff or not pd.api.types.is_numeric_dtype(x_data[each]):
                categorical_variables.append(each)
    else:
        categorical_variables = categorical_feature_headers
    return categorical_variables

def countsSummary(data,class_label,experiment_path,dataset_name,instance_label,match_label,categorical_variables,jupyterRun):
    #Return instance and feature counts
    fCount = data.shape[1]-1
    if not instance_label == 'None':
        fCount -= 1
    if not match_label == 'None':
        fCount -=1
    print('Data Counts: ----------------')
    print('Instance Count = '+str(data.shape[0]))
    print('Feature Count = '+str(fCount))
    print('    Categorical  = '+str(len(categorical_variables)))
    print('    Quantitative = '+str(fCount - len(categorical_variables)))

    summary = [['instances',data.shape[0]],['features',fCount],['categorical_features',len(categorical_variables)],['quantitative_features',fCount - len(categorical_variables)]]
    dfSummary = pd.DataFrame(summary, columns = ['Variable','Count'])
    dfSummary.to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'DataCounts.csv',index=None)

    #Check class counts
    class_counts = data[class_label].value_counts()
    class_counts.to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'ClassCounts.csv',header=['Count'],index_label='Class')
    print('Class Counts: ----------------')
    print(class_counts)

    #Export Class Count Bar Graph
    class_counts.plot(kind='bar')
    plt.ylabel('Count')
    plt.title('Class Counts')
    plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/'+'ClassCounts.png',bbox_inches='tight')
    if jupyterRun:
        plt.show()
    else:
        plt.close('all')

def basicExploratory(data,experiment_path,dataset_name,jupyterRun):
    #Assess Missingness in Attributes
    missing_count = data.isnull().sum()
    missing_count.to_csv(experiment_path + '/' + dataset_name + '/exploratory/'+'FeatureMissingness.csv',header=['Count'],index_label='Variable')

    #Plot a histogram of the missingness observed over all features in the dataset
    plt.hist(missing_count,bins=data.shape[0])
    plt.xlabel("Missing Value Counts")
    plt.ylabel("Frequency")
    plt.title("Histogram of Missing Value Counts In Feature Set")
    plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/'+'FeatureMissingnessHistogram.png',bbox_inches='tight')
    if jupyterRun:
        plt.show()
    else:
        plt.close('all')

def featureCorrelationPlot(data,class_label,instance_label,match_label,experiment_path,dataset_name,jupyterRun):
    data_cor = data.drop([class_label],axis=1)
    if not instance_label =='None':
        data_cor = data.drop([instance_label],axis=1)
    if not match_label == 'None':
        data_cor = data.drop([match_label],axis=1)

    corrmat = data_cor.corr(method='pearson')
    f,ax=plt.subplots(figsize=(40,20))
    sns.heatmap(corrmat,vmax=1,square=True)
    plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/'+'FeatureCorrelations.png',bbox_inches='tight')
    if jupyterRun:
        plt.show()
    else:
        plt.close('all')

def univariateAnalysis(data,experiment_path,dataset_name,class_label,instance_label,match_label,categorical_variables,jupyterRun,topFeatures):
    if not os.path.exists(experiment_path + '/' + dataset_name + '/exploratory/univariate'):
        os.mkdir(experiment_path + '/' + dataset_name + '/exploratory/univariate')
    p_value_dict = {}
    for column in data:
        if column != class_label and column != instance_label:
            p_value_dict[column] = test_selector(column,class_label,data,categorical_variables)
    sorted_p_list = sorted(p_value_dict.items(),key = lambda item:item[1])
    #Save p-values to file
    pval_df = pd.DataFrame.from_dict(p_value_dict, orient='index')
    pval_df.to_csv(experiment_path + '/' + dataset_name + '/exploratory/univariate/Significance.csv',index_label='Feature',header=['p-value'])

    if jupyterRun:
        fCount = data.shape[1]-1
        if not instance_label == 'None':
            fCount -= 1
        if not match_label == 'None':
            fCount -=1
        min_num = min(topFeatures,fCount)
        sorted_p_list_temp = sorted_p_list[: min_num]
        print('Plotting top significant '+ str(min_num) + ' features.')

        # summarize significant values of selected number of features
        print('###################################################')
        print('Significant Univariate Associations:')
        for each in sorted_p_list_temp[:min_num]:
            print(each[0]+": (p-val = "+str(each[1]) +")")

    return sorted_p_list

def univariatePlots(data,sorted_p_list,class_label,categorical_variables,experiment_path,dataset_name,sig_cutoff):
    for i in sorted_p_list:
        for j in data:
            if j == i[0] and i[1] <= sig_cutoff: #ONLY EXPORTS SIGNIFICANT FEATURES
                graph_selector(j,class_label,data,categorical_variables,experiment_path,dataset_name)

def reportHeaders(data,experiment_path,dataset_name,class_label,instance_label,match_label,partition_method):
    #Get and Export Original Headers
    headers = data.columns.values.tolist()
    headers.remove(class_label)
    if instance_label != "None":
        headers.remove(instance_label)
    if partition_method == 'M':
        headers.remove(match_label)

    with open(experiment_path + '/' + dataset_name + '/exploratory/OriginalHeaders.csv',mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
    file.close()

def saveCVDatasets(experiment_path,dataset_name,train_dfs,test_dfs):
    #Save CV'd data as .csv files
    if not os.path.exists(experiment_path + '/' + dataset_name + '/CVDatasets'):
        os.mkdir(experiment_path + '/' + dataset_name + '/CVDatasets')
    counter = 0
    for each in train_dfs:
        a = each.values
        with open(experiment_path + '/' + dataset_name + '/CVDatasets/'+dataset_name+'_CV_' + str(counter) +"_Train.csv", mode="w") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(each.columns.values.tolist())
            for row in a:
                writer.writerow(row)
        counter += 1

    counter = 0
    for each in test_dfs:
        a = each.values
        with open(experiment_path + '/' + dataset_name + '/CVDatasets/'+dataset_name+'_CV_' + str(counter) +"_Test.csv", mode="w") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(each.columns.values.tolist())
            for row in a:
                writer.writerow(row)
        file.close()
        counter += 1

def saveRuntime(experiment_path,dataset_name,job_start_time):
    #Save Runtime
    if not os.path.exists(experiment_path + '/' + dataset_name + '/runtime'):
        os.mkdir(experiment_path + '/' + dataset_name + '/runtime')
    runtime_file = open(experiment_path + '/' + dataset_name + '/runtime/runtime_exploratory.txt','w')
    runtime_file.write(str(time.time()-job_start_time))
    runtime_file.close()

########Univariate##############
def test_selector(featureName, outcomeLabel, td, categorical_variables):
    p_val = 0
    # Feature and Outcome are discrete/categorical/binary
    if featureName in categorical_variables:
        # Calculate Contingency Table - Counts
        table = pd.crosstab(td[featureName], td[outcomeLabel])

        # Univariate association test (Chi Square Test of Independence - Non-parametric)
        c, p, dof, expected = scs.chi2_contingency(table)
        p_val = p

    # Feature is continuous and Outcome is discrete/categorical/binary
    else:
        # Univariate association test (Mann-Whitney Test - Non-parametric)
        c, p = scs.mannwhitneyu(x=td[featureName].loc[td[outcomeLabel] == 0], y=td[featureName].loc[td[outcomeLabel] == 1])
        p_val = p

    return p_val

def graph_selector(featureName, outcomeLabel, td, categorical_variables,experiment_path,dataset_name):
    # Feature and Outcome are discrete/categorical/binary
    if featureName in categorical_variables:
        # Generate contingency table count bar plot. ------------------------------------------------------------------------
        # Calculate Contingency Table - Counts
        table = pd.crosstab(td[featureName], td[outcomeLabel])
        geom_bar_data = pd.DataFrame(table)
        mygraph = geom_bar_data.plot(kind='bar')
        plt.ylabel('Count')
        new_feature_name = featureName.replace(" ","")       # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = new_feature_name.replace("*","")  # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = new_feature_name.replace("/","")  # Deal with the dataset specific characters causing problems in this dataset.
        plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/univariate/'+'Barplot_'+str(new_feature_name)+".png",bbox_inches="tight", format='png')
        plt.close('all')
    # Feature is continuous and Outcome is discrete/categorical/binary
    else:
        # Generate boxplot-----------------------------------------------------------------------------------------------------
        mygraph = td.boxplot(column=featureName, by=outcomeLabel)
        plt.ylabel(featureName)
        plt.title('')
        new_feature_name = featureName.replace(" ","")       # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = new_feature_name.replace("*","")  # Deal with the dataset specific characters causing problems in this dataset.
        new_feature_name = new_feature_name.replace("/","")  # Deal with the dataset specific characters causing problems in this dataset.
        plt.savefig(experiment_path + '/' + dataset_name + '/exploratory/univariate/'+'Boxplot_'+str(new_feature_name)+".png",bbox_inches="tight", format='png')
        plt.close('all')

###################################
def cv_partitioner(td, cv_partitions, partition_method, outcomeLabel, categoricalOutcome, matchName, randomSeed):
    """ Takes data frame (td), number of cv partitions, partition method
    (R, S, or M), outcome label, Boolean indicated whether outcome is categorical
    and the column name used for matched CV. Returns list of training and testing
    dataframe partitions.
    """
    # Partitioning-----------------------------------------------------------------------------------------
    # Shuffle instances to avoid potential biases
    td = td.sample(frac=1, random_state=randomSeed).reset_index(drop=True)

    # Temporarily convert data frame to list of lists (save header for later)
    header = list(td.columns.values)
    datasetList = list(list(x) for x in zip(*(td[x].values.tolist() for x in td.columns)))

    # Handle Special Variables for Nominal Outcomes
    outcomeIndex = None
    classList = None
    if categoricalOutcome:
        outcomeIndex = td.columns.get_loc(outcomeLabel)
        classList = []
        for each in datasetList:
            if each[outcomeIndex] not in classList:
                classList.append(each[outcomeIndex])

    # Initialize partitions
    partList = []  # Will store partitions
    for x in range(cv_partitions):
        partList.append([])

    # Random Partitioning Method----------------------------
    if partition_method == 'R':
        currPart = 0
        counter = 0
        for row in datasetList:
            partList[currPart].append(row)
            counter += 1
            currPart = counter % cv_partitions

    # Stratified Partitioning Method-----------------------
    elif partition_method == 'S':
        if categoricalOutcome:  # Discrete outcome

            # Create data sublists, each having all rows with the same class
            byClassRows = [[] for i in range(len(classList))]  # create list of empty lists (one for each class)
            for row in datasetList:
                # find index in classList corresponding to the class of the current row.
                cIndex = classList.index(row[outcomeIndex])
                byClassRows[cIndex].append(row)

            for classSet in byClassRows:
                currPart = 0
                counter = 0
                for row in classSet:
                    partList[currPart].append(row)
                    counter += 1
                    currPart = counter % cv_partitions

        else:  # Do stratified partitioning for continuous endpoint data
            raise Exception("Error: Stratified partitioning only designed for discrete endpoints. ")

    elif partition_method == 'M':
        if categoricalOutcome:
            # Get match variable column index
            outcomeIndex = td.columns.get_loc(outcomeLabel)
            matchIndex = td.columns.get_loc(matchName)

            # Create data sublists, each having all rows with the same match identifier
            matchList = []
            for each in datasetList:
                if each[matchIndex] not in matchList:
                    matchList.append(each[matchIndex])

            byMatchRows = [[] for i in range(len(matchList))]  # create list of empty lists (one for each match group)
            for row in datasetList:
                # find index in matchList corresponding to the matchset of the current row.
                mIndex = matchList.index(row[matchIndex])
                row.pop(matchIndex)  # remove match column from partition output
                byMatchRows[mIndex].append(row)

            currPart = 0
            counter = 0
            for matchSet in byMatchRows:  # Go through each unique set of matched instances
                for row in matchSet:  # put all of the instances
                    partList[currPart].append(row)
                # move on to next matchset being placed in the next partition.
                counter += 1
                currPart = counter % cv_partitions

            header.pop(matchIndex)  # remove match column from partition output
        else:
            raise Exception("Error: Matched partitioning only designed for discrete endpoints. ")

    else:
        raise Exception('Error: Requested partition method not found.')

    train_dfs = []
    test_dfs = []
    for part in range(0, cv_partitions):
        testList = partList[part]  # Assign testing set as the current partition

        trainList = []
        tempList = []
        for x in range(0, cv_partitions):
            tempList.append(x)
        tempList.pop(part)

        for v in tempList:  # for each training partition
            trainList.extend(partList[v])

        train_dfs.append(pd.DataFrame(trainList, columns=header))
        test_dfs.append(pd.DataFrame(testList, columns=header))

    return train_dfs, test_dfs

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],int(sys.argv[5]),sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],sys.argv[11],int(sys.argv[12]),sys.argv[13],sys.argv[14],float(sys.argv[15]))
