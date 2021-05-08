#######################################
## Richard Zhang And Wilson Zhang    ##
## March 30, 2021                    ##
## ML Pipeline Report Generator V. 1 ##
## Requirements: pip install fpdf
#######################################

import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import re
import sys
import argparse


def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--experiment-path',dest='experiment_path',type=str,help='path to directory containing ML experiment results')

    options = parser.parse_args(argv[1:])
    experiment_path = options.experiment_path


    time = str(datetime.now())
    #Function to Convert Dataset lists into Usable Strings to Write to the PDF

    #Analysis Settings, Global Analysis Settings, ML Modeling Algorithms
    ars_df = pd.read_csv(experiment_path+ '/'+'metadata.csv')
    analy_report = FPDF('P', 'mm', 'A4')
    analy_report.set_font(family='times', size=9)
    analy_report.set_margins(left=15, top=10, right=15, )
    analy_report.add_page(orientation='P')


    top = analy_report.y

    #Find folders inside directory
    #ds = os. listdir(os.getcwd())
    ds = os.listdir(experiment_path)
    #print(ds)
    #ds = [item for item in ds if os.path.isdir(item)]
    #print(ds)
    nonds = ['DatasetComparisons', 'jobs', 'jobsCompleted', 'logs','metadata.csv']
    for i in nonds:
        if i in ds:
            ds.remove(i)
    if '.idea' in ds:
        ds.remove('.idea')
    #print(ds)

    ars_dic = []
    for i in range(len(ars_df)):
       if i >= 0:
          ars_dic.append(ars_df.iloc[i, 0]+': ')
          ars_dic.append(ars_df.iloc[i, 1])
          ars_dic.append('\n')
       else:
          pass

    #ML Pipeline Analysis Report-------------------------------------------------------------------------------------------------------
    print("Starting Report")
    ls1 = ars_dic[0:5]
    ls2 = ars_dic[6:32]
    ls3 = ars_dic[33:72]
    analy_report.cell(w=0, h=6, txt='ML Pipeline Analysis Report: '+time, ln=2, border=1, align='C')
    analy_report.y += 3
    analy_report.multi_cell(w = 0,h = 4,txt='Analysis Settings Summary:'+'\n'+'\n'+listToString(ls1), border=1, align='C')
    analy_report.y += 3
    analy_report.multi_cell(w = 90,h = 6,txt='Global Analysis Settings:'+'\n'+'\n'+listToString(ls2), border=1, align='C')
    analy_report.x += 90
    analy_report.y = analy_report.y - 66
    analy_report.multi_cell(w = 90,h = 4.3959,txt='ML Modeling Algorithms:'+'\n'+'\n'+listToString(ls3), border=1, align='C')
    analy_report.y +=3
    analy_report.multi_cell(w = 180, h = 6, txt='Datasets: '+'\n'+ds[0]+'\n'+ds[1], border=1, align='C')
    footer(analy_report)

    #Exploratory Univariate Analysis for each Dataset
    print("Publishing Univariate Analysis")
    analy_report.add_page(orientation='P')
    analy_report.cell(w=180, h = 6, txt='Univariate Analysis of Each Dataset (Top 10 Features)', border=1, align='C', ln=2)
    for n in range(len(ds)):
        analy_report.y += 3
        #os.chdir(ds[n] + '/exploratory/univariate')
        sig_df = pd.read_csv(experiment_path+'/'+ds[n]+'/exploratory/univariate/'+'Significance.csv')
        #os.chdir('../../..')
        sig_ls = []
        sig_df = sig_df.nsmallest(10, ['0'])
        for i in range(len(sig_df)):
            sig_ls.append(sig_df.iloc[i,0]+': ')
            sig_ls.append(str(sig_df.iloc[i,1]))
            sig_ls.append('\n')
        analy_report.multi_cell(w=180, h=7, txt='Exploratory Univariate Analysis of '+ds[n]+'\n'+'Feature            P-Value'+'\n'+listToString(sig_ls), border=1, align='L')
        analy_report.y += 3
    footer(analy_report)

    #ML Dataset Prediction Summary
    print("Publishing Model Prediction Summary")
    for n in range(len(ds)):
        #Create PDF and Set Options
        analy_report.set_font(family='times', size=13)

        analy_report.set_margins(left=1, top=1, right=1, )

        analy_report.add_page()
        analy_report.cell(0, 10, "ML Dataset Prediction Summary:  "+ds[n], 1, align="C")
        #Exploratory Analysis
            #Determining Best AUC & APS for ROC and PRC
                #Best ROC_AUC
        summary_performance = pd.read_csv(experiment_path+'/'+ds[n]+"/training/results/Summary_performance_mean.csv")

        #for i in range(0,13):
        #    x = summary_performance.iloc[i, 11]
        #    summary_performance.iloc[i, 11] = (re.sub("[\(\[].*?[\)\]]", "", x))

        summary_performance['ROC_AUC'] = summary_performance['ROC_AUC'].astype(float)
        highest_result = summary_performance['ROC_AUC'].max()

        algorithm = summary_performance[summary_performance['ROC_AUC'] == highest_result].index.values
        algorithm2 = summary_performance[summary_performance['ROC_AUC'] == summary_performance['ROC_AUC'].max()].index[0]
        str(algorithm2)
        highest_result_algorithm = summary_performance.iloc[algorithm, 0]
        best_alg = highest_result_algorithm
                #Best PRC_AUC
        #try:
        #    summary_performance = pd.read_csv(ds[n]+'/training/results/Summary_performance.csv')
        #except:
        #    summary_performance = pd.read_csv("Summary_performance.csv")

        #for i in range(0,13):
        #    y = summary_performance.iloc[i, 12]
        #    summary_performance.iloc[i, 12] = (re.sub("[\(\[].*?[\)\]]", "", y))

        summary_performance['PRC_AUC'] = summary_performance['PRC_AUC'].astype(float)
        highest_result2 = summary_performance['PRC_AUC'].max()

        algorithm = summary_performance[summary_performance['PRC_AUC'] == highest_result2].index.values
        highest_result_algorithm2 = summary_performance.iloc[algorithm, 0]
        best_alg2 = highest_result_algorithm2

            #Best PRC_APS
        #try:
        #    summary_performance = pd.read_csv(ds[n] + '/training/results/Summary_performance.csv')
        #except:
        #    summary_performance = pd.read_csv("Summary_performance.csv")

        #for i in range(0,13):
        #    z = summary_performance.iloc[i, 13]
        #    summary_performance.iloc[i, 13] = (re.sub("[\(\[].*?[\)\]]", "", z))

        summary_performance['PRC_APS'] = summary_performance['PRC_APS'].astype(float)
        highest_result3 = summary_performance['PRC_APS'].max()

        algorithm = summary_performance[summary_performance['PRC_APS'] == highest_result3].index.values
        highest_result_algorithm3 = summary_performance.iloc[algorithm, 0]
        best_alg3 = highest_result_algorithm3
            #Images
        analy_report.image(experiment_path+'/'+ds[n]+'/exploratory/ClassCounts.png', 10, 30, 82)
        analy_report.image(experiment_path+'/'+ds[n]+'/exploratory/FeatureCorrelations.png', 88, 30, 120, 60)
        analy_report.image(experiment_path+'/'+ds[n]+'/training/results/Summary_ROC.png', 15, 120, 70)
        analy_report.image(experiment_path+'/'+ds[n]+'/training/results/performanceBoxplots/Compare_ROC_AUC.png', 120, 123, 70)
        analy_report.image(experiment_path+'/'+ds[n]+'/training/results/Summary_PRC.png', 15, 210, 70)
        analy_report.image(experiment_path+'/'+ds[n]+'/training/results/performanceBoxplots/Compare_PRC_AUC.png', 120, 210, 70)
            #Text
        analy_report.x = 85
        analy_report.y = 15
        analy_report.cell(50, 8, "Exploratory Analysis:", 1, align="C")

        analy_report.x = analy_report.x - 100
        analy_report.y += 85
        analy_report.cell(30, 8, "Model ROC:", 1, align="C")

        analy_report.x += 35
        analy_report.cell(90, 8, "Best (AUC):" + str(best_alg.values) + ", " + str(highest_result), 1, align="C")
        analy_report.x = analy_report.x - 90
        analy_report.y += 90
        analy_report.cell(100, 8, "Best (AUC):" + str(best_alg2.values) + ", " + str(highest_result2), 1, align="C")
        analy_report.x = analy_report.x - 100
        analy_report.y += 10
        analy_report.cell(100, 8, "Best (APS):" + str(best_alg3.values) + ", " + str(highest_result3), 1, align="C")
        analy_report.x = analy_report.x - 160
        analy_report.y = analy_report.y - 2
        analy_report.cell(25, 8, "Model PRC:", 1, align="C")
        footer(analy_report)

    for k in range(len(ds)):
        #ML Dataset Feature Importance Summary
        analy_report.add_page()
        analy_report.cell(0, 10, "ML Dataset Feature Importance Summary:  " + ds[k] , 1, align="C")
        analy_report.x = analy_report.x - 208
        analy_report.y += 158
        analy_report.cell(0, 8, "Compound Feature Importance Plot", 1, align="C")
            #Images
        analy_report.image(experiment_path+'/'+ds[k]+'/mutualinformation/TopAverageScores.png', 2, 15, 0, 140)
        analy_report.image(experiment_path+'/'+ds[k]+'/multisurf/TopAverageScores.png', 113, 15, 0, 140)
        analy_report.image(experiment_path+'/'+ds[k]+'/training/results/FI/Compare_FI_Norm_Frac_Weight.png', 5, 178, 200, 119)
        footer(analy_report)

    #Create Kruskall Wallis Dataset Comparison Page
    print("Publishing Statistical Analysis")

    analy_report.add_page(orientation='P')
    analy_report.set_font(family='times', size=9)
    analy_report.set_margins(left=5, top=10, right=5, )
    d = []
    for i in range(len(ds)):
        d.append('Data '+str(i+1)+'= '+ ds[i])
        d.append('\n')
    analy_report.y += 3
    analy_report.multi_cell(w = 180, h = 6, txt='Datasets:  '+'\n'+listToString(d), border=1, align='L')
    analy_report.y += 3

    #os.chdir('DatasetComparisons')
    kw_ds = pd.read_csv(experiment_path+'/DatasetComparisons/'+'BestCompare_KruskalWallis.csv', header=None)
    for i in range((2*len(ds))):
        for k in range(len(ds)):
            if kw_ds.iloc[0, i+4] == 'mean_'+ds[k] or kw_ds.iloc[0, i+4] == 'std_'+ds[k]:
                kw_ds.iloc[0, i+4] = kw_ds.iloc[0, i+4].replace(ds[k], 'data'+str(k+1))
            elif kw_ds.iloc[0, i+4] == 'mean_'+ds[k] or kw_ds.iloc[0, i+4] == 'std_'+ds[k]:
                kw_ds.iloc[0, i+4] = kw_ds.iloc[0, i+4].replace(ds[k], 'data'+str(k+1))
    kw_ds = kw_ds.to_numpy()
    epw = analy_report.w - 2*analy_report.l_margin #Amount of Space (length) Avaliable
    th = analy_report.font_size
    col_width = epw/(4+2*len(ds))
    for row in kw_ds:
        for datum in row:
            analy_report.cell(col_width, 2 * th, str(datum), border=1)
        analy_report.ln(2 * th)
    footer(analy_report)
    #Output The PDF Object
    try:
        os.chdir(experiment_path)
        experiment_name = experiment_path.split('/')[-1].split('.')[-1]
        analy_report.output(name=experiment_name+'_ML_Pipeline_Report.pdf')
        print('PDF Generation Complete')
    except:
        print('Pdf Output Failed')

def listToString(s):
    str1 = " "
    return (str1.join(s))


#Create Footer
def footer(self):
    self.set_auto_page_break(auto=False, margin=3)
    self.set_y(285)
    self.set_font('Times', 'I', 7)
    self.cell(0, 7,'Generated with the URBS-Lab Automated ML Comparison Pipeline:    https://github.com/UrbsLab/scikit_ML_Pipeline_Binary_Parallel', 0, 0, 'C')
    self.set_font(family='times', size=9)

#Find N greatest ingegers within a list
def ngi(list1, N):
    final_list = []
    for i in range(0, N):
        max1 = 0
        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];
        list1.remove(max1);
        final_list.append(max1)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
