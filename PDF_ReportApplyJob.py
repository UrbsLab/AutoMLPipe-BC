"""
File: PDF_ReportApplyJob.py
Authors: Ryan J. Urbanowicz, Richard Zhang, Wilson Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 11 of AutoMLPipe-BC - This 'Job' script is called by PDF_ReportApplyMain.py which generates a formatted PDF summary report of key
pipeline results (applying trained models to hold out replication data). It is run once.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import glob
import os
import re
import sys

def job(experiment_path,rep_data_path,data_path):

    time = str(datetime.now())
    train_name = data_path.split('/')[-1].split('.')[0]
    experiment_name = experiment_path.split('/')[-1]
    #Function to Convert Dataset lists into Usable Strings to Write to the PDF
    #Find folders inside directory
    ds = []
    for datasetFilename in glob.glob(rep_data_path+'/*'):
        datasetFilename = str(datasetFilename).replace('\\','/')
        apply_name = datasetFilename.split('/')[-1].split('.')[0] #Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
        ds.append(apply_name)
    ds = sorted(ds)
    print(ds)

    ars_df = pd.read_csv(experiment_path+ '/'+'metadata.csv')
    ars_dic = []
    for i in range(len(ars_df)):
       if i >= 0:
          ars_dic.append(ars_df.iloc[i, 0]+': ')
          ars_dic.append(ars_df.iloc[i, 1])
          ars_dic.append('\n')
       else:
          pass

    #Analysis Settings, Global Analysis Settings, ML Modeling Algorithms
    analy_report = FPDF('P', 'mm', 'A4')
    analy_report.set_margins(left=10, top=5, right=10, )
    analy_report.add_page(orientation='P')
    top = analy_report.y

    #ML Pipeline Analysis Report-------------------------------------------------------------------------------------------------------
    print("Starting Report")
    ls1 = ars_dic[0:55]
    ls2 = ars_dic[56:95]  #ML modeling algorithms
    ls3 = ars_dic[94:111]
    ls4 = ars_dic[110:125]  #LCS parameters
    analy_report.set_font('Times', 'B', 12)
    analy_report.cell(w=180, h=8, txt='AutoMLPipe-BC Apply Summary Report: '+time, ln=2, border=1, align='L')
    analy_report.y += 3
    analy_report.set_font(family='times', size=9)
    analy_report.multi_cell(w = 90,h = 4,txt='Pipeline Settings:'+'\n'+'\n'+listToString(ls1)+' '+listToString(ls3), border=1, align='L')
    analy_report.x += 90
    analy_report.y = analy_report.y - 104 #96
    analy_report.multi_cell(w = 90,h = 4,txt='ML Modeling Algorithms:'+'\n'+'\n'+listToString(ls2), border=1, align='L')
    analy_report.x += 90
    analy_report.y += 4
    analy_report.multi_cell(w = 90,h = 4,txt='LCS Settings (ExSTraCS,eLCS,XCS):'+'\n'+listToString(ls4), border=1, align='L')
    analy_report.y +=10

    analy_report.cell(w = 180, h = 4, txt='Target Training Dataset: '+train_name, border=1, align='L')
    analy_report.y +=8
    analy_report.x = 10

    listDatasets = ''
    i = 1
    for each in ds:
        listDatasets = listDatasets+('D'+str(i)+' = '+str(each)+'\n')
        i += 1
    analy_report.multi_cell(w = 180, h = 4, txt='Applied Datasets: '+'\n'+listDatasets, border=1, align='L')
    footer(analy_report)

    #Data and Model Prediction Summary--------------------------------------------------------------------------------------
    print("Publishing Model Prediction Summary")
    for n in range(len(ds)):
        #Create PDF and Set Options
        analy_report.set_margins(left=1, top=1, right=1, )
        analy_report.add_page()
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt="Dataset and Model Prediction Summary:  D"+str(n+1)+" = "+ds[n], border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=9)

        #Exploratory Analysis ----------------------------
        analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/exploratory/ClassCounts.png', 5, 12, 70,48) #10, 30, 82)

        analy_report.x = 125
        analy_report.y = 55
        try:
            analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/exploratory/FeatureCorrelations.png', 85, 12, 115) #88, 30, 120, 60)
        except:
            analy_report.cell(40, 4, 'No Feature Correlation Plot', 1, align="L")
            pass

        data_summary = pd.read_csv(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+"/exploratory/DataCounts.csv")
        info_ls = []
        for i in range(len(data_summary)):
            info_ls.append(data_summary.iloc[i,0]+': ')
            info_ls.append(str(data_summary.iloc[i,1]))
            info_ls.append('\n')

        analy_report.x = 23
        analy_report.y = 62
        analy_report.multi_cell(w=40, h=4, txt='Variable:  Count'+'\n'+listToString(info_ls), border=1, align='L')

        #Report Best Algorithms by metric
        summary_performance = pd.read_csv(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+"/training/results/Summary_performance_mean.csv")
        summary_performance['ROC_AUC'] = summary_performance['ROC_AUC'].astype(float)
        highest_ROC = summary_performance['ROC_AUC'].max()
        algorithm = summary_performance[summary_performance['ROC_AUC'] == highest_ROC].index.values
        best_alg_ROC =  summary_performance.iloc[algorithm, 0]

        summary_performance['Balanced Accuracy'] = summary_performance['Balanced Accuracy'].astype(float)
        highest_BA = summary_performance['Balanced Accuracy'].max()
        algorithm = summary_performance[summary_performance['Balanced Accuracy'] == highest_BA].index.values
        best_alg_BA =  summary_performance.iloc[algorithm, 0]

        summary_performance['F1_Score'] = summary_performance['F1_Score'].astype(float)
        highest_F1 = summary_performance['F1_Score'].max()
        algorithm = summary_performance[summary_performance['F1_Score'] == highest_F1].index.values
        best_alg_F1 =  summary_performance.iloc[algorithm, 0]

        summary_performance['PRC_AUC'] = summary_performance['PRC_AUC'].astype(float)
        highest_PRC = summary_performance['PRC_AUC'].max()
        algorithm = summary_performance[summary_performance['PRC_AUC'] == highest_PRC].index.values
        best_alg_PRC = summary_performance.iloc[algorithm, 0]

        summary_performance['PRC_APS'] = summary_performance['PRC_APS'].astype(float)
        highest_APS = summary_performance['PRC_APS'].max()
        algorithm = summary_performance[summary_performance['PRC_APS'] == highest_APS].index.values
        best_alg_APS = summary_performance.iloc[algorithm, 0]

        analy_report.x = 5
        analy_report.y = 87
        analy_report.multi_cell(w=70, h=4, txt="Best (ROC_AUC): "+ str(best_alg_ROC.values)+' = '+ str("{:.3f}".format(highest_ROC))+
                    '\n'+"Best (Balanced Acc.): "+ str(best_alg_BA.values)+' = '+ str("{:.3f}".format(highest_BA))+
                    '\n'+"Best (F1 Score): "+ str(best_alg_F1.values)+' = '+ str("{:.3f}".format(highest_F1))+
                    '\n'+"Best (PRC_AUC): "+ str(best_alg_PRC.values)+' = '+ str("{:.3f}".format(highest_PRC))+
                    '\n'+"Best (PRC_APS): "+ str(best_alg_APS.values)+' = '+ str("{:.3f}".format(highest_APS)), border=1, align='L')

        #ROC-------------------------------
        analy_report.x = 5
        analy_report.y = 112
        analy_report.cell(10, 4, 'ROC', 1, align="L")
        analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/training/results/Summary_ROC.png', 4, 118, 120)
        analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/training/results/performanceBoxplots/Compare_ROC_AUC.png', 124, 118, 82,85)

        #PRC-------------------------------
        analy_report.x = 5
        analy_report.y = 200
        analy_report.cell(10, 4, 'PRC', 1, align="L")
        analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/training/results/Summary_PRC.png', 4, 206, 133) #wider to account for more text
        analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/training/results/performanceBoxplots/Compare_PRC_AUC.png', 138, 205, 68,80)

        footer(analy_report)

    #Average Model Prediction Statistics--------------------------------------------------------------------------------------
    print("Publishing Average Model Prediction Statistics")
    for n in range(len(ds)):
        #Create PDF and Set Options
        analy_report.set_margins(left=1, top=1, right=1, )
        analy_report.add_page()
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt="Average Model Prediction Statistics:  D"+str(n+1)+" = "+ds[n], border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=7)

        stats_ds = pd.read_csv(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/training/results/Summary_performance_mean.csv',sep=',',index_col=0)
        stats_ds = stats_ds.round(4)

        #Format
        stats_ds.reset_index(inplace=True)
        stats_ds = stats_ds.columns.to_frame().T.append(stats_ds, ignore_index=True)
        stats_ds.columns = range(len(stats_ds.columns))
        epw = 208 #Amount of Space (width) Avaliable
        th = analy_report.font_size
        col_width = epw/float(10) #maximum column width

        #Print next 3 datasets
        table1 = stats_ds.iloc[: , :10]
        table1 = table1.to_numpy()
        for row in table1:
            for datum in row:
                analy_report.cell(col_width, th, str(datum), border=1)
            analy_report.ln(th) #critical
        analy_report.y += 5

        table1 = stats_ds.iloc[: , 10:18]
        met = stats_ds.iloc[:,0]
        table1 = pd.concat([met, table1], axis=1)
        table1 = table1.to_numpy()
        for row in table1:
            for datum in row:
                analy_report.cell(col_width, th, str(datum), border=1)
            analy_report.ln(th) #critical
        analy_report.y += 5

        footer(analy_report)

    #Output The PDF Object
    try:
        fileName = str(experiment_name)+'_ML_Pipeline_Apply_Report.pdf'
        analy_report.output(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/'+fileName)
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
    self.cell(0, 7,'Generated with the URBS-Lab AutoMLPipe-BC: (https://github.com/UrbsLab/AutoMLPipe-BC)', 0, 0, 'C')
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
    job(sys.argv[1],sys.argv[2],sys.argv[3])
