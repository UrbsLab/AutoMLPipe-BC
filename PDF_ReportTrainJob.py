"""
File: PDF_ReportTrainJob.py
Authors: Ryan J. Urbanowicz, Richard Zhang, Wilson Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 9 of AutoMLPipe-BC - This 'Job' script is called by PDF_ReportTrainMain.py which generates a formatted PDF summary report of key
pipeline results It is run once for the whole pipeline analysis.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import re
import sys

def job(experiment_path):

    time = str(datetime.now())
    print(time)
    #Function to Convert Dataset lists into Usable Strings to Write to the PDF

    #Find folders inside directory
    ds = os.listdir(experiment_path)
    experiment_name = experiment_path.split('/')[-1]

    nonds = ['DatasetComparisons', 'jobs', 'jobsCompleted', 'logs','KeyFileCopy','metadata.csv',experiment_name+'_ML_Pipeline_Report.pdf']
    for i in nonds:
        if i in ds:
            ds.remove(i)
    if '.idea' in ds:
        ds.remove('.idea')
    ds = sorted(ds)

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
    analy_report.cell(w=180, h=8, txt='AutoMLPipe-BC Training Summary Report: '+time, ln=2, border=1, align='L')
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

    listDatasets = ''
    i = 1
    for each in ds:
        listDatasets = listDatasets+('D'+str(i)+' = '+str(each)+'\n')
        i += 1
    analy_report.multi_cell(w = 180, h = 4, txt='Datasets: '+'\n'+listDatasets, border=1, align='L')
    #analy_report.multi_cell(w = 180, h = 6, txt='Datasets: '+'\n'+ds[0]+'\n'+ds[1], border=1, align='L')
    footer(analy_report)

    #Exploratory Univariate Analysis for each Dataset------------------------------------------------------------------
    print("Publishing Univariate Analysis")
    analy_report.add_page(orientation='P')
    analy_report.set_font('Times', 'B', 12)
    analy_report.cell(w=180, h = 8, txt='Univariate Analysis of Each Dataset (Top 10 Features)', border=1, align='L', ln=2)
    analy_report.set_font(family='times', size=9)

    for n in range(len(ds)):
        if n > 4: #more than 5 datasets
            break
        analy_report.y += 1
        sig_df = pd.read_csv(experiment_path+'/'+ds[n]+'/exploratory/univariate/Significance.csv')
        sig_ls = []
        sig_df = sig_df.nsmallest(10, ['p-value'])
        for i in range(len(sig_df)):
            sig_ls.append(sig_df.iloc[i,0]+': ')
            sig_ls.append(str(sig_df.iloc[i,1]))
            sig_ls.append('\n')
        analy_report.multi_cell(w=180, h=4, txt='Exploratory Univariate Analysis: '+'D'+str(n+1)+' = '+ds[n]+'\n'+'Feature:  P-Value'+'\n'+listToString(sig_ls), border=1, align='L')
        analy_report.y += 1
    footer(analy_report)

    if len(ds) > 5:
        analy_report.add_page(orientation='P')
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=180, h = 8, txt='Univariate Analysis of Each Dataset (Top 10 Features)', border=1, align='L', ln=2)
        analy_report.set_font(family='times', size=9)

        for n in range(5,len(ds)):
            if n > 9:
                break
            analy_report.y += 1
            sig_df = pd.read_csv(experiment_path+'/'+ds[n]+'/exploratory/univariate/Significance.csv')
            sig_ls = []
            sig_df = sig_df.nsmallest(10, ['p-value'])
            for i in range(len(sig_df)):
                sig_ls.append(sig_df.iloc[i,0]+': ')
                sig_ls.append(str(sig_df.iloc[i,1]))
                sig_ls.append('\n')
            analy_report.multi_cell(w=180, h=4, txt='Exploratory Univariate Analysis: '+'D'+str(n+1)+' = '+ds[n]+'\n'+'Feature:  P-Value'+'\n'+listToString(sig_ls), border=1, align='L')
            analy_report.y += 1
        footer(analy_report)

    if len(ds) > 10:
        analy_report.add_page(orientation='P')
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=180, h = 8, txt='Univariate Analysis of Each Dataset (Top 10 Features)', border=1, align='L', ln=2)
        analy_report.set_font(family='times', size=9)

        for n in range(10,len(ds)):
            if n > 14:
                break
            analy_report.y += 1
            sig_df = pd.read_csv(experiment_path+'/'+ds[n]+'/exploratory/univariate/Significance.csv')
            sig_ls = []
            sig_df = sig_df.nsmallest(10, ['p-value'])
            for i in range(len(sig_df)):
                sig_ls.append(sig_df.iloc[i,0]+': ')
                sig_ls.append(str(sig_df.iloc[i,1]))
                sig_ls.append('\n')
            analy_report.multi_cell(w=180, h=4, txt='Exploratory Univariate Analysis: '+'D'+str(n+1)+' = '+ds[n]+'\n'+'Feature:  P-Value'+'\n'+listToString(sig_ls), border=1, align='L')
            analy_report.y += 1
        footer(analy_report)

    if len(ds) > 15:
        analy_report.add_page(orientation='P')
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=180, h = 8, txt='Univariate Analysis of Each Dataset (Top 10 Features)', border=1, align='L', ln=2)
        analy_report.set_font(family='times', size=9)

        for n in range(15,len(ds)):
            if n > 19:
                break
            analy_report.y += 1
            sig_df = pd.read_csv(experiment_path+'/'+ds[n]+'/exploratory/univariate/Significance.csv')
            sig_ls = []
            sig_df = sig_df.nsmallest(10, ['p-value'])
            for i in range(len(sig_df)):
                sig_ls.append(sig_df.iloc[i,0]+': ')
                sig_ls.append(str(sig_df.iloc[i,1]))
                sig_ls.append('\n')
            analy_report.multi_cell(w=180, h=4, txt='Exploratory Univariate Analysis: '+'D'+str(n+1)+' = '+ds[n]+'\n'+'Feature:  P-Value'+'\n'+listToString(sig_ls), border=1, align='L')
            analy_report.y += 1
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
        analy_report.image(experiment_path+'/'+ds[n]+'/exploratory/ClassCounts.png', 5, 12, 70,48) #10, 30, 82)

        analy_report.x = 125
        analy_report.y = 55
        try:
            analy_report.image(experiment_path+'/'+ds[n]+'/exploratory/FeatureCorrelations.png', 85, 12, 115) #88, 30, 120, 60)
        except:
            analy_report.cell(40, 4, 'No Feature Correlation Plot', 1, align="L")
            pass

        data_summary = pd.read_csv(experiment_path+'/'+ds[n]+"/exploratory/DataCounts.csv")
        info_ls = []
        for i in range(len(data_summary)):
            info_ls.append(data_summary.iloc[i,0]+': ')
            info_ls.append(str(data_summary.iloc[i,1]))
            info_ls.append('\n')

        analy_report.x = 23
        analy_report.y = 62
        analy_report.multi_cell(w=40, h=4, txt='Variable:  Count'+'\n'+listToString(info_ls), border=1, align='L')

        #Report Best Algorithms by metric
        summary_performance = pd.read_csv(experiment_path+'/'+ds[n]+"/training/results/Summary_performance_mean.csv")
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
        analy_report.image(experiment_path+'/'+ds[n]+'/training/results/Summary_ROC.png', 4, 118, 120)
        analy_report.image(experiment_path+'/'+ds[n]+'/training/results/performanceBoxplots/Compare_ROC_AUC.png', 124, 118, 82,85)

        #PRC-------------------------------
        analy_report.x = 5
        analy_report.y = 200
        analy_report.cell(10, 4, 'PRC', 1, align="L")
        analy_report.image(experiment_path+'/'+ds[n]+'/training/results/Summary_PRC.png', 4, 206, 133) #wider to account for more text
        analy_report.image(experiment_path+'/'+ds[n]+'/training/results/performanceBoxplots/Compare_PRC_AUC.png', 138, 205, 68,80)

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

        stats_ds = pd.read_csv(experiment_path+'/'+str(ds[n])+'/training/results/Summary_performance_mean.csv',sep=',',index_col=0)
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

    #ML Dataset Feature Importance Summary----------------------------------------------------------------
    for k in range(len(ds)):
        analy_report.add_page()
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt="ML Feature Importance Summary:  D"+str(k+1) +' = '+ ds[k], border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=9)

        analy_report.image(experiment_path+'/'+ds[k]+'/mutualinformation/TopAverageScores.png', 5, 12, 100,135) #Images adjusted to fit a width of 100 and length of 135
        analy_report.image(experiment_path+'/'+ds[k]+'/multisurf/TopAverageScores.png', 105, 12, 100,135)

        analy_report.x = 0
        analy_report.y = 150
        analy_report.cell(0, 8, "Composite Feature Importance Plot", 1, align="L")
            #Images

        analy_report.image(experiment_path+'/'+ds[k]+'/training/results/FI/Compare_FI_Norm_Weight.png', 5, 159, 200)
        footer(analy_report)

    #Create Best Kruskall Wallis Dataset Comparison Page---------------------------------------
    print("Publishing Statistical Analysis")

    analy_report.add_page(orientation='P')
    analy_report.set_margins(left=1, top=10, right=1, )

    d = []
    for i in range(len(ds)):
        d.append('Data '+str(i+1)+'= '+ ds[i])
        d.append('\n')

    analy_report.set_font('Times', 'B', 12)
    analy_report.cell(w=0, h = 8, txt='Using Best Performing Algorithms (Kruskall Wallis Compare Datasets)', border=1, align="L", ln=2)
    analy_report.set_font(family='times', size=7)

    #Key
    listDatasets = ''
    i = 1
    for each in ds:
        listDatasets = listDatasets+('D'+str(i)+' = '+str(each)+'\n')
        i += 1

    analy_report.x = 5
    analy_report.y = 14
    analy_report.multi_cell(w = 180, h = 4, txt='Datasets: '+'\n'+listDatasets, border=1, align='L')
    analy_report.y += 5

    try:
        #Kruskall Wallis Table
        #A table can take at most 4 datasets to fit comfortably with these settings
        kw_ds = pd.read_csv(experiment_path+'/DatasetComparisons/'+'BestCompare_KruskalWallis.csv',sep=',',index_col=0)
        kw_ds = kw_ds.round(4)

        #Process
        for i in range(len(ds)):
            kw_ds = kw_ds.drop('Std_D'+str(i+1),1)
        kw_ds = kw_ds.drop('Statistic',1)
        kw_ds = kw_ds.drop('Sig(*)',1)

        #Format
        kw_ds.reset_index(inplace=True)
        kw_ds = kw_ds.columns.to_frame().T.append(kw_ds, ignore_index=True)
        kw_ds.columns = range(len(kw_ds.columns))
        epw = 208 #Amount of Space (width) Avaliable
        th = analy_report.font_size
        col_width = epw/float(10) #maximum column width

        dfLength = len(ds)
        print(dfLength)
        if len(ds) <= 4:
            kw_ds = kw_ds.to_numpy()
            for row in kw_ds:
                for datum in row:
                    analy_report.cell(col_width, th, str(datum), border=1)
                analy_report.ln(th) #critical
        else:
            #Print next 3 datasets
            table1 = kw_ds.iloc[: , :10]
            table1 = table1.to_numpy()
            for row in table1:
                for datum in row:
                    analy_report.cell(col_width, th, str(datum), border=1)
                analy_report.ln(th) #critical
            analy_report.y += 5

            table1 = kw_ds.iloc[: , 10:18]
            met = kw_ds.iloc[:,0]
            met2 = kw_ds.iloc[:,1]
            table1 = pd.concat([met,met2, table1], axis=1)
            table1 = table1.to_numpy()
            for row in table1:
                for datum in row:
                    analy_report.cell(col_width, th, str(datum), border=1)
                analy_report.ln(th) #critical
            analy_report.y += 5

            if len(ds) > 8:
                table1 = kw_ds.iloc[: , 18:26]
                met = kw_ds.iloc[:,0]
                met2 = kw_ds.iloc[:,1]
                table1 = pd.concat([met,met2,table1], axis=1)
                table1 = table1.to_numpy()
                for row in table1:
                    for datum in row:
                        analy_report.cell(col_width, th, str(datum), border=1)
                    analy_report.ln(th) #critical
                analy_report.y += 5

            if len(ds) > 12:
                table1 = kw_ds.iloc[: , 26:34]
                met = kw_ds.iloc[:,0]
                met2 = kw_ds.iloc[:,1]
                table1 = pd.concat([met,met2,table1], axis=1)
                table1 = table1.to_numpy()
                for row in table1:
                    for datum in row:
                        analy_report.cell(col_width, th, str(datum), border=1)
                    analy_report.ln(th) #critical
                analy_report.y += 5

            if len(ds) > 16:
                analy_report.x = 0
                analy_report.y = 260
                analy_report.cell(0, 4, 'Warning: Additional dataset results could not be displayed', 1, align="C")
    except:
        pass

    footer(analy_report)

    #Create Runtime Summary Page---------------------------------------
    print("Publishing Runtime Summary")

    analy_report.add_page(orientation='P')
    analy_report.set_margins(left=1, top=10, right=1, )
    analy_report.set_font('Times', 'B', 12)
    analy_report.cell(w=0, h = 8, txt='Pipeline Runtime Summary', border=1, align="L", ln=2)
    analy_report.set_font(family='times', size=7)
    analy_report.y += 4
    col_width = 50 #maximum column width

    for n in range(len(ds)):
        if n > 3:
            break
        analy_report.cell(100, 4, str(ds[n]), 1, align="L")
        analy_report.y += 4
        analy_report.x = 1
        time_df = pd.read_csv(experiment_path+'/'+ds[n]+'/runtimes.csv')
        time_df.iloc[:, 1] = time_df.iloc[:, 1].round(2)
        time_df = time_df.columns.to_frame().T.append(time_df, ignore_index=True)
        time_df = time_df.to_numpy()
        for row in time_df:
            for datum in row:
                analy_report.cell(col_width, th, str(datum), border=1)
            analy_report.ln(th) #critical
        analy_report.y += 4
    footer(analy_report)

    if len(ds) > 4:
        analy_report.add_page(orientation='P')
        analy_report.set_margins(left=1, top=10, right=1, )
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt='Pipeline Runtime Summary', border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=7)
        analy_report.y += 4
        col_width = 50 #maximum column width
        for n in range(4,len(ds)):
            if n > 7:
                break
            analy_report.cell(100, 4, str(ds[n]), 1, align="L")
            analy_report.y += 4
            analy_report.x = 1
            time_df = pd.read_csv(experiment_path+'/'+ds[n]+'/runtimes.csv')
            time_df.iloc[:, 1] = time_df.iloc[:, 1].round(2)
            time_df = time_df.columns.to_frame().T.append(time_df, ignore_index=True)
            time_df = time_df.to_numpy()
            for row in time_df:
                for datum in row:
                    analy_report.cell(col_width, th, str(datum), border=1)
                analy_report.ln(th) #critical
            analy_report.y += 4
        footer(analy_report)

    if len(ds) > 8:
        analy_report.add_page(orientation='P')
        analy_report.set_margins(left=1, top=10, right=1, )
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt='Pipeline Runtime Summary', border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=7)
        analy_report.y += 4
        col_width = 50 #maximum column width
        for n in range(8,len(ds)):
            if n > 11:
                break
            analy_report.cell(100, 4, str(ds[n]), 1, align="L")
            analy_report.y += 4
            analy_report.x = 1
            time_df = pd.read_csv(experiment_path+'/'+ds[n]+'/runtimes.csv')
            time_df.iloc[:, 1] = time_df.iloc[:, 1].round(2)
            time_df = time_df.columns.to_frame().T.append(time_df, ignore_index=True)
            time_df = time_df.to_numpy()
            for row in time_df:
                for datum in row:
                    analy_report.cell(col_width, th, str(datum), border=1)
                analy_report.ln(th) #critical
            analy_report.y += 4
        footer(analy_report)

    if len(ds) > 12:
        analy_report.add_page(orientation='P')
        analy_report.set_margins(left=1, top=10, right=1, )
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt='Pipeline Runtime Summary', border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=7)
        analy_report.y += 4
        col_width = 50 #maximum column width
        for n in range(12,len(ds)):
            if n > 15:
                break
            analy_report.cell(100, 4, str(ds[n]), 1, align="L")
            analy_report.y += 4
            analy_report.x = 1
            time_df = pd.read_csv(experiment_path+'/'+ds[n]+'/runtimes.csv')
            time_df.iloc[:, 1] = time_df.iloc[:, 1].round(2)
            time_df = time_df.columns.to_frame().T.append(time_df, ignore_index=True)
            time_df = time_df.to_numpy()
            for row in time_df:
                for datum in row:
                    analy_report.cell(col_width, th, str(datum), border=1)
                analy_report.ln(th) #critical
            analy_report.y += 4
        footer(analy_report)

    if len(ds) > 16:
        analy_report.add_page(orientation='P')
        analy_report.set_margins(left=1, top=10, right=1, )
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt='Pipeline Runtime Summary', border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=7)
        analy_report.y += 4
        col_width = 50 #maximum column width
        for n in range(16,len(ds)):
            if n > 19:
                break
            analy_report.cell(100, 4, str(ds[n]), 1, align="L")
            analy_report.y += 4
            analy_report.x = 1
            time_df = pd.read_csv(experiment_path+'/'+ds[n]+'/runtimes.csv')
            time_df.iloc[:, 1] = time_df.iloc[:, 1].round(2)
            time_df = time_df.columns.to_frame().T.append(time_df, ignore_index=True)
            time_df = time_df.to_numpy()
            for row in time_df:
                for datum in row:
                    analy_report.cell(col_width, th, str(datum), border=1)
                analy_report.ln(th) #critical
            analy_report.y += 4
        footer(analy_report)

    #Output The PDF Object
    try:
        fileName = str(experiment_name)+'_ML_Pipeline_Report.pdf'
        analy_report.output(experiment_path+'/'+fileName)
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
    job(sys.argv[1])
