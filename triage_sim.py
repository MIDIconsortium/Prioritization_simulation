
# code adapted from the study of Annarumma et al. "Automated Triaging of Adult Chest Radiographs with Deep Artificial Neural Networks"
# https://github.com/WMGDataScience/chest_xrays_triaging/blob/master/reporting_delays_simulation/simulate_reporting.py 

import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import pydicom as dcm
import numpy as np
import pickle
from striprtf.striprtf import rtf_to_text
import datetime as dt
from IPython.display import clear_output
import queue

def simulate_delays():

    with open('/path/to/predictions/acccession2prob.pkl','rb') as f:
        acc2prob = pickle.load(f)  # load dictionary of prediction probabities {'Accession number':'Probability_abnormal'} which will be used to assign priority
                                   # these represent the output of the CNN for each image. Note that 'Accession' is a unique identifier for each examination/image
                               

    with open('/path/to/nlp/predictions/accession2gt','rb') as f:
        acc2gt = pickle.load(f)  # load 'ground truth' labels which have been assigned on the basis of the historical radiology reports by a text classification model
                                 # {'Accession number':'NLP_probability_abnormal'}

    df = pd.read_excel('/path/to/csv/data_frame.csv') # dataframe with columns 'Accession number', 'Date Last Ver','Event Date' where 'Date Last Ver' is the date that 
    # the radiology report was finalized, and 'Event Date' is the date of imaging. The difference between the two corresponds to the historical report delay. 
    # Note: 'Event Date' and 'Date Last Ver' should strings in format ('%d/%m/%Y')

    gt = []
    preds = []
    thresh = 0.5 # Threshold for converting NLP prediction probabilities to binary label (i.e., gt = 1 if prob > threshold, else gt = 0) 
    for acc in df['Accession number']:
        preds.append(acc2prob[acc]) # CNN prediction (probability)
        if acc2prob[acc]>thresh:
            gt.append('abnormal') # 'ground truth' label for stratifying delay by category
        else:
            gt.append('normal')

    df['preds'] = preds # add CNN preds and gt to dataframe df
    df['classes'] = gt

    report_delay_real = {'normal':{},'abnormal':{}} # dict for populating historical report delay 
    report_delay_ourPrioritization = {'normal':{},'abnormal':{}} # dict for populating report delay using our prioritization 
    report_delay_randPrioritization = {'normal':{},'abnormal':{}} # dict for populating report delay using randomly assigned prioritization (for generating p-values)    

    start_date = dt.datetime.strptime('2018/01/01', "%Y/%m/%d") # date to start retrospective simulation  
    stats_start_date = start_date
    start_date = start_date - dt.timedelta(days=30) # need to initialize the queue - we do this by adding 'unreported' examinations from the previous 30 days

    end_date = dt.datetime.strptime('2019/01/01', "%Y/%m/%d") # date to stop retrospective simulation

    q_priority, q_random = queue.PriorityQueue(), queue.PriorityQueue() # initilize empty python queues 

    reports_time_bins, events_time_bins = {}, {} 
    exams_dict = {}


    for i, exam in df.iterrows():
        real_reporting_time = (pd.to_datetime(exam['Date Last Ver'],format='%d/%m/%Y') - pd.to_datetime(exam['Event Date'],format='%d/%m/%Y'))/np.timedelta64(1, 'h') # historical reporting delay 
        if real_reporting_time >= 0:
            exams_dict[exam['Accession number']] = exam[['Accession number','Event Date','Date Last Ver','preds','classes']]
            reporting_time_bin_index = int((dt.datetime.strptime(exam['Date Last Ver'], "%d/%m/%Y") - dt.datetime.min).total_seconds()//int(60*60*24)) # calculate current `report time bin' index (days)
            if not reporting_time_bin_index in reports_time_bins.keys():
                reports_time_bins[reporting_time_bin_index] = []
            reports_time_bins[reporting_time_bin_index].append(exam['Accession number']) # Add accession number to the report time bin using historical report date

            event_time_bin_index = int((dt.datetime.strptime(exam['Event Date'], "%d/%m/%Y") - dt.datetime.min).total_seconds()//int(60*60*24)) # calculate current `event time bin' index (days)
            if not event_time_bin_index in events_time_bins.keys():
                events_time_bins[event_time_bin_index] = []
            events_time_bins[event_time_bin_index].append(exam['Accession number']) # Add accession number to the event time bin using historical event date

    counter,end_loop = 0,int((end_date - start_date).total_seconds()) // int(60*60*24) 
    while start_date < end_date: # loop over each time bin (day) in retrospective simulation time interval (2018-2019)
        current_time_bin_index = int((start_date-dt.datetime.min).total_seconds()) // int(60*60*24)
        if current_time_bin_index in reports_time_bins:
            reported_count = len(reports_time_bins[current_time_bin_index]) # calculate number of exams reported historically on this day (or in this interval, if interval > or < a day)
        else:
            reported_count = 0
        for i in range(reported_count):  
            if not q_priority.empty(): # skip if no reporting performed historically in this interval (or if first loop iteration and queue hasn't been populated yet)
                class_id_tuple = q_priority.get(timeout=0) # 'pop' an exam off the top of the priority queue 
                d = exams_dict[class_id_tuple[1]]          
                NLP_priority_class = getPriorityClass([d['classes']],priority_list) # get gt for this exam                     
                if dt.datetime.strptime(d['Event Date'], "%d/%m/%Y") >= stats_start_date:
                    report_delay_ourPrioritization[NLP_priority_class][class_id_tuple[1]] = (start_date + dt.timedelta(seconds=60*60*24) - dt.datetime.strptime(d['Event Date'], "%d/%m/%Y")).total_seconds()  / 60.0 / 60.0
                    # calculate the delay for this exam using our prioritization
            else:
                break 

        for i in range(reported_count):  
            if not q_random.empty(): # skip if no reporting performed historically in this interval (or if first loop iteration and queue hasn't been populated yet)
                class_id_tuple = q_random.get(timeout=0) # 'pop' an exam off the top of the 'randomized' priority queue
                d = exams_dict[class_id_tuple[1]]          
                NLP_priority_class = getPriorityClass([d['classes']],priority_list) # get gt for this exam                        
                if dt.datetime.strptime(d['Event Date'], "%d/%m/%Y") >= stats_start_date:
                    report_delay_randPrioritization[NLP_priority_class][class_id_tuple[1]] = (start_date + dt.timedelta(seconds=60*60*24) - dt.datetime.strptime(d['Event Date'], "%d/%m/%Y")).total_seconds()  / 60.0 / 60.0
                    # calculate the delay for this exam using radnom prioritization
            else:
                break 

        # fill the queue for the next time bin
        if current_time_bin_index in events_time_bins.keys(): 
            for acc_numb in events_time_bins[current_time_bin_index]:
                d = exams_dict[acc_numb]
                NLP_priority_class = getPriorityClass([d['classes']],priority_list)
                if np.random.rand()>0.0:
                    confidence = d['preds']     
    #               rather than providing continuous probability as the priority, this can be categorized      
    #               priority_level = priority_list['normal']
    #               if confidence > 0.8:
    #                   priority_level = priority_level - 4
    #               elif confidence > 0.6:
    #                   priority_level = priority_level - 3
    #               elif confidence > 0.4:
    #                   priority_level = priority_level - 2
    #               elif confidence > 0.2:
    #                   priority_level = priority_level - 1  

                    q_priority.put((1-confidence,d['Accession number'])) # 1 - confidence since lower number represents higher priority with python queue
                    q_random.put((np.random.rand(),d['Accession number'])) # assign priority [0-1] 

                    else:
                        if dt.datetime.strptime(d['Event Date'], "%d/%m/%Y") >= stats_start_date:
                            # add priority delay for each accession number (stratified by gt)
                            report_delay_ourPrioritization[NLP_priority_class][d['Accession number']] = (dt.datetime.strptime(d['Date Last Ver'], "%d/%m/%Y") - dt.datetime.strptime(d['Event Date'], "%d/%m/%Y")).total_seconds() / 60.0 / 60.0
                            # add 'random' delay for each accession number (stratified by gt)
                            report_delay_randPrioritization[NLP_priority_class][d['Accession number']] = (dt.datetime.strptime(d['Date Last Ver'], "%d/%m/%Y") - dt.datetime.strptime(d['Event Date'], "%d/%m/%Y")).total_seconds() / 60.0 / 60.0
                        this_exam_reporting_time_bin_index = int((dt.datetime.strptime(d['Date Last Ver'], "%d/%m/%Y")-dt.datetime.min).total_seconds()) // int(60*60*24)
                        reports_time_bins[this_exam_reporting_time_bin_index].remove(reports_time_bins[this_exam_reporting_time_bin_index][0])

                    if dt.datetime.strptime(d['Event Date'], "%d/%m/%Y") >= stats_start_date:    
                        # add historical delay for each accession number (stratified by gt)
                        report_delay_real[NLP_priority_class][d['Accession number']] = (dt.datetime.strptime(d['Date Last Ver'], "%d/%m/%Y") - dt.datetime.strptime(d['Event Date'], "%d/%m/%Y")).total_seconds() / 60.0 / 60.0  

        #Move to next bin:
        start_date = start_date + dt.timedelta(seconds=60*60*24)
        counter = counter + 1   
        if counter % 100000 == 0:
            print(counter,"/",end_loop)

    return report_delay_real, report_delay_ourPrioritization, report_delay_randPrioritization


if __name__ == "__main__":
    report_delay_real, report_delay_ourPrioritization, report_delay_randPrioritization = simulate_delays()
    
    
    historical_normal_times = list(report_delay_real['normal'].values())
    historical_abnormal_times = list(report_delay_real['abnormal'].values())
    priority_normal_times = list(report_delay_ourPrioritization['normal'].values())
    priority_abnormal_times = list(report_delay_ourPrioritization['abnormal'].values())
    rand_normal_times = list(report_delay_randPrioritization['normal'].values())
    rand_abnormal_times = list(report_delay_randPrioritization['abnormal'].values())

  
    do_p_values = False
    if do_p_values:
        mean_ab_diffs = []
        mean_normal_diffs = []
        for j in range(1000):
            report_delay_real, report_delay_ourPrioritization, report_delay_randPrioritization = simulate_delays()
            rand_normal_times = list(report_delay_randPrioritization['normal'].values())
            rand_abnormal_times = list(report_delay_randPrioritization['abnormal'].values())

            ab_mean = np.mean(seq_abnormal_times).round(decimals=2)
            ab_std = np.std(seq_abnormal_times).round(decimals=2)

            mean_ab_diffs.append(ab_mean)
            mean_normal_diffs.append(mean)
            







