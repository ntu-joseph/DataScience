import sys  # handle command argument
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing   #for normalization
from sklearn.model_selection import cross_val_score  # for cross validation
from sklearn import linear_model
from sklearn import  metrics # confusion matrix
from sklearn.model_selection import train_test_split
import itertools #  for  plot_confusion_matrix
import pandas as pd
import os, sys,glob
from pathlib import Path
from datetime import datetime,date


#TRAIN



print("Import End")

printtime=0


def process_time(stime):
    #'2017-11-24 7:59:25.334'
    time_slot=-1
    date1,time1= stime.split(' ')
    date1=[int(i) for i in date1.split('-')]   #[2017, 6, 8] Y/M/D
    time1=[float(i) for i in time1.split(':')]  #[15.0, 18.0, 25.334] H/M/S
    t=datetime(date1[0], date1[1],date1[2],int(time1[0]),int(time1[1])).time()
    
    t0=t.replace(hour=1,minute=0)
    t1=t.replace(hour=9,minute=0)
    t2=t.replace(hour=17,minute=0)
    t3=t.replace(hour=21,minute=0)
    global printtime
    printtime+=1
    if printtime<15:
        print("t = ",t)
        print("t0 = ",t0)  #01:00:00
        print("date1=",date1)
        print("Is this ok:",date1[0],date1[1])



    if (t>=t0 and t<t1): #AM1:00~AM9:00
        time_slot=0
    elif (t>=t1 and t<t2):  #AM9:00~PM17:00
        time_slot=1
    elif (t>=t2 and t<t3): #PM17:00~PM9:00 
        time_slot=2
    else:  # >PM 9:00
        time_slot=3
    # weekday
    weekday=date(date1[0], date1[1],date1[2]).weekday()   #Modnday:0, Tuesday:1 ,...Sunday:6
    weightNum=(date1[1]-1)*4+(date1[2]/7)      #which slot in total time
    #weightNum=1 (1/1) to 30.~ (8/21)
    if printtime<15:
        print("WN=",weightNum)
    return weekday,time_slot,weightNum




# load data

def read_csv(files):
    #column filter
    
    '''
    all column list:
    user_id,device_id,session_id,title_id,event_time,played_duration,
    action_trigger,platform,episode_number,series_total_episodes_count,internet_connection_type,is_trailer
    '''
    #olny use partial column
    cols=['user_id','event_time','played_duration']
    df=pd.read_csv(files,usecols=cols)
    return df




def preprocess_file(file,start_id):    
   
    df=read_csv(file)

    max_user_id=df.iloc[-1:, 0] # last row of column 0
    print("in pre pro , max id is:",max_user_id)
    max_id=np.array(max_user_id, dtype=pd.Series)[0] # series to ndarray
    print(max_id)
    print(df.shape)
    print("PREPRO")
    

    user_timeslot_list=[]
    #output file 'data.csv'
    #should i should previous ...
    
    for i in range(start_id,max_id+1):
        user_df=df[df['user_id']==i]
 #       print(user_df.shape)
#        print("PREPRO IN RANGE",start_id,max_id," i ",i)

        n_events=user_df.shape[0] #number of row
        total_duration=0
        total_weight_duration=0
        weight_week=0
        global printtime
        time_slot=[0]*28
        for row in user_df.itertuples(index=True):
            weekday,slot,weight_week=process_time(getattr(row, "event_time"))
            #time_slot[weekday*4+slot]+=1 #event count in each timeslot
            t=getattr(row, "played_duration")
            time_slot[weekday*4+slot]+=t*(weight_week/15) #event count in each timeslot
            total_duration+=t
            total_weight_duration+=t*(weight_week/15)
            if printtime<15:
                print(" weekday,slot,weight_week,t:",weekday,slot,weight_week,t)

        #print(i,n_events,total_duration,time_slot)
        #write file for each row
        if (i==0):
            #write header
            f.write('user_id,n_event,total_duratioin,total_weight_duration,ratio_of_duration,')
            for index in range(len(time_slot)):
                f.write('slot_%d,' %index)
            f.write('\n')
                
        #Ratio of dura and weight dura might affect answer?
        ratioofTDandTWD=float(total_weight_duration)/float(total_duration)

        f.write('%d,%d,%d,%d,%f,' %(i,n_events,total_duration,total_weight_duration,ratioofTDandTWD))
        for s in time_slot:
            f.write('%s,' %(str(s)))
        f.write('\n')
        
    return  max_id   #we know the last one


def preprocess_Xdata():
    start_id=-1
    #in Test
   # start_id=57158

    print("concatenate all files")
    path = "./" 
    allFiles=[]
    total_files=45
    #Correct is 75 change to 25 to faster
    global f
    f=open('dataTemp.csv', 'w')
    for i in range(total_files):
        buf = "data-%03d.csv"%(i+1)

        print("what is buf is preprocess_Xdata:",buf)

        buf=os.path.join(path,buf)
        print(i,'------------------------')
        start_id=preprocess_file(buf,start_id+1)
        allFiles.append(buf)
    f.close()
   
    print(allFiles)

def preprocess_Ydata():
    print("concatenate all files")
    path = "./" 
    allFiles=[]
    total_files=1
    #Correct is 45 change to 15 to faster

    #allFiles = glob.glob(os.path.join(path,"label-*.csv"))
    for i in range(total_files):
        buf = "label-%03d.csv"%(i+1)
        buf=os.path.join(path,buf)
        allFiles.append(buf)
        
    print(allFiles)
    df = pd.concat(map(pd.read_csv,allFiles))
    print(df.shape)
    print("save file")
    df.to_csv('all_label_Temp.csv', encoding='utf-8', index=False)
    print("save file all_label Successfully!")


def load_data(train_data_path,test_data_path):
    
    Train_Data = pd.read_csv(train_data_path, sep=',', header=None)
    Train_Data = np.array(Train_Data.values)
    
    X_train=Train_Data[:,:Train_Data.shape[1]-1]  # 0~(last column-1) are features
    Y_train=Train_Data[:,Train_Data.shape[1]-1]   # the last column is Y (target)

    #X_test = pd.read_csv(test_data_path, sep=',', header=None)
    #X_test = np.array(X_test.values)
    #print(X_test.shape)

    return (X_train, Y_train, X_test)

#write data
def save_data(data,output_file):
    print('write file....')
    path = Path(output_file)
    with open(output_file, 'w') as f:
        for i in data:
            #print(int(item[1]))  #[1] the probability of Spam. 
            f.write('%d\n' %i)
    return


def main():
    preprocess_Xdata()
    preprocess_Ydata()
    ''' 
    20171126
    process data first
    
    '''
    return 

    
    # Load feature and label
    #X_all, Y_all, X_test=load_data('alldata.csv')


    print('X_all:',X_all.shape)
    print('X_test:',X_test.shape)
    #Normalization
    X_all =  preprocessing.scale(X_all)
    X_test = preprocessing.scale(X_test)
    #train & validation & test
    #Xtrain, Xvalid, ytrain, yvalid =train_test_split(X_all,Y_all, test_size = 0.1)


    #y=options[method](Xtrain,Xvalid,ytrain,yvalid,X_test)
    #save_data(y,'predict.csv')

    print('Finished !')


main()

print("Test End")