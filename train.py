#%%

import numpy as np
import pandas as pd
import time
import json
import math
import matplotlib.pyplot as plt

def read_data():
    print("Reading Data from CSV...")
    time.sleep(2)
    df = pd.read_csv("./datasets/salary.csv")
    print("Done!\n")
    return df

def pre_processing(df):
    print("Processing Data. Splitting Training Data and Testing Data. Taking 100 Data for Testing..")
    time.sleep(3)
    df_headers=df.columns.values
    X=df[df_headers[0]]
    y=df[df_headers[1]]
    X=X.truncate(0,4899)
    y=y.truncate(0,4899)
    print("Done!")
    return X, y

def cal_mean(data):
    s=0
    for val in data:
        s+=val
    mean=s/len(data)
    return mean

def cal_slope(X,y,meanX,meanY):
    # sum_X=0
    # sum_X_sq=0
    # sum_y=0
    # sum_y_sq=0
    # sum_X_y=0
    # for val in X:
    #     sum_X+=val
    #     sum_X_sq+=math.pow(val,2)
    # for val in y:
    #     sum_y+=val
    #     sum_y_sq+=math.pow(val,2)
    # for indx in range(len(X)):
    #     sum_X_y+=(X[indx]*y[indx])
    
    upper=0
    lower=0
    for indx in range(len(X)):
        upper+=((X[indx]-meanX)*(y[indx]-meanY))
        lower+=(math.pow((X[indx]-meanX),2))
    slope=upper/lower
    # slope=((len(X)*sum_X_y)-(sum_X*sum_y))/((len(X)*sum_X_sq)-(math.pow(sum_X,2)))
    # c=((sum_y*sum_X_sq)-(sum_X*sum_X_y))/((len(X)*sum_X_sq)-(pow(sum_X,2)))
    return slope

def train_model(df,X,y):
    print(f"Training Model with {len(X)} Data...")
    time.sleep(3)
    meanX=cal_mean(X)
    meanY=cal_mean(y)
    m=cal_slope(X,y,meanX,meanY)
    print(m)
    c=meanY-(m*meanX)
    print(c)
    trained_data={}
    trained_data["meanX"]=meanX
    trained_data["meanY"]=meanY
    trained_data["m"]=m
    trained_data["c"]=c
    with open('./trained/trained_data.txt', 'w') as convert_file:
        convert_file.write(json.dumps(trained_data))
    print("Model Trained. Trained Data Extracted in trained_data.txt File. Now You Can Test Your Model with Test Data!\n")
        

df=read_data()
X,y=pre_processing(df)
train_model(df,X,y)





# %%
