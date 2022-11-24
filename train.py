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
    upper=0
    lower=0
    for indx in range(len(X)):
        upper+=((X[indx]-meanX)*(y[indx]-meanY))
        lower+=(math.pow((X[indx]-meanX),2))
    slope=upper/lower
    return slope

def train_model(df,X,y):
    print(f"Training Model with {len(X)} Data...")
    time.sleep(3)
    meanX=cal_mean(X)
    meanY=cal_mean(y)
    m=cal_slope(X,y,meanX,meanY)
    c=meanY-(m*meanX)
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
