import pandas as pd
import json
import numpy as np
import time
import math
import matplotlib.pyplot as plt

def pre_processing(df):
    print("Processing Test Data...")
    time.sleep(2)
    df_headers=df.columns.values
    X=df[df_headers[0]]
    y=df[df_headers[1]]
    X_test=X.truncate(4900,4999)
    y_test=y.truncate(4900,4999)
    X_train=X.truncate(0,4899)
    y_train=y.truncate(0,4899)
    print("Done!\n")
    return X_train,X_test, y_train,y_test

def read_data():
    print("Reading Test Data...")
    time.sleep(2)
    df = pd.read_csv("./datasets/salary.csv")
    print("Done!\n")
    return df

def load_model():
    print("Loading Trained Data from trained_data.txt...")
    time.sleep(2)
    with open('./trained/trained_data.txt') as f:
        data = f.read()
        js = json.loads(data)
    print("Loaded!\n")
    return js

def test_model(df,X_test,y_test,trained_data):
    upper=0
    lower=0
    print(f"Testing with {len(X_test)} data...")
    y_predict_list=[]
    m=trained_data["m"]
    c=trained_data["c"]
    meanY=trained_data["meanY"]
    for indx in range(len(X_test)):
        y_pred=(m*X_test[indx])+c
        y_predict_list.append(y_pred)
        upper+=math.pow((y_test[indx]-y_pred),2)
        lower+=math.pow((y_test[indx]-meanY),2)
    r_square=1-(upper/lower)
    print("Done!\n")
    return r_square,y_predict_list

df=read_data()
X_train,X_test,y_train,y_test=pre_processing(df)
trained_data=load_model()
r_square,y_pred=test_model(df,X_train,y_train,trained_data)
print(f"The value of R Square - {format(r_square,'.3f')}")
plt.scatter(X_train,y_train,color='g')
plt.plot(X_train,y_pred,color='r')
plt.show()  
