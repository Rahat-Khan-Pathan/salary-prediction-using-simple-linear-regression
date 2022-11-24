import pandas as pd
import time
import json

def load_model():
    print("Loading Trained Data from trained_data.txt...")
    time.sleep(2)
    with open('./trained/trained_data.txt') as f:
        data = f.read()
        js = json.loads(data)
    print("Loaded!\n")
    return js

trained_data=load_model()
while True:
    inp = int(input("Enter Years of Experience: "))
    m=trained_data["m"]
    c=trained_data["c"]
    ans = (m*inp)+c
    print(f"Estimated Salary : {format(ans,'.2f')}\n")