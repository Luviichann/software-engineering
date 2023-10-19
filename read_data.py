import pandas as pd

def scan_data(path):
    df = pd.read_csv(path)
    features = df.drop(['label'],axis=1)
    labels = df.loc[:,'label']-1
    return features,labels