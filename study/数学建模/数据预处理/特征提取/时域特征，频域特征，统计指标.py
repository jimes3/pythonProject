import pandas as pd

df =  pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                  usecols=['lstat','rm', 'rad','chas'])
#########################    统计指标         #######################
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def extract_stats(arr):
    return {
        'mean': np.mean(arr),
        'std': np.std(arr, ddof=1),
        'var': np.var(arr, ddof=1),
        'skewness': skew(arr),
        'kurtosis': kurtosis(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'range': np.ptp(arr),  # max - min
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75)
    }

features = extract_stats(df['rm'])
print(pd.Series(features))