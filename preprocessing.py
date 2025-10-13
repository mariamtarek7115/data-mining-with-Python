import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler


data = {
    'Region': ['India', 'Brazil', 'USA', 'Brazil', 'USA', 'India', 'Brazil', 'India', 'USA', 'India'],
    'Age': [49, 32, 35, 43, 45, 40, np.nan, 53, 55, 42],
    'Income': [86400, 57600, 64800, 73200, np.nan, 69600, 62400, 94800, 99600, 80400],
    'Online Shopper': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}
df=pd.DataFrame(data)
print(df)

#handling missing numerical data
numerical_cols=['Age', 'Income']
dfNum=df[numerical_cols]
imputer=SimpleImputer(strategy='mean')
df[numerical_cols]=imputer.fit_transform(dfNum)
print(df)

#handling missing categorical data
categorical_cols=['Region', 'Online Shopper']
dfCat=df[categorical_cols]
imputer=SimpleImputer(strategy='most_frequent')
df[categorical_cols]=imputer.fit_transform(dfCat)
print(df)

#discretizer
est= KBinsDiscretizer(n_bins=3,strategy='uniform',encode='ordinal')
df[numerical_cols] = est.fit_transform(df[numerical_cols])
print(df)

#normalization
scaler=MinMaxScaler()
df_scale=scaler.fit_transform(dfNum)
df_scale=pd.DataFrame(df_scale,columns=numerical_cols)
print(df_scale)



