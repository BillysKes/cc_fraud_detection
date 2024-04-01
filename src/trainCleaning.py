

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)  #
df = pd.read_csv('C:\\Users\\Vasil\\Downloads\\fraudTrain.csv')

df = df.drop('Unnamed: 0', axis=1)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time'] = df['trans_date_trans_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

print(df.head())
print('\ndata types : \n', df.dtypes)  # verifying that data types are all correct
print('\nmissing values : \n', df.isna().sum())  # missing values detection
print('\nduplicates :\n', df[df.duplicated()])  # duplicates detection
print('\n\n', df['is_fraud'].unique())  # Inconsistencies detection - in one column
print("\n\n", round(df.describe(), 2))  # statistics information
print('\n\n', df.describe(include='object'))  # statistics for categorical variables

