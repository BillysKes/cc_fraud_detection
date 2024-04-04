

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


df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df.sort_values(by=['cc_num', 'trans_date_trans_time'], inplace=True)
df['transaction_frequency'] = 0
first_transaction_dates = {}
for cc_num, group in df.groupby('cc_num'):
    first_transaction_date = group['trans_date_trans_time'].min()
    first_transaction_dates[cc_num] = first_transaction_date

    transaction_count = 0
    for index, row in group.iterrows():
        days_diff = (row['trans_date_trans_time'] - first_transaction_date).days
        if days_diff is not None and days_diff > 30:
            first_transaction_date = row['trans_date_trans_time']
            transaction_count = 1
        else:
            transaction_count += 1
        df.at[index, 'transaction_frequency'] = transaction_count

#df.to_csv("updated_dataset.csv", index=False)


