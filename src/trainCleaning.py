

import numpy as np
import pandas as pd

from tqdm import tqdm

# Step 3: Sliding Window Calculation
def calculate_transaction_velocity(data):
    data.sort_values(by='trans_date_trans_time', inplace=True)

    data['moving_avg_amt'] = data.groupby('cc_num').apply(
        lambda x: x.rolling(window='30D', on='trans_date_trans_time')['amt'].mean()).reset_index(level=0, drop=True)

    data['std_dev_amt'] = data.groupby('cc_num').apply(
        lambda x: x.rolling(window='30D', on='trans_date_trans_time')['amt'].std()).reset_index(level=0, drop=True)
    data['transaction_velocity'] = ((data['amt'] - data['moving_avg_amt']) > (2 * data['std_dev_amt'])).astype(int)

    data.drop(['moving_avg_amt', 'std_dev_amt'], axis=1, inplace=True)

    return data

def calculate_transaction_frequency(data):
    data.sort_values(by='trans_date_trans_time', inplace=True)

    # Calculate the number of transactions within each rolling window
    data['transactions_count'] = data.groupby('cc_num').apply(
        lambda x: x.rolling(window='30D', on='trans_date_trans_time')['trans_date_trans_time'].count()).reset_index(level=0, drop=True)

    # Calculate the average number of transactions per cc_num over a 30-day period
    data['moving_avg_transaction_count'] = data.groupby('cc_num').apply(lambda x: x.rolling(window='30D',on='trans_date_trans_time')['transactions_count'].mean()).reset_index(level=0, drop=True)
    # Calculate standard deviation of transaction count
    data['std_dev_transaction_count'] = data.groupby('cc_num').apply(
        lambda x: x.rolling(window='30D',on='trans_date_trans_time')['transactions_count'].std()).reset_index(level=0, drop=True)

    # Calculate transaction frequency flags
    data['transaction_frequency'] = ((data['transactions_count'] - data['moving_avg_transaction_count']) > (2 * data['std_dev_transaction_count'])).astype(int)

    # Drop intermediate columns
    data.drop(['transactions_count', 'moving_avg_transaction_count', 'std_dev_transaction_count'], axis=1, inplace=True)

    return data


pd.set_option('display.max_columns', None)  #
df = pd.read_csv('cleaned_cc_dataset.csv')

#df = df.drop('Unnamed: 0', axis=1)
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

'''df.sort_values(by=['cc_num', 'trans_date_trans_time'], inplace=True)
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
        df.at[index, 'transaction_frequency'] = transaction_count '''

#time_window = pd.Timedelta(hours=1)  # Adjust as needed
# Step 6: Output
#df33 = pd.read_csv('updated3_dataset.csv')
#df33['trans_date_trans_time'] = pd.to_datetime(df33['trans_date_trans_time'])

#df_updated=calculate_transaction_velocity(df)

df_updatedFreq=calculate_transaction_frequency(df)
df_updatedFreq.to_csv("cleaned_cc_dataset.csv", index=False)


