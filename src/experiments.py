import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)  #
df = pd.read_csv('C:\\Users\\Vasil\\Downloads\\fraudTrain2.csv')


#df = df.drop('Unnamed: 0', axis=1)
def calculate_rolling_sum(group):
    return group.rolling(window='30D', on='trans_date_trans_time')['amt'].sum()



'''stateC = df.groupby(['cc_num','merch_lat','merch_long'])['cc_num'].value_counts().reset_index(name='counts')
print(stateC.sort_values('cc_num',ascending=False))'''

merchantCount=df.groupby('cc_num')['merchant'].nunique().reset_index(name='counts')
print(merchantCount)

merchantCount=df.groupby('cc_num')['merchant'].value_counts().reset_index(name='counts')
print(merchantCount)

merchantCount = df.groupby('cc_num')['lat'].nunique().reset_index(name='counts')
print(merchantCount.sort_values('counts',ascending=False))


#exit(df.loc[df['is_fraud']==1])
#merchantFrauddistr=df.loc[df['is_fraud']==1].groupby('cc_num')['merchant'].value_counts().reset_index(name='counts').head()
#print("top 10 most frequent fraud merchant",merchantFrauddistr.sort_values('counts',ascending=False))
#merchantFrauddistr=df.loc[df['is_fraud']==1].groupby('cc_num')['merchant'].value_counts().reset_index(name='counts')
#print(merchantFrauddistr.sort_values('counts',ascending=False))
fraud_groupbyMerchant=df.loc[df['is_fraud']==1].groupby('merchant')['merchant'].value_counts().reset_index(name='counts')
print("frequencies of merchants in fraud transactions : \n",fraud_groupbyMerchant.sort_values('counts',ascending=False))
print("\n",fraud_groupbyMerchant.describe())
print(fraud_groupbyMerchant['merchant'])

print("cc_num shape : \n",df['cc_num'].shape)
print(df.shape)
print(df['cc_num'].nunique())  # almost 1000 unique cc_num, so, almost 1000 unique customers


'''print(df['category'].loc[df['is_fraud']==1].unique())
print(df.loc[df['is_fraud']==1,['lat','merch_lat','long','merch_long']].head(40))
print(df.loc[df['is_fraud']==0,['lat','merch_lat','long','merch_long']].head(40))'''


ccTransCount = df.groupby('cc_num')['trans_num'].nunique().reset_index(name='counts')
print(ccTransCount.sort_values('counts'))

ccAmtavg = df.groupby('cc_num')['amt'].mean().reset_index(name='amountAVG')
print("average spending amount per transaction :\n",ccAmtavg)

ccAmtmedian = df.groupby('cc_num')['amt'].median().reset_index(name='amountMedian')
print("median spending amount per transaction :\n",ccAmtmedian)

# Define cutoff date (e.g., transactions within the past 30 days are recent)
#cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=30)

'''cutoff_date = pd.to_datetime(df['trans_date_trans_time']).max()
# Add a new column for labeling transactions
df['transaction_label'] = ''
# Label transactions as historic or recent
for index, row in df.iterrows():
    transaction_date = pd.to_datetime(row['trans_date_trans_time'])
    if transaction_date >= cutoff_date - pd.Timedelta(days=30):
        df.at[index, 'transaction_label'] = 'recent'
    else:
        df.at[index, 'transaction_label'] = 'historic'''

'''amt_avg_perCC=df.groupby(['cc_num','transaction_label'])['amt'].mean().reset_index(name='amt_avg')
print(amt_avg_perCC)

std_per_cc_num = df.groupby('cc_num')['amt'].std().reset_index(name='std_dev')
print(std_per_cc_num)
#df = df.merge(std_per_cc_num.rename('std_dev'), left_on='cc_num', right_index=True)
amt_avg_perCC['amt_avg_diff']=amt_avg_perCC.groupby('cc_num')['amt_avg'].diff()
exit(amt_avg_perCC)'''
std_per_cc_num = df.groupby('cc_num')['amt'].std().reset_index(name='std_dev')
amt_avg_perCC=df.groupby(['cc_num','transaction_label'])['amt'].mean().reset_index(name='amt_avg')
recent_data = amt_avg_perCC[amt_avg_perCC['transaction_label'] == 'recent']
#print(recent_data)
historic_data = amt_avg_perCC[amt_avg_perCC['transaction_label'] == 'historic']
merged_data = pd.merge(recent_data, historic_data, on='cc_num', suffixes=('_recent', '_historic'))
recent_z_scores = (merged_data['amt_avg_recent'] - merged_data['amt_avg_historic']) / std_per_cc_num['std_dev']
#print(merged_data)
#flagged_users = merged_data[recent_z_scores > 1]
#print(flagged_users)

'''print(df[df['cc_num'] == 3592931352252641])
print(df[df['cc_num'] == 3592931352252641].describe())'''
cc_num_transactionsCount = df.groupby(['cc_num','transaction_label'])['cc_num'].value_counts().reset_index(name='counts')
baseline_transaction_frequency = cc_num_transactionsCount.loc[cc_num_transactionsCount['transaction_label']=='historic', ['cc_num', 'counts']]
recent_transaction_frequency = cc_num_transactionsCount.loc[cc_num_transactionsCount['transaction_label']=='recent', ['cc_num', 'counts']]
merged_frequencies = pd.merge(recent_transaction_frequency, baseline_transaction_frequency, on='cc_num', suffixes=('_recent', '_baseline'), how='outer')
merged_frequencies['deviations'] = (merged_frequencies['counts_recent'] / merged_frequencies['counts_baseline']) - 1
merged_frequencies = merged_frequencies.dropna()
threshold = 0.5  # Example: 50% change from baseline
flagged_cards = merged_frequencies[(merged_frequencies['deviations'] > threshold) | (merged_frequencies['deviations'] < -threshold)]
print("Cards with significant deviations in transaction frequency:")
print(flagged_cards)

numofTranspermonth=df.groupby('trans_date_trans_time')['trans_num'].value_counts().reset_index(name='counts')
print(numofTranspermonth)

print("median amount per fraudulent transaction : ",df['amt'].loc[df['is_fraud'] == 1].median())
print("\nmedian amount per legitimate transaction : ",df['amt'].loc[df['is_fraud'] == 0].median())
print("amt statistics for fraudulent transactions : ",df['amt'].loc[df['is_fraud'] == 1].describe())
print("\namt statistics for legitimate transactions : ",df['amt'].loc[df['is_fraud'] == 0].describe())

# na paiksw me is_fraud=1

#fraud_numOfcc=df.groupby('is_fraud')['cc_num'].value_counts()
fraudTransactions=df.loc[df['is_fraud'] == 1]
print("\nnumber of unique cc_num with fraudulent activity : ", fraudTransactions['cc_num'].nunique())  #  762 unique cc_num almost 80% of total number of cc_num


df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
transactions_per_month = df.groupby([df['trans_date_trans_time'].dt.to_period('M'),'is_fraud']).size().reset_index(name='count')
print("transactions per month : ",transactions_per_month)


fraudulent_transactions = df[df['is_fraud'] == 1]
fraudulent_transactions_per_day = fraudulent_transactions.groupby(fraudulent_transactions['trans_date_trans_time'].dt.date).size()
print("\n transactions per day : ",fraudulent_transactions_per_day)

pd.set_option('display.max_rows', 250)  # Change 10 to the desired maximum number of rows

df_updated = pd.read_csv('updated_dataset.csv')
user_groups = df.groupby('cc_num')
#print(user_groups)
#print(df_updated.sort_values(['cc_num','trans_date_trans_time']).head(20))

#print(df_updated[df_updated['cc_num']==60416207185].sort_values('trans_date_trans_time').head(40))
#print(df_updated[df_updated['cc_num'] == 60423098130][['cc_num', 'trans_date_trans_time', 'transaction_frequency']].sort_values('trans_date_trans_time').head(250))

'''category_frequency=df['category'].value_counts()
print(category_frequency)
category_frequency = df.loc[df['is_fraud']==1]['category'].value_counts().reset_index(name='trans_count')
print(category_frequency)'''
#pd.set_option('display.max_columns', 10)

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])  # Convert to datetime if not already

df.sort_values(by='trans_date_trans_time', inplace=True)


# Define the window as a Timedelta object
window = pd.Timedelta(days=30)
# Apply the rolling_avg function to each group
#df['user_baseline'] = df.groupby('cc_num', as_index=False, group_keys=False).apply( lambda x: x.rolling(window='30D', min_periods=1,on='trans_date_trans_time')['amt'].mean().shift()).reset_index(level=0, drop=True)
#df['total_amount'] = df.groupby('cc_num', as_index=False, group_keys=False).apply(lambda x: x.rolling(window='1h', on='trans_date_trans_time')['amt'].sum().shift(-1)).reset_index(level=0,drop=True)
#df['user_baseline'] = df.groupby('cc_num')['amt'].transform(lambda x: x.rolling(30).mean().shift())
#print(df[df['cc_num'] == 60423098130][[ 'trans_date_trans_time','user_baseline','amt']].sort_values('trans_date_trans_time').head(250))


df = pd.read_csv('updated3_dataset.csv')
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df.sort_values(by='trans_date_trans_time', inplace=True)
df['moving_avg_amt'] = df.groupby('cc_num').apply(lambda x: x.rolling(window='30D', on='trans_date_trans_time')['amt'].mean()).reset_index(level=0, drop=True)
df['std_dev_amt'] = df.groupby('cc_num').apply(lambda x: x.rolling(window='30D', on='trans_date_trans_time')['amt'].std()).reset_index(level=0, drop=True)
df['transaction_velocity'] = ((df['amt'] - df['moving_avg_amt']) > (2 * df['std_dev_amt'])).astype(int)
print(df[df['cc_num'] == 60423098130][[ 'trans_date_trans_time','transaction_frequency','transaction_velocity','moving_avg_amt','amt']].sort_values('trans_date_trans_time').head(250))

exit()
'''df33['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])  # Convert to datetime if not already

df33.sort_values(by='trans_date_trans_time', inplace=True)

# Calculate the number of transactions within each rolling window
df33['transactions_count'] = df33.groupby('cc_num').apply(
    lambda x: x.rolling(window='30D', on='trans_date_trans_time')['trans_date_trans_time'].count()).reset_index(level=0,
                                                                                                                drop=True)

# Calculate the average number of transactions per cc_num over a 30-day period
df33['moving_avg_transaction_count'] = df33.groupby('cc_num').apply(
    lambda x: x.rolling(window='30D', on='trans_date_trans_time')['transactions_count'].mean()).reset_index(level=0,
                                                                                                            drop=True)
# Calculate standard deviation of transaction count
df33['std_dev_transaction_count'] = df33.groupby('cc_num').apply(
    lambda x: x.rolling(window='30D', on='trans_date_trans_time')['transactions_count'].std()).reset_index(level=0,
                                                                                                           drop=True)

# Calculate transaction frequency flags
df33['transaction_frequency'] = ((df33['transactions_count'] - df33['moving_avg_transaction_count']) > (
            2 * df33['std_dev_transaction_count'])).astype(int)
'''
print(df33[df33['cc_num'] == 60423098130][[ 'trans_date_trans_time','transaction_frequency','transactions_count','moving_avg_transaction_count','amt']].sort_values('trans_date_trans_time').head(250))