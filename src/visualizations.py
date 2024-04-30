

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)  #
df = pd.read_csv('cleaned_cc_dataset.csv')


plt.figure(figsize=(8, 6))
sns.kdeplot(df['amt'], color='skyblue', shade=True)
plt.title('Density Plot of Amount (Log Scaled)')
plt.xlabel('amount')
plt.ylabel('Density')
plt.yscale('log')  # Set y-axis scale to logarithmic
plt.grid(True)

plt.figure(figsize=(8, 6))
is_fraud_counts=df.groupby('is_fraud')['is_fraud'].value_counts().reset_index()
plt.title('Class distribution')
plt.pie(is_fraud_counts['count'], labels=is_fraud_counts.columns, autopct='%1.1f%%')


'''plt.figure(figsize=(8, 6))
plt.hist(df['amt'], bins=50, color='skyblue')
plt.title('Distribution of the amount spent in a transaction (Log Scale)')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.yscale('log')  # Set y-axis scale to logarithmic
plt.grid(True)'''

ccTransCount = df.groupby('cc_num')['trans_num'].nunique().reset_index(name='counts')
fig, axes = plt.subplots(1, 2, figsize=(8, 6))
sns.boxplot( y='amt', data=df, ax=axes[0],log_scale=True)
plt.title('Boxplot of amt for legitimate and fraud transactions (log scaled)')
plt.yscale('log')  # Set y-axis scale to logarithmicaxes[0].set_title('Boxplot of Transaction Frequency per Card')
axes[1].hist(df['amt'], bins=50, color='skyblue')
axes[1].set_title('Distribution of the amount spent in a transaction (Log Scale)')
axes[1].set_xlabel('Amount')
axes[1].set_ylabel('Frequency')
axes[1].set_yscale('log')  # Set y-axis scale to logarithmic
axes[1].grid(True)
plt.tight_layout()


'''plt.figure(figsize=(8, 6))
plt.title('Scatter plot of lat X long')
plt.scatter(df['lat'], df['long'])
plt.figure(figsize=(8, 6))
plt.title('Scatter plot of merch_lat X merch_long')
plt.scatter(df['merch_lat'], df['merch_long'])
plt.figure(figsize=(8, 6))
plt.title('Scatter plot of merch_lat X merch_long fraud')
plt.scatter(df['merch_lat'].loc[df['is_fraud']==1], df['merch_long'].loc[df['is_fraud']==1])
plt.figure(figsize=(8, 6))
plt.title('Scatter plot of lat X long fraud')
plt.scatter(df['lat'].loc[df['is_fraud']==1], df['long'].loc[df['is_fraud']==1])'''

'''plt.figure(figsize=(8, 6))
location_cols = ['city', 'state', 'zip', 'lat', 'long', 'merch_lat', 'merch_long']
location_data = df[location_cols]
location_data.hist(bins=20, figsize=(12, 10))
plt.suptitle('Histograms of Location-Related Features')'''

plt.figure(figsize=(8, 6))
fraud_groupbyMerchant=df.loc[df['is_fraud']==1].groupby('merchant')['merchant'].value_counts().reset_index(name='counts')
plt.hist(fraud_groupbyMerchant['counts'],bins=20)

'''plt.figure(figsize=(8, 6))
plt.hist(ccTransCount['counts'], bins=50)
plt.title('Distribution of Transaction Frequency per Card')
plt.xlabel('Number of Transactions')
plt.ylabel('Frequency')'''



plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=ccTransCount, x='counts', bins=50)
plt.title('Distribution of Transaction Frequency per Card')
plt.xlabel('Number of Transactions')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
sns.boxplot(data=ccTransCount, y='counts')
plt.title('Boxplot of the number of Transactions per Card')
plt.ylabel('Number of Transactions')



plt.figure(figsize=(8, 6))
sns.heatmap(data=df.corr(numeric_only=True), cmap="YlGnBu", annot=True)
'''plt.figure(figsize=(8, 6))
sns.histplot(data=df[df['cc_num'] == 3592931352252641], x='amt', hue='is_fraud', bins=50)
plt.yscale('log')  # Set y-axis scale to logarithmic'''

# Convert 'trans_date_trans_time' to datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
plt.figure(figsize=(8, 6))
transactions_per_month = df.groupby(df['trans_date_trans_time'].dt.to_period('M')).size()
transactions_per_month.plot(kind='bar', figsize=(10, 6))
plt.title('Distribution of Transactions per Month')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
# Your code for grouping transactions
transactions_per_month = df.groupby([df['trans_date_trans_time'].dt.to_period('M'), 'is_fraud']).size().reset_index(name='count')
plt.figure(figsize=(12, 6))
sns.lineplot(x=transactions_per_month['trans_date_trans_time'].astype(str), y='count', data=transactions_per_month, hue='is_fraud')
plt.title('Transactions per Month (Separated by Fraud Status)')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure(figsize=(12, 6))
fraudulent_transactions = df[df['is_fraud'] == 1]
fraudulent_transactions['day_of_week'] = fraudulent_transactions['trans_date_trans_time'].dt.day_name()

# Group by day of the week and count the number of fraudulent transactions
fraudulent_transactions_per_day_of_week = fraudulent_transactions.groupby('day_of_week').size()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
fraudulent_transactions_per_day_of_week = fraudulent_transactions_per_day_of_week.reindex(days_order)
plt.figure(figsize=(10, 6))
fraudulent_transactions_per_day_of_week.plot(kind='bar', color='red')
plt.title('Number of Fraudulent Transactions per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Fraudulent Transactions')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()

fraudulent_transactions['hour_of_day'] = fraudulent_transactions['trans_date_trans_time'].dt.hour
fraudulent_transactions_per_hour = fraudulent_transactions.groupby('hour_of_day').size()
plt.figure(figsize=(10, 6))
fraudulent_transactions_per_hour.plot(kind='line', marker='o', color='red')
plt.title('Number of Fraudulent Transactions per Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Fraudulent Transactions')
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()


category_frequency = df.loc[df['is_fraud']==1]['category'].value_counts().reset_index(name='trans_count')
plt.figure(figsize=(8, 6))
plt.barh(category_frequency['category'], category_frequency['trans_count'])
plt.xlabel('Transaction Count')
plt.ylabel('Category')
plt.title('Fraudulent Transaction frequency per Category')
plt.gca().invert_yaxis()  # Invert y-axis to display categories from top to bottom

'''plt.figure(figsize=(10, 6))
sns.boxplot(x='is_fraud', y='amt', data=df)
plt.title('Boxplot of amt for legitimate and fraud transactions (log scaled)')
plt.yscale('log')  # Set y-axis scale to logarithmic'''


print()

plt.show()