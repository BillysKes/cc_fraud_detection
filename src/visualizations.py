

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)  #
df = pd.read_csv('C:\\Users\\Vasil\\Downloads\\fraudTrain.csv')

df = df.drop('Unnamed: 0', axis=1)

plt.figure()
sns.kdeplot(df['amt'], color='skyblue', shade=True)
plt.title('Density Plot of Amount (Log Scaled)')
plt.xlabel('amount')
plt.ylabel('Density')
plt.yscale('log')  # Set y-axis scale to logarithmic
plt.grid(True)

plt.figure()
is_fraud_counts=df.groupby('is_fraud')['is_fraud'].value_counts().reset_index()
plt.title('Class distribution')
plt.pie(is_fraud_counts['count'], labels=is_fraud_counts.columns, autopct='%1.1f%%')


plt.figure(figsize=(8, 6))
plt.hist(df['amt'], bins=50, color='skyblue')
plt.title('Distribution of amount spent (Log Scale)')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.yscale('log')  # Set y-axis scale to logarithmic
plt.grid(True)

plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='amt', hue='is_fraud', bins=50)
plt.title('Distribution of amount spent (Log Scale)')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.yscale('log')  # Set y-axis scale to logarithmic


'''plt.figure(figsize=(8, 6))
plt.title('Scatter plot of lat X long')
plt.scatter(df['lat'], df['long'])
plt.figure(figsize=(8, 6))
plt.title('Scatter plot of merch_lat X merch_long')
plt.scatter(df['merch_lat'], df['merch_long'])'''
plt.figure(figsize=(8, 6))
plt.title('Scatter plot of merch_lat X merch_long fraud')
plt.scatter(df['merch_lat'].loc[df['is_fraud']==1], df['merch_long'].loc[df['is_fraud']==1])
plt.figure(figsize=(8, 6))
plt.title('Scatter plot of lat X long fraud')
plt.scatter(df['lat'].loc[df['is_fraud']==1], df['long'].loc[df['is_fraud']==1])

plt.figure(figsize=(8, 6))
location_cols = ['city', 'state', 'zip', 'lat', 'long', 'merch_lat', 'merch_long']
location_data = df[location_cols]
location_data.hist(bins=20, figsize=(12, 10))
plt.suptitle('Histograms of Location-Related Features')

plt.figure(figsize=(8, 6))
fraud_groupbyMerchant=df.loc[df['is_fraud']==1].groupby('merchant')['merchant'].value_counts().reset_index(name='counts')
plt.hist(fraud_groupbyMerchant['counts'],bins=20)

sns.heatmap(data=df.corr(numeric_only=True), cmap="YlGnBu", annot=True)


plt.show()