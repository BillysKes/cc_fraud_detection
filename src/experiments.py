

import pandas as pd


pd.set_option('display.max_columns', None)  #
df = pd.read_csv('C:\\Users\\Vasil\\Downloads\\fraudTrain.csv')

df = df.drop('Unnamed: 0', axis=1)



cc_num_transactionsCount=df.groupby('cc_num')['cc_num'].value_counts()
print(cc_num_transactionsCount)


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

ccTransCount = df.groupby('cc_num')['trans_num'].count().reset_index(name='counts')
print(ccTransCount.sort_values('counts'))