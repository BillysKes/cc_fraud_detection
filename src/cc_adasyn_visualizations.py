
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)  #
df = pd.read_csv('cc_adasyn_dataset.csv')

print(df.shape)
print('\ndata types : \n', df.dtypes)  # verifying that data types are all correct
print(df.sort_values('unix_timestamps'))
plt.figure(figsize=(8, 6))
is_fraud_counts=df.groupby('is_fraud')['is_fraud'].value_counts().reset_index()
plt.title('Class distribution')
plt.pie(is_fraud_counts['count'], labels=is_fraud_counts.columns, autopct='%1.2f%%')

plt.show()
