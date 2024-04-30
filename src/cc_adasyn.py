import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.datasets import make_classification
import datetime
import time
from sklearn.preprocessing import LabelEncoder


def convert_to_datetime(unix_timestamp):
    return datetime.datetime.utcfromtimestamp(unix_timestamp)


def convert_to_unix(datetime_obj):
    return int(datetime_obj.timestamp())


pd.set_option('display.max_columns', None)  #
df = pd.read_csv('cleaned_cc_dataset.csv')
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
print('\ndata types : \n', df.dtypes)  # verifying that data types are all correct

df['unix_timestamps'] = df['trans_date_trans_time'].apply(convert_to_unix)
LE = LabelEncoder()
categories = ['first', 'last', 'gender','job','street','city','state','category','merchant','trans_num']
for label in categories:
    df[label] = LE.fit_transform(df[label])

X = df.drop(columns=['is_fraud','trans_date_trans_time','dob'])
y = df['is_fraud']
adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X, y)
#print(y_resampled.value_counts())

resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='is_fraud')], axis=1)
#resampled_df.to_csv('resampled_credit_card_data.csv', index=False)
print(resampled_df['is_fraud'].value_counts())
resampled_df.to_csv("cc_adasyn_dataset.csv", index=False)
