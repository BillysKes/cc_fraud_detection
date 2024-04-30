

1. [Introduction](#Introduction)
2. [Dataset Description](#dataset-description)
3. [Exploratory Data Analysis (EDA)](#eda)
   1. [Descriptive Statistics](#cleaning-python)
   2. [Visualization of Data](#cleaning-sql)
   3. [Feature Engineering](#feature-eng)
4. [Adaptive Synthetic Sampling (ADASYN)](#adasyn)
5. [Long short-term memory(LSTM)](#)
6. [Conclusions](#conclusions)




# 1. Introduction

Credit card fraud poses significant financial risks to both consumers and financial institutions. Fraudsters continually devise new methods to exploit vulnerabilities in payment systems and so there is a pressing need for robust fraud detection mechanisms to safeguard against fraudulent activities. By analyzing patterns and anomalies within a Credit Card Transactions Fraud Detection Dataset, we aim to develop predictive models capable of identifying fraudulent transactions accurately. These models will be trained on historical data, enabling them to learn patterns indicative of fraudulent behavior and then, they will be evaluated on their ability to generalize to unseen data.


# 2. Dataset Description

The dataset utilized for this analysis contains detailed information about each credit card transactions, including transaction date and time, credit card number, merchant details, transaction amount, and geographical information. You can find more information here : https://www.kaggle.com/datasets/kartik2112/fraud-detection

# 3. Exploratory Data Analysis (EDA)


## 3.1 Descriptive Statistics
```
              cc_num         amt         zip         lat        long  \
count  1.296675e+06  1296675.00  1296675.00  1296675.00  1296675.00   
mean   4.171920e+17       70.35    48800.67       38.54      -90.23   
std    1.308806e+18      160.32    26893.22        5.08       13.76   
min    6.041621e+10        1.00     1257.00       20.03     -165.67   
25%    1.800429e+14        9.65    26237.00       34.62      -96.80   
50%    3.521417e+15       47.52    48174.00       39.35      -87.48   
75%    4.642255e+15       83.14    72042.00       41.94      -80.16   
max    4.992346e+18    28948.90    99783.00       66.69      -67.95   

         city_pop     unix_time   merch_lat  merch_long    is_fraud  
count  1296675.00  1.296675e+06  1296675.00  1296675.00  1296675.00  
mean     88824.44  1.349244e+09       38.54      -90.23        0.01  
std     301956.36  1.284128e+07        5.11       13.77        0.08  
min         23.00  1.325376e+09       19.03     -166.67        0.00  
25%        743.00  1.338751e+09       34.73      -96.90        0.00  
50%       2456.00  1.349250e+09       39.37      -87.44        0.00  
75%      20328.00  1.359385e+09       41.96      -80.24        0.00  
max    2906700.00  1.371817e+09       67.51      -66.95        1.00  


        trans_date_trans_time           merchant       category        first  \
count                1296675            1296675        1296675      1296675   
unique               1274791                693             14          352   
top      2019-04-22 16:02:01  fraud_Kilback LLC  gas_transport  Christopher   
freq                       4               4403         131659        26669   

           last   gender                      street        city    state  \
count   1296675  1296675                     1296675     1296675  1296675   
unique      481        2                         983         894       51   
top       Smith        F  0069 Robin Brooks Apt. 695  Birmingham       TX   
freq      28794   709863                        3123        5617    94876   

                      job         dob                         trans_num  
count             1296675     1296675                           1296675  
unique                494         968                           1296675  
top     Film/video editor  1977-03-23  0b242abb623afc578575680df30655b9  
freq                 9779        5636                                 1

amt statistics for fraudulent transactions :  count    7506.000000
mean      531.320092
std       390.560070
min         1.060000
25%       245.662500
50%       396.505000
75%       900.875000
max      1376.040000
Name: amt, dtype: float64

amt statistics for legitimate transactions :  count    1.289169e+06
mean     6.766711e+01
std      1.540080e+02
min      1.000000e+00
25%      9.610000e+00
50%      4.728000e+01
75%      8.254000e+01
max      2.894890e+04
Name: amt, dtype: float64

```

## 3.2 Visualization of Data


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/cc73219e-c675-4607-9d7a-4600fbbd7ea6)




![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/e5b530a8-6413-4df9-a3de-7bc184a6dbfa)

- There is a high pressence of outliers of the amt variable on the legitimate transactions.


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/582074ad-4d77-4f38-b578-3998bed06f26)

- The high majority of cardholders spent less than 500$ per transaction.


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/56ce6e02-c866-45df-a413-cc49d347581d)
  


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/9581fb41-a72b-45bb-b330-d576bc44b28d)





- 50% of the credit cards have made less than 1000 transactions on the timeframe of 18 months(1.5 years)

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/1012c594-1270-44de-9b47-2b5d41157e76)


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/d9c1abf3-9d8d-44db-bea9-c6fce9f74c61)

- We notice a high increase of transactions happening on December. This is probably because of christmas holidays.

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/ee7a10a6-b76b-48f8-b95d-2e2ef968d224)


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/c440f2bd-3cb9-49be-aa80-688b25bf88d4)

- There are significantly more fraudulent transactions happening at night and more specifically at the time between 21:00 until 04:00




## 3.3 Feature Engineering

### trans_freq_flag

trans_freq_flag : flags the transaction if the total number of transactions made by the credit card user deviates a lot compared to the usual(past 30 days) total number of transactions the credit card user makes.



```
def create_trans_freq_flag(data):
    data.sort_values(by='trans_date_trans_time', inplace=True)

    data['transactions_count'] = data.groupby('cc_num').apply(
        lambda x: x.rolling(window='30D', on='trans_date_trans_time')['trans_date_trans_time'].count()).reset_index(level=0, drop=True)

    data['moving_avg_transaction_count'] = data.groupby('cc_num').apply(lambda x: x.rolling(window='30D',on='trans_date_trans_time')['transactions_count'].mean()).reset_index(level=0, drop=True)
    data['std_dev_transaction_count'] = data.groupby('cc_num').apply(
        lambda x: x.rolling(window='30D',on='trans_date_trans_time')['transactions_count'].std()).reset_index(level=0, drop=True)

    data['trans_freq_flag'] = ((data['transactions_count'] - data['moving_avg_transaction_count']) > (2 * data['std_dev_transaction_count'])).astype(int)

    data.drop(['transactions_count', 'moving_avg_transaction_count', 'std_dev_transaction_count'], axis=1, inplace=True)

    return data
```

This function creates the trans_freq_flag feature. For every transaction made by a credit card, it calculates the 30-days moving average of the total number of transactions made(moving_avg_transaction_count) and also it calculates the standard deviation of moving_avg_transaction_count inside that 30-days window. Then it flags every transaction of the credit card where the difference of the total number of transactions and the moving average number of transactions made is 2 times higher than the standard deviation.


### trans_amt_flag

trans_amt_flag : flags the transaction if the total amount of money spent with the the credit card deviates a lot compared to the usual(past 30 days) amount of money the credit card user spends
```
def create_trans_amt_flag(data):
    data.sort_values(by='trans_date_trans_time', inplace=True)

    df['moving_avg_amt'] = df.groupby('cc_num').apply(
        lambda x: x.rolling(window='30D', on='trans_date_trans_time')['amt'].mean()).reset_index(level=0, drop=True)

    df['std_dev_amt'] = df.groupby('cc_num').apply(
        lambda x: x.rolling(window='30D', on='trans_date_trans_time')['amt'].std()).reset_index(level=0, drop=True)
    df['trans_amt_flag'] = ((df['amt'] - df['moving_avg_amt']) > (2 * df['std_dev_amt'])).astype(int)

    data.drop(['moving_avg_amt', 'std_dev_amt'], axis=1, inplace=True)

    return data
```

This function creates the trans_amt_flag feature. For every transaction made by a credit card, it calculates the 30-days moving average of the amount of money were spent and it also calculates the standard deviation of the amount of money that were spent within the past 30 days. Then it flags every transaction where the difference of the amount of the transaction and the moving average amount is 2 times higher than the standard deviation.


# 4. Adaptive Synthetic Sampling (ADASYN)

We apply the Adaptive Synthetic Sampling (ADASYN), which is an oversampling method that generates synthetic samples for the minority class(fraud) balancing the dataset and improving the classification accuracy.

```
def convert_to_unix(datetime_obj):
    return int(datetime_obj.timestamp())


df['unix_timestamps'] = df['trans_date_trans_time'].apply(convert_to_unix)
LE = LabelEncoder()
categories = ['first', 'last', 'gender','job','street','city','state','category','merchant','trans_num']
for label in categories:
    df[label] = LE.fit_transform(df[label])

X = df.drop(columns=['is_fraud','trans_date_trans_time','dob'])
y = df['is_fraud']
adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X, y)
```
Categorical features of the dataset are encoded as numericals with the label encoding technique before applying the algorithm. We also encode the datetime feature as unix time, which is essentially a measurement of time by the number of non-leap seconds that have elapsed since 00:00:00 UTC on 1 January 1970.

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/ae8c2178-d120-4760-ba75-f945742ada01)


# 5. Long short-term memory(LSTM)

Long short-term memory (LSTM) network is a recurrent neural network (RNN) that deals with the vanishing gradient problem that is present in traditional RNNs, and it performs very well with sequential data. The model is been implemented with Keras. 


```
selected_features = ['amt', 'merchant', 'category', 'gender', 'lat', 'long','transaction_velocity','transaction_frequency']
data = df[selected_features]

scaler = StandardScaler()
data[['amt', 'lat', 'long']] = scaler.fit_transform(data[['amt', 'lat', 'long']])

batch_size = 32
sequence_length = 10
generator = TimeseriesGenerator(data.values, df['is_fraud'].values, length=sequence_length, batch_size=batch_size)

X = df[selected_features].values
y = df['is_fraud'].values

sequence_length = 10
X_sequences = []
y_sequences = []


accuracies = []  # List to store accuracies of each fold
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True)

for train_index, val_index in tqdm(kf.split(X_sequences), total=k):
    X_train, X_val = X_sequences[train_index], X_sequences[val_index]
    y_train, y_val = y_sequences[train_index], y_sequences[val_index]

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    _, accuracy = model.evaluate(X_val, y_val)
    accuracies.append(accuracy)

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("Mean Accuracy:", mean_accuracy)
print("Standard Deviation of Accuracy:", std_accuracy)
```

The features are being scaled with the StandardScaler and also the model was trained for 10 epochs with a batch size of 32. Specifically, the model consists of an LSTM layer with 64 units, followed by a dense layer with a single unit and a sigmoid activation function, and it is compiled with the adam optimizer and binary cross-entropy loss function. Also, the sequence length is 10, meaning that each sequence will contain 10 transactions in sequential order. Finally, 5-fold cross validation is used for evaluating the model performance.


