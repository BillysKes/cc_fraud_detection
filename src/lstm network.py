
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm


df = pd.read_csv('cc_adasyn_dataset.csv')

selected_features = ['amt', 'merchant', 'category', 'gender', 'lat', 'long', 'transaction_velocity', 'transaction_frequency', 'cc_num']
data = df[selected_features]
data[['amt', 'lat', 'long']] = StandardScaler().fit_transform(data[['amt', 'lat', 'long']])

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    cardholders = data['cc_num'].unique()
    for cardholder in cardholders:
        cardholder_data = data[data['cc_num'] == cardholder]
        cardholder_data = cardholder_data.sort_values(by='timestamp')  # Ensure the data is sorted by time
        for i in range(len(cardholder_data) - seq_length):
            seq = cardholder_data.iloc[i:i+seq_length][['amt', 'merchant', 'category', 'gender', 'lat', 'long', 'transaction_velocity', 'transaction_frequency']].values
            label = cardholder_data.iloc[i+seq_length]['is_fraud']
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)


sequence_length = 10
X_sequences, y_sequences = create_sequences(data, sequence_length)

k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True)

accuracies = []  # List to store accuracies of each fold

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
