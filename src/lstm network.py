
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

pd.set_option('display.max_columns', None)  #
df = pd.read_csv('cc_adasyn_dataset.csv')

selected_features = ['amt', 'merchant', 'category', 'gender', 'lat', 'long','transaction_velocity','transaction_frequency']
data = df[selected_features]

scaler = StandardScaler()
data[['amt', 'lat', 'long']] = scaler.fit_transform(data[['amt', 'lat', 'long']])
 
# Define sequence length and batch size
batch_size = 32
sequence_length = 10
generator = TimeseriesGenerator(data.values, df['is_fraud'].values, length=sequence_length, batch_size=batch_size)

X = df[selected_features].values
y = df['is_fraud'].values

sequence_length = 10
X_sequences = []
y_sequences = []

for i in range(len(X) - sequence_length):
    X_sequences.append(X[i:i+sequence_length])
    y_sequences.append(y[i+sequence_length-1])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

accuracies = []  # List to store accuracies of each fold
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True)

for train_index, val_index in tqdm(kf.split(X_sequences), total=k):
    X_train, X_val = X_sequences[train_index], X_sequences[val_index]
    y_train, y_val = y_sequences[train_index], y_sequences[val_index]

    # Build and compile the model (same as before)
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the validation fold
    _, accuracy = model.evaluate(X_val, y_val)
    accuracies.append(accuracy)

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("Mean Accuracy:", mean_accuracy)
print("Standard Deviation of Accuracy:", std_accuracy)