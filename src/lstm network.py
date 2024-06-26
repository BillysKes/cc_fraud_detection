import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cc_adasyn_dataset.csv')

# Select features
selected_features = ['amt', 'merchant', 'category', 'gender', 'lat', 'long', 'transaction_velocity', 'transaction_frequency', 'cc_num', 'unix_timestamps', 'is_fraud']
data = df[selected_features]

# Standardize selected columns
columns_to_standardize = ['amt', 'lat', 'long', 'merchant', 'category']
data.loc[:, columns_to_standardize] = StandardScaler().fit_transform(data[columns_to_standardize])

# Function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    cardholders = data['cc_num'].unique()
    for cardholder in cardholders:
        cardholder_data = data[data['cc_num'] == cardholder]
        cardholder_data = cardholder_data.sort_values(by='unix_timestamps')  # Ensure the data is sorted by time
        for i in range(len(cardholder_data) - seq_length):
            seq = cardholder_data.iloc[i:i+seq_length][['amt', 'merchant', 'category', 'gender', 'lat', 'long', 'transaction_velocity', 'transaction_frequency']].values
            label = cardholder_data.iloc[i+seq_length]['is_fraud']
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

# Create sequences
sequence_length = 10
X_sequences, y_sequences = create_sequences(data, sequence_length)

# Split data into training + validation and hold-out sets
X_temp, X_holdout, y_temp, y_holdout = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # Further split temp into train and validation

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

# Evaluate the model
final_loss, final_accuracy = model.evaluate(X_holdout, y_holdout)
print("Final Loss on Hold-out Set:", final_loss)
print("Final Accuracy on Hold-out Set:", final_accuracy)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.show()
