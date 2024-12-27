# Import libraries. You may or may not use all of these.
!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()

# Add your code here:

# Convert categorical data to numeric
dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
dataset['smoker'] = dataset['smoker'].map({'yes': 1, 'no': 0})
dataset['region'] = dataset['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

# Split dataset into train and test sets (80/20 split)
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Pop off the 'expenses' column to create labels
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# Normalize the data (feature scaling)
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def normalize(x):
    return (x - train_stats['mean']) / train_stats['std']

norm_train_dataset = normalize(train_dataset)
norm_test_dataset = normalize(test_dataset)

# Build the model
from tensorflow.keras import regularizers

# Build the improved model
def build_improved_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())],
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),

        layers.Dense(1)  # Single output for regression
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model

model = build_improved_model()

# Early stopping and learning rate scheduler
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train the model
history = model.fit(
    norm_train_dataset,
    train_labels,
    epochs=500,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    shuffle=True,
    callbacks=[early_stop, reduce_lr, tfdocs.modeling.EpochDots()]
)

# Re-check normalization during evaluation
norm_test_dataset = normalize(test_dataset)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)