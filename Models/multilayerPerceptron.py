import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as ipydisplay
import scipy as sp 
import seaborn as sbs
import os
import sklearn

dataframe = pd.read_csv("data_v2.csv")
dataframe = dataframe.sample(frac=1).reset_index()
dataframe = dataframe.drop(['index'], axis=1)
#ipydisplay.display(dataframe)

def preprocess_features(dataframe):
  #feature selection
  selected_features = dataframe[["CL", "CD", "CM"]]
  processed_features = selected_features.copy()
  #if needed, create a synthetic feature below
  return processed_features

# Choose examples for training.
training_examples = preprocess_features(dataframe.head(81156))
training_targets = dataframe.loc[:81155, ["thickness", "thickness_pos", "camber", "camber_pos", "Alpha"]]
# Choose examples for validation.
validation_examples = preprocess_features(dataframe.tail(20290))
validation_targets = dataframe.loc[81155:, ["thickness", "thickness_pos", "camber", "camber_pos", "Alpha"]]

# Double-check 
print("Training examples summary:")
ipydisplay.display(training_examples.describe())
print("Validation examples summary:")
ipydisplay.display(validation_examples.describe())

print("Training targets summary:")
ipydisplay.display(training_targets.describe())
print("Validation targets summary:")
ipydisplay.display(validation_targets.describe())

#turn into array
training_examples_np = np.asarray(training_examples).astype('float32')
training_targets_np = np.asarray(training_targets).astype('float32')
validation_examples_np = np.asarray(validation_examples).astype('float32')
validation_targets_np = np.asarray(validation_targets)#.astype('float32')

print(validation_targets_np)
print(validation_examples_np)
print(training_examples.shape)
print(training_targets.shape)
print(validation_examples.shape)
print(validation_targets.shape)

#multilayer perceptron
import os
import re
import shutil
import string
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import losses
from keras import metrics
from tensorflow import keras
from keras import optimizers 

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(3),
  tf.keras.layers.Dense(30, activation = "softmax"),
  keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(30, activation = "softmax"),
  keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(30, activation = "softmax"),
  keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(30, activation = "softmax"),
  keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(30, activation = "softmax"),
  keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(30, activation = "softmax"),
  tf.keras.layers.Dense(5)
])

model.compile(
    optimizer= tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,amsgrad=False, name="Adam"),
    loss=tf.keras.losses.MeanSquaredError(),
)

train_dataset = tf.data.Dataset.from_tensor_slices((training_examples_np, training_targets_np))
test_dataset = tf.data.Dataset.from_tensor_slices((validation_examples_np, validation_targets_np))

history = model.fit(
    training_examples_np, training_targets_np,
    epochs=100,
    batch_size = 32,
    validation_data = (validation_examples_np, validation_targets_np)
)

#loss and accuracy evaluation
loss = model.evaluate(validation_examples_np, validation_targets_np)
print("Loss: ", loss)

history_dict = history.history
history_dict.keys()

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

#loss plots
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


b = model.predict(validation_examples_np)
a = pd.DataFrame(b, columns = ["thickness", "thickness_pos", "camber", "camber_pos", "Alpha"])