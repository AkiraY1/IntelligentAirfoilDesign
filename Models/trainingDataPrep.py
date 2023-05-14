import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as ipydisplay
import scipy as sp 
import seaborn as sbs

dataframe = pd.read_csv("data_v2.csv")
dataframe = dataframe.sample(frac=1).reset_index()
dataframe = dataframe.drop(['index'], axis=1)
ipydisplay.display(dataframe)

def preprocess_features(dataframe):
  
  #feature selection
  selected_features = dataframe[["CL", "CD", "CM"]]
  processed_features = selected_features.copy()
  #if needed, create a synthetic feature below
  return processed_features

"""def preprocess_targets(dataframe):
   output_targets = pd.DataFrame()
  # Scale the target if needed, replace training_target's output
   return output_targets"""

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

corrMatrix = dataframe.corr()
ipydisplay.display(corrMatrix)
corrMatrix.to_csv("correalation_matrix.csv")

import seaborn as sns; sns.set_theme()

plt.figure(figsize=(8, 6))
corrMatrix_heatmap = sns.heatmap(corrMatrix)

