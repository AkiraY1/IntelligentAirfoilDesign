import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as ipydisplay
import scipy as sp 
import seaborn as sbs
import os

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

# Choose examples for training.
training_examples = preprocess_features(dataframe.head(81156))
training_targets_thickness = dataframe.loc[:81155, ["thickness"]]
training_targets_thickness_pos= dataframe.loc[:81155, ["thickness_pos"]]
training_targets_camber= dataframe.loc[:81155, ["camber"]]
training_targets_camber_pos= dataframe.loc[:81155, ["camber_pos"]]
training_targets_alpha= dataframe.loc[:81155, ["Alpha"]]

# Choose examples for validation.
validation_examples = preprocess_features(dataframe.tail(20290))
validation_targets_thickness = dataframe.loc[81155:, ["thickness"]]
validation_targets_thickness_pos = dataframe.loc[81155:, ["thickness_pos"]]
validation_targets_camber = dataframe.loc[81155:, ["camber"]]
validation_targets_camber_pos = dataframe.loc[81155:, ["camber_pos"]]
validation_targets_alpha = dataframe.loc[81155:, ["Alpha"]]

# Double-check 
print("Training examples summary:")
ipydisplay.display(training_examples.describe())
print("Validation examples summary:")
ipydisplay.display(validation_examples.describe())

print("Training targets summary:")
ipydisplay.display(training_targets_thickness.describe())
ipydisplay.display(training_targets_thickness_pos.describe())
ipydisplay.display(training_targets_camber .describe())
ipydisplay.display(training_targets_camber_pos.describe())
ipydisplay.display(training_targets_alpha .describe())
print("Validation targets summary:")
ipydisplay.display(validation_targets_thickness.describe())
ipydisplay.display(validation_targets_thickness_pos.describe())
ipydisplay.display(validation_targets_camber .describe())
ipydisplay.display(validation_targets_camber_pos.describe())
ipydisplay.display(validation_targets_alpha .describe())

#turn into array
training_examples_np = np.asarray(training_examples).astype('float32')
training_targets_thickness_np = np.asarray(training_targets_thickness).astype('float32')
training_targets_thickness_pos_np = np.asarray(training_targets_thickness_pos).astype('float32')
training_targets_camber_np = np.asarray(training_targets_camber).astype('float32')
training_targets_camber_pos_np = np.asarray(training_targets_camber_pos).astype('float32')
training_targets_alpha_np = np.asarray(training_targets_alpha).astype('float32')

validation_examples_np = np.asarray(validation_examples).astype('float32')
validation_targets_thickness_np = np.asarray(training_targets_thickness).astype('float32')
validation_targets_thickness_pos_np = np.asarray(training_targets_thickness_pos).astype('float32')
validation_targets_camber_np = np.asarray(training_targets_camber).astype('float32')
validation_targets_camber_pos_np = np.asarray(training_targets_camber_pos).astype('float32')
validation_targets_alpha_np = np.asarray(training_targets_alpha).astype('float32')

print(validation_targets_thickness_np)
print(validation_targets_thickness_pos_np)
print(validation_targets_camber_np)
print(validation_targets_camber_pos_np)
print(validation_targets_alpha_np)
print(validation_examples_np)
print(validation_targets_thickness_np.shape)
print(validation_targets_thickness_pos_np.shape)
print(validation_targets_camber_np.shape)
print(validation_targets_camber_pos_np.shape)
print(validation_targets_alpha_np.shape)

print(training_targets_thickness_np)
print(training_targets_thickness_pos_np)
print(training_targets_camber_np)
print(training_targets_camber_pos_np)
print(training_targets_alpha_np)
print(training_examples_np)
print(training_examples.shape)
print(training_targets_thickness_np.shape)
print(training_targets_thickness_pos_np.shape)
print(training_targets_camber_np.shape)
print(training_targets_camber_pos_np.shape)
print(training_targets_alpha_np.shape)

import sklearn
from sklearn import ensemble
from sklearn.metrics import accuracy_score

###################################### THICKNESS ######################################
regressor_thickness = sklearn.ensemble.RandomForestRegressor(n_estimators = 35,
                                                      max_depth=None, 
                                                      min_samples_split=2, 
                                                      min_samples_leaf=10, 
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto',
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0, 
                                                      bootstrap=True, 
                                                      oob_score=False,
                                                      n_jobs=None,
                                                      random_state=None,
                                                      verbose=0,
                                                      warm_start=False,
                                                      ccp_alpha=0.0)

# Train the model using the training sets
regressor_thickness.fit(training_examples_np, training_targets_thickness_np)
#regressor.fit(training_examples_np, training_targets_cd_np)

thickness_pred_np = regressor_thickness.predict(validation_examples_np)
thickness_pred = pd.DataFrame({'thickness_pred': thickness_pred_np})
validation_targets_thickness = validation_targets_thickness.assign(index=range(20290))
validation_targets_thickness = validation_targets_thickness.set_index('index')
thickness = thickness_pred.join(validation_targets_thickness)

res_thickness = thickness["thickness_pred"]
res_thickness_base = thickness["thickness"]
residual_thickness = 0
for i in range(20290):
  x = abs((res_thickness_base[i] - res_thickness[i]))
  residual_thickness += x
print(residual_thickness)
print(residual_thickness/20290)

print(sklearn.metrics.mean_squared_error(thickness['thickness_pred'], thickness['thickness']))

###################################### THICKNESS POSITION ######################################
regressor_thickness_pos = sklearn.ensemble.RandomForestRegressor(n_estimators = 35,
                                                      max_depth=None, 
                                                      min_samples_split=2, 
                                                      min_samples_leaf=10, 
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto',
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0, 
                                                      bootstrap=True, 
                                                      oob_score=False,
                                                      n_jobs=None,
                                                      random_state=None,
                                                      verbose=0,
                                                      warm_start=False,
                                                      ccp_alpha=0.0)

regressor_thickness_pos.fit(training_examples_np, training_targets_thickness_pos_np)
#regressor.fit(training_examples_np, training_targets_cd_np)
thickness_pos_pred_np = regressor_thickness_pos.predict(validation_examples_np)
thickness_pos_pred = pd.DataFrame({'thickness_pos_pred': thickness_pos_pred_np})
validation_targets_thickness_pos = validation_targets_thickness_pos.assign(index=range(20290))
validation_targets_thickness_pos = validation_targets_thickness_pos.set_index('index')
thickness_pos = thickness_pos_pred.join(validation_targets_thickness_pos)

res_thickness_pos = thickness_pos["thickness_pos_pred"]
res_thickness_pos_base = thickness_pos["thickness_pos"]
residual_thickness_pos = 0
for i in range(20290):
  x = abs((res_thickness_pos_base[i] - res_thickness_pos[i]))
  residual_thickness_pos += x
print(residual_thickness_pos)
print(residual_thickness_pos/20290)

print(sklearn.metrics.mean_squared_error(thickness_pos['thickness_pos_pred'], thickness_pos['thickness_pos']))

###################################### CAMBER ######################################
regressor_camber = sklearn.ensemble.RandomForestRegressor(n_estimators = 35,
                                                      max_depth=None, 
                                                      min_samples_split=2, 
                                                      min_samples_leaf=10, 
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto',
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0, 
                                                      bootstrap=True, 
                                                      oob_score=False,
                                                      n_jobs=None,
                                                      random_state=None,
                                                      verbose=0,
                                                      warm_start=False,
                                                      ccp_alpha=0.0)

regressor_camber.fit(training_examples_np, training_targets_camber_np)
#regressor.fit(training_examples_np, training_targets_cd_np)
camber_pred_np = regressor_camber.predict(validation_examples_np)
camber_pred = pd.DataFrame({'camber_pred': camber_pred_np})
validation_targets_camber = validation_targets_camber.assign(index=range(20290))
validation_targets_camber = validation_targets_camber.set_index('index')
camber = camber_pred.join(validation_targets_camber)

res_camber = camber["camber_pred"]
res_camber_base = camber["camber"]
residual_camber = 0
for i in range(20290):
  x = abs((res_camber_base[i] - res_camber[i]))
  residual_camber += x
print(residual_camber)
print(residual_camber/20290)

print(sklearn.metrics.mean_squared_error(camber['camber_pred'], camber['camber']))

###################################### CAMBER POSITION ######################################
regressor_camber_pos = sklearn.ensemble.RandomForestRegressor(n_estimators = 35,
                                                      max_depth=None, 
                                                      min_samples_split=2, 
                                                      min_samples_leaf=10, 
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto',
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0, 
                                                      bootstrap=True, 
                                                      oob_score=False,
                                                      n_jobs=None,
                                                      random_state=None,
                                                      verbose=0,
                                                      warm_start=False,
                                                      ccp_alpha=0.0)

regressor_camber_pos.fit(training_examples_np, training_targets_camber_pos_np)
#regressor.fit(training_examples_np, training_targets_cd_np)
camber_pos_pred_np = regressor_camber_pos.predict(validation_examples_np)
camber_pos_pred = pd.DataFrame({'camber_pos_pred': camber_pos_pred_np})
validation_targets_camber_pos = validation_targets_camber_pos.assign(index=range(20290))
validation_targets_camber_pos = validation_targets_camber_pos.set_index('index')
camber_pos = camber_pos_pred.join(validation_targets_camber_pos)

res_camber_pos = camber_pos["camber_pos_pred"]
res_camber_pos_base = camber_pos["camber_pos"]
residual_camber_pos = 0
for i in range(20290):
  x = abs((res_camber_pos_base[i] - res_camber_pos[i]))
  residual_camber_pos += x
print(residual_camber_pos)
print(residual_camber_pos/20290)

print(sklearn.metrics.mean_squared_error(camber_pos['camber_pos_pred'], camber_pos['camber_pos']))

###################################### ALPHA ######################################
regressor_alpha = sklearn.ensemble.RandomForestRegressor(n_estimators = 35,
                                                      max_depth=None, 
                                                      min_samples_split=2, 
                                                      min_samples_leaf=10, 
                                                      min_weight_fraction_leaf=0.0,
                                                      max_features='auto',
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0, 
                                                      bootstrap=True, 
                                                      oob_score=False,
                                                      n_jobs=None,
                                                      random_state=None,
                                                      verbose=0,
                                                      warm_start=False,
                                                      ccp_alpha=0.0)

regressor_alpha.fit(training_examples_np, training_targets_alpha_np)
#regressor.fit(training_examples_np, training_targets_cd_np)
alpha_pred_np = regressor_alpha.predict(validation_examples_np)
alpha_pred = pd.DataFrame({'alpha_pred': alpha_pred_np})
validation_targets_alpha = validation_targets_alpha.assign(index=range(20290))
validation_targets_alpha = validation_targets_alpha.set_index('index')
alpha = alpha_pred.join(validation_targets_alpha)

res_alpha = alpha["alpha_pred"]
res_alpha_base = alpha["Alpha"]
residual_alpha = 0
for i in range(20290):
  x = abs((res_alpha_base[i] - res_alpha[i]))
  residual_alpha += x
print(residual_alpha)
print(residual_alpha/20290)

print(sklearn.metrics.mean_squared_error(alpha['alpha_pred'], alpha['Alpha']))

validation_np = np.array([[1, 1, 1],
                         [2, 2, 2],
                         [-0.0695, 0.04327, 0.0021], #spline_foil1, alpha=0
                         [0.4532, 0.14578, -0.0097], #spline_foil1, alpha=15
                         [0.8141, 0.40411, -0.1259], #spline_foil1, alpha=29
                         [-0.185, 0.07311, -0.0097], #spline_foil2, alpha=0
                         [0.3746, 0.17089, -0.006], #spline_foil2, alpha=15
                         [0.6467, 0.29717, -0.0708]]) #spline_foil2, alpha=28

actual_validation_np = np.array([[0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0.1165, 0.2382, 0.0145, 0.1892, 0],
                                  [0.1165, 0.2382, 0.0145, 0.1892, 15],
                                  [0.1165, 0.2382, 0.0145, 0.1892, 29], 
                                  [0.1886, 0.3453, 0.0504, 0.4615, 0], 
                                  [0.1886, 0.3453, 0.0504, 0.4615, 15],
                                  [0.1886, 0.3453, 0.0504, 0.4615, 28]])

actual_validation_df = pd.DataFrame(actual_validation_np, columns = ['thickness_act', 'thickness_pos_act', 'camber_act', 'camber_pos_act', 'alpha_act'])

#validation
validation_alpha_pred_np = regressor_alpha.predict(validation_np)
alpha_pred = pd.DataFrame({'alpha_pred': validation_alpha_pred_np})
validation_thickness_pred_np = regressor_thickness.predict(validation_np)
thickness_pred = pd.DataFrame({'thickness_pred': validation_thickness_pred_np})
validation_thickness_pos_pred_np = regressor_thickness_pos.predict(validation_np)
thickness_pos_pred = pd.DataFrame({'thickness_pos_pred': validation_thickness_pos_pred_np})
validation_camber_pred_np = regressor_camber.predict(validation_np)
camber_pred = pd.DataFrame({'camber_pred': validation_camber_pred_np})
validation_camber_pos_pred_np = regressor_camber_pos.predict(validation_np)
camber_pos_pred = pd.DataFrame({'camber_pos_pred': validation_camber_pos_pred_np})

df = thickness_pred.join(thickness_pos_pred)
df =df.join(camber_pred)
df =df.join(camber_pos_pred)
df =df.join(alpha_pred)
df =df.join(actual_validation_df)

print(df)
df.to_csv("RF_validation.csv")