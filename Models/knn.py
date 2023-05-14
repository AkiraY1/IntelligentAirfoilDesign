import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('snappy.csv', header=None, sep='\t')
num = int(len(df)*0.8)

df1 = df.sample(frac = 0.8)
df2 = df.drop(df1.index)

x = df1.values[:, -3:].astype('float32')
y = df1.values[:, :-3].astype('float32')

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 113, weights="uniform")
knn.fit(x, y)

filename = 'knn.sav'
pickle.dump(knn, open(filename, 'wb'))

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import pickle

df = pd.read_csv('snappy.csv', header=None, sep='\t')
num = int(len(df)*0.9)
df1 = df.sample(frac = 0.9)
df2 = df.drop(df1.index)

x = df1.values[:, -3:].astype('float32')
y = df1.values[:, :-3].astype('float32')

bagged_knn = KNeighborsRegressor(n_neighbors=113, weights="uniform")
bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)
bagging_model.fit(x, y)

filename = 'knn_bagging.sav'
pickle.dump(bagging_model, open(filename, 'wb'))

#Grid search to optimize K
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('snappy.csv', header=None, sep='\t')
num = int(len(df)*0.8)
df1 = df.sample(frac = 0.8)
df2 = df.drop(df1.index)

x = df1.values[:, -3:].astype('float32')
y = df1.values[:, :-3].astype('float32')

parameters = {"n_neighbors": range(1, 200), "weights": ["uniform", "distance"]}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(x, y)

print(gridsearch.best_params_)

#Calculation of residues of parameters
test_cases = df2.values.tolist()
residue_thickness, residue_thick_pos, residue_camber, residue_cam_pos, residue_alpha = 0, 0, 0, 0, 0
i = 0

for case in test_cases:
    i += 1
    predicted_values = knn.predict([case[-3:]])
    actual_values = case[0:5]

    residue_thickness += abs(predicted_values[0][0] - actual_values[0])
    residue_thick_pos += abs(predicted_values[0][1] - actual_values[1])
    residue_camber += abs(predicted_values[0][2] - actual_values[2])
    residue_cam_pos += abs(predicted_values[0][3] - actual_values[3])
    residue_alpha += abs(predicted_values[0][4] - actual_values[4])

print(f"finalR: {residue_thickness}")
print(f"finalR: {residue_thick_pos}")
print(f"finalR: {residue_camber}")
print(f"finalR: {residue_cam_pos}")
print(f"finalR: {residue_alpha}")

#Evaluation of RSE vs. values of K
import math
import matplotlib.pyplot as plt

n = len(y)
index = []
error = []
for i in range(1, 41):
    knn = KNeighborsRegressor(n_neighbors = i)
    knn.fit(x, y)
    y_pre = knn.predict(x)
    rss = sum((y_pre - y)**2)
    print(rss)
    rse = math.sqrt(sum(rss)/(n-2))
    index.append(i)
    error.append(rse)
print(index, error)

plt.plot(index, error)
plt.xlabel('Values of K')
plt.ylabel('RSE Error')
plt.title('Graph: K VS Error')
plt.show()