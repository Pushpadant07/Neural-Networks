import pandas as pd
import numpy as np

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

Concrete = pd.read_csv("D:\\ExcelR Data\\Assignments\\Neural Networks\\concrete.csv")
Concrete.columns

predictors = Concrete.iloc[:,0:8]
target = Concrete.iloc[:,8]

cont_model = Sequential()
cont_model.add(Dense(50,input_dim=8,activation="relu"))
cont_model.add(Dense(40,activation="relu"))
cont_model.add(Dense(20,activation="relu"))
cont_model.add(Dense(1,kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])

#first_model
cont_model.fit(predictors,target,epochs=10)
pred = cont_model.predict(predictors)
pred = pd.Series([i[0] for i in pred])
rmse_value = np.sqrt(np.mean(pred-target)**2)

import matplotlib.pyplot as plt
plt.plot(pred,target,"bo")
np.corrcoef(pred,target) # we got high correlation 

#      array([[1.        , 0.85577629],
 #           [0.85577629, 1.        ]])
 
 # we got high correlation 