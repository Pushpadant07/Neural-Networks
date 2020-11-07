import pandas as pd
import numpy as np
from sklearn import preprocessing

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

Startups = pd.read_csv("D:\\ExcelR Data\\Assignments\\Neural Networks\\50_Startups.csv")
Startups.columns
startups=Startups

Le = preprocessing.LabelEncoder() ##Label encoder() using for levels of categorical features into numerical values
startups['State'] = Le.fit_transform(Startups['State'])

# Splitting the data input and output
predictors=Startups.iloc[:,0:4]
target =Startups.iloc[:,4]

# Building the Model
stp_model = Sequential()
stp_model.add(Dense(50,input_dim=4,activation="relu"))
stp_model.add(Dense(40,activation="relu"))
stp_model.add(Dense(20,activation="relu"))
stp_model.add(Dense(1,kernel_initializer="normal"))
stp_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])

#first_model
stp_model.fit(predictors,target,epochs=10)
pred = stp_model.predict(predictors)
Startups["pred"]=stp_model.predict(predictors)
pred=pd.Series(i[0] for i in pred)

rmse_value=np.sqrt(np.mean(pred-target)**2)
import matplotlib.pyplot as plt
plt.plot(pred,target,"bo")

np.corrcoef(pred,target)  
# array([[1.        , 0.87806135],
#       [0.87806135, 1.        ]])
