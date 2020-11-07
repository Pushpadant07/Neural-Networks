import pandas as pd
import numpy as np
from sklearn import preprocessing

# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

Forestfires = pd.read_csv("D:\\ExcelR Data\\Assignments\\Neural Networks\\forestfires.csv")
Forestfires.columns
forestfires = Forestfires

Le = preprocessing.LabelEncoder()
#convertong catogorical to numerical
Forestfires['month']=Le.fit_transform(Forestfires['month'])
Forestfires['day']=Le.fit_transform(Forestfires['day'])
Forestfires['size_category']=Le.fit_transform(Forestfires['size_category'])

predictors = Forestfires.drop(["area"],axis=1)
target =Forestfires.iloc[:,10]


Ffires_model = Sequential()
Ffires_model.add(Dense(50,input_dim=30,activation="relu"))
Ffires_model.add(Dense(40,activation="relu"))
Ffires_model.add(Dense(20,activation="relu"))
Ffires_model.add(Dense(1,kernel_initializer="normal"))
Ffires_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])


#first_model
Ffires_model.fit(predictors,target,epochs=10)
pred = Ffires_model.predict(predictors)
Forestfires["pred"]=Ffires_model.predict(predictors)
pred=pd.Series([i[0] for i in pred])
rmse_value = np.sqrt(np.mean(pred-target)**2)



import matplotlib.pyplot as plt
plt.plot(pred,target,"bo")
np.corrcoef(pred,target) 
#array([[1.        , 0.08544648],
#       [0.08544648, 1.        ]])
