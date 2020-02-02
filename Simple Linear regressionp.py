#Simple Linear Regression
 #import libarires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
  

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Splitting dataset to training dataset and test dataset
from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size =1/3 ,random_state=0)
#Simple Linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
 #Predicting the test set reults
y_pred = regressor.predict(x_test)
 #visiualising the training set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Trainingset)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
 #Visualising test set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Testset)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()