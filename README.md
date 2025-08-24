# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Developed by: M.Mounika
RegisterNumber: 212224040202


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## DATASET:
<img width="331" height="735" alt="Screenshot 2025-08-24 221704" src="https://github.com/user-attachments/assets/3cb0a039-2d33-4fb7-aa8e-6b271ceb77f2" />

## HEAD VALUES:
<img width="261" height="202" alt="image" src="https://github.com/user-attachments/assets/02d898b0-6c35-4c27-8866-607ce04f5600" />

## TAIL VALUES:
<img width="271" height="197" alt="image" src="https://github.com/user-attachments/assets/fa1082ac-2f98-4035-a8b6-fb14fe79c52f" />

## X AND Y VALUES:
<img width="810" height="690" alt="image" src="https://github.com/user-attachments/assets/b3a2c005-a488-4518-a3be-5cf5f741de2e" />

## PREDICATION VALUES OF X AND Y:
<img width="813" height="117" alt="image" src="https://github.com/user-attachments/assets/2373cb67-f0de-48d0-90e2-2bd29cd0dcc8" />

## TRAINING SET:
<img width="793" height="605" alt="image" src="https://github.com/user-attachments/assets/1669730a-9da1-4f54-af55-fa0f7b99c467" />

## TESTING SET AND MSE,MAE and RMSE:
<img width="738" height="587" alt="image" src="https://github.com/user-attachments/assets/e4c397eb-baac-4c32-9405-fef7083156f8" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
