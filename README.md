# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: M.MOHAMED ASHFAQ
RegisterNumber:  212224240090
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

y_pred

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred)
mse

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test, y_pred)
mae

from sklearn.metrics import r2_score
r1=r2_score(y_test,y_pred)
r1

dt.predict([[5,6]])

```

## Output:
head:
<img width="1134" height="388" alt="9ml1" src="https://github.com/user-attachments/assets/156423e2-f3a4-490d-8a8e-891cc4438869" />


info:
<img width="768" height="394" alt="9ml2" src="https://github.com/user-attachments/assets/17a9a387-7b92-40d8-832d-d426c98219e7" />


isnull:
<img width="896" height="219" alt="9ml3" src="https://github.com/user-attachments/assets/71debec7-a1ff-4f7a-95fc-cf60e58df0bb" />


head:
<img width="590" height="404" alt="9ml4" src="https://github.com/user-attachments/assets/1c3750cb-9cee-49d0-b297-4140b221a7b1" />


x.head:
<img width="316" height="420" alt="9ml5" src="https://github.com/user-attachments/assets/731191d1-cce9-465e-850b-7ae73e977d0e" />

y.head:
<img width="222" height="426" alt="9ml6" src="https://github.com/user-attachments/assets/f3beab00-5877-45dc-ba67-9fde9779b6f4" />


y_pred:
<img width="608" height="122" alt="9ml7" src="https://github.com/user-attachments/assets/9c17765c-d32c-45e0-a5ef-6aaa980389e1" />


mse:
<img width="326" height="104" alt="9ml8" src="https://github.com/user-attachments/assets/0f013713-398d-4f6e-8956-d1d3a30ef386" />


mae:
<img width="616" height="172" alt="9ML9" src="https://github.com/user-attachments/assets/cedc87f7-c861-4b11-9d9c-8703d813e4cb" />


r2:
<img width="602" height="86" alt="9ML10" src="https://github.com/user-attachments/assets/fb143dd4-7405-4f98-826d-71c699a4cd13" />


predict:
<img width="1612" height="146" alt="9ml11" src="https://github.com/user-attachments/assets/91fe739a-997b-40f5-b2b3-810e7219948f" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
