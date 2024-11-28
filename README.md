# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M.K.Suriya prakash
RegisterNumber:24901016
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Suriya\Downloads\Employee.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

print(data["left"].value_counts())


le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)


dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")


new_data = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
print(f"Prediction for new data: {dt.predict(new_data)}")


plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=x.columns, class_names=["Stayed", "Left"], filled=True)
plt.show()
*/
```

## Output:
![Screenshot 2024-11-28 204145](https://github.com/user-attachments/assets/8cef7aff-ffc2-4b61-8491-325cb5852ea7)
![Screenshot 2024-11-28 204206](https://github.com/user-attachments/assets/24cf37cb-f30b-4f06-9bd7-b6258d38f67b)
![Screenshot 2024-11-28 204221](https://github.com/user-attachments/assets/1f15f05a-6256-4313-ab6f-d289860788bd)







## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
