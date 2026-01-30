# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.

2.Scale the features using a standard scaler to normalize the data.

3.Initialize model parameters (theta) and add an intercept term to the feature set.

4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.

5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SUDARSAN.A
RegisterNumber:  212224220111
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X, y, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(X)), X]
    theta = np.zeros((X.shape[1], 1))
    for _ in range(num_iters):
        theta -= learning_rate * (1 / len(y)) * X.T.dot(X.dot(theta) - y)
    return theta

data = pd.read_csv('/content/drive/MyDrive/50_Startups (1).csv', header=None)
print("First 5 rows of the dataset:")
print(data.head())

X = data.iloc[1:, :-2].astype(float).values
y = data.iloc[1:, -1].values.reshape(-1, 1)

scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_scaled, y_scaled = scaler_X.fit_transform(X), scaler_y.fit_transform(y)

theta = linear_regression(X_scaled, y_scaled)

print("Theta values after training:")
print(theta)

new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_data_scaled = scaler_X.transform(new_data)
print("New data (scaled):")
print(new_data_scaled)

prediction_scaled = np.dot(np.append(1, new_data_scaled[0]), theta)
prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

print(f"Predicted value: {prediction[0][0]}")
```

## Output:
<img width="1228" height="423" alt="Screenshot 2026-01-30 112431" src="https://github.com/user-attachments/assets/0a26f62b-a008-450c-8931-3051055a68e2" />
<img width="881" height="414" alt="Screenshot 2026-01-30 112442" src="https://github.com/user-attachments/assets/6d7f9c74-e8d4-4c7f-b07b-523f46a74a44" />
<img width="777" height="399" alt="Screenshot 2026-01-30 112450" src="https://github.com/user-attachments/assets/7053c8a8-fc5e-454e-8c71-60d9f7cda21e" />
<img width="745" height="423" alt="Screenshot 2026-01-30 112458" src="https://github.com/user-attachments/assets/5b3dd738-960d-4c65-b7b5-95f84824d20e" />
<img width="802" height="391" alt="Screenshot 2026-01-30 112506" src="https://github.com/user-attachments/assets/88c2002b-4754-4fe2-8ada-dc9bf559cd9b" />
<img width="617" height="85" alt="Screenshot 2026-01-30 112510" src="https://github.com/user-attachments/assets/ac0c51a4-050a-47ca-9563-b10e503575b6" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
