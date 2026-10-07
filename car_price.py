from sklearn import preprocessing   
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

# read a Data
df = pd.read_csv("car_prices_1000.csv")
df = df.fillna("nan")

# set X and Y data
X = df.drop(columns=["price", "id"])
y = df["price"]
X = pd.get_dummies(X, drop_first=True)

# train and test split
train_x, test_x, y_train, test_y = train_test_split(X, y)

# create and fit model
model = LinearRegression()
model.fit(train_x, y_train)
yhat = model.predict(test_x)

# show score with R2Score
print("r2score =", r2_score(test_y, yhat))

# Enter new x Data
year = int(input("Year: "))
mileage = float(input("Mileage: "))
engine_size = float(input("Engine size: "))
horsepower = int(input("Horsepower: "))
doors = int(input("Doors: "))
brand = input("Brand: ")
fuel_type = input("Fuel type: ")
transmission = input("Transmission: ")
accident_history = input("Accident history: ")
condition = input("Condition: ")

# create new x data
new_car = pd.DataFrame([{
    "year": year,
    "mileage": mileage,
    "engine_size": engine_size,
    "horsepower": horsepower,
    "doors": doors,
    "brand": brand,
    "fuel_type": fuel_type,
    "transmission": transmission,
    "accident_history": accident_history,
    "condition": condition
}])

# set new x data
new_car = pd.get_dummies(new_car)
new_car = new_car.reindex(columns=train_x.columns, fill_value=0)

# predict a new x and print a model answer
prediction = model.predict(new_car)
print("model answer is :", prediction)
