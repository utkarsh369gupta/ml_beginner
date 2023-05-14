# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Load the data
# data = pd.read_csv('house_prices.csv')
# data = pd.get_dummies(data, columns=['neighborhood'])
# # Split the data into training and testing sets
# X = data.drop('price', axis=1)
# y = data['price']
# print(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

# Make predictions on the test set
# y_pred = model.predict(X_test)

# Evaluate the model's performance
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean squared error: {mse}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the data
data = pd.read_csv('house_prices.csv')
data = pd.get_dummies(data, columns=['neighborhood'])

# Split the data into training and testing sets
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R-squared value:", r2)
print("Mean squared error:", mse)

