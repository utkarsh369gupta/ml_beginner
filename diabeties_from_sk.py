import numpy as np 
from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error
import pandas as pd 
import matplotlib.pyplot as plt

x=[1,2,3]
y=[3,2,4]
x1=np.array(x)
y1=np.array(y)
x1=x1.reshape(-1,1)
y1=y1.reshape(-1,1)

model=linear_model.LinearRegression()
model.fit(x1,y1)
print(model.coef_)
print(model.intercept_)

'''
diabetes=datasets.load_diabetes()

# print(diabetes.keys())
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']

# print(diabetes.DESCR)
# [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]

# print(len(diabetes.data))
# print(len(diabetes.target))

# print(type(diabetes))

# dia_x=diabetes.data[:,np.newaxis,2]
# dia_y=diabetes.target[:,np.newaxis]
dia_x=diabetes.data
dia_y=diabetes.target
# print(dia_x)

dia_x_train=dia_x[:-30]
dia_x_test=dia_x[-20:]
dia_y_train=dia_y[:-30]
dia_y_test=dia_y[-20:]

model=linear_model.LinearRegression()
model.fit(dia_x_train,dia_y_train)
print(model.score(dia_x_test, dia_y_test))

y_predict=model.predict(dia_x_test)
print("the mean square error is: ", mean_squared_error(dia_y_test,y_predict))

print(model.coef_)
print(model.intercept_)

# plt.scatter(dia_x_test,dia_y_test)
# plt.plot(dia_x_test,y_predict)
# plt.show()


# 0.4698332434180642
# the mean square error is:  2561.320427728385
# [[941.43097333]]
# [153.39713623]
'''












