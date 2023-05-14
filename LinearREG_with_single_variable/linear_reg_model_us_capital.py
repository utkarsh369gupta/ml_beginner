import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import linear_model

df1=pd.read_csv("LinearREG_with_single_variable/us_capital.csv")
plt.scatter(df1['year'],df1['capita'],color="r",marker='*')
plt.plot(df1['year'],df1['capita'], color="b")
# plt.show()

'''
slope, intercept=np.polyfit(df1.year,df1.capita,1)
reg_line=np.poly1d([slope,intercept])
plt.plot(np.unique(df1.year),reg_line(np.unique(df1.year)), color='r')
plt.xlabel("year")
plt.ylabel("per capita income")
plt.show()
print(reg_line(2023))
'''

x=df1.year
y=df1.capita
# print(x)
x=x.values.reshape(-1,1)
# print(x)
y=y.values.reshape(-1,1)
# model=linear_model.LinearRegression()
# model.fit(x,y)
# print(model.predict([[2023]]))


