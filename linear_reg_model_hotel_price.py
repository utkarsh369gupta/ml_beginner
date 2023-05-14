import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import linear_model
import math
import word2number

df=pd.read_csv("hotel_pri.csv")
# print(df)
# print(df.bedrooms.median())
median_bed= math.floor(df.bedrooms.median())
print(median_bed)
df.bedrooms=df.bedrooms.fillna(median_bed)
print(df)
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[3000,3,40]]))
print(reg.predict([[2500,4,5]]))






