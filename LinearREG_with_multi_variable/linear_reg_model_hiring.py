import numpy as np
import pandas as pd
from sklearn import linear_model
import word2number as w2n 
# import math

df=pd.read_csv("LinearREG_with_multi_variable/hiring.csv")
# print(df)   
med_test_score=round(df.test_score.median())
df.test_score=df.test_score.fillna(med_test_score)
df.experience=df.experience.fillna(0)
# print(df)   

reg=linear_model.LinearRegression()
x=df.drop('salary',axis=1)
x=x.values
y=df.salary
y=y.values
reg.fit(x,y)
# print(reg.coef_)
# print(reg.intercept_)
print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))
print(reg.score(x, y))

# import pickle
# with open('model_pickel','wb') as f:
#     pickle.dump(li, file)
    
# with open('model_pickel','rb') as f:
#     mp=pickle.load(f)

# mp.predict(2,9,6)





