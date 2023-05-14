import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df=pd.read_csv("LinearREG_with_single_variable/area_pri_train.csv")
df1=pd.read_csv("LinearREG_with_single_variable/area_pri_test.csv")

plt.scatter(df.area,df.price, color='r', marker="*")
plt.show()

'''
slope,intercept=np.polyfit(df.area,df.price,1)
reg_line=np.poly1d([slope,intercept])
plt.plot(np.unique(df.area),reg_line(np.unique(df.area)))
plt.xlabel("area")
plt.ylabel("price")
for i in range(13):
    print("the value of area ",df1.area[i],"is",reg_line(df1.area[i]))
p=reg_line(df1[['area']])
df1['price']=p
df1.to_csv("area_pri_test.csv",index=False)
print((df1))
plt.scatter(df1.area,df1.price,color='b',marker="+")
plt.show()
'''

# or

reg = LinearRegression()
reg.fit(df[['area']],df.price)
# print(reg.coef_)
# print(reg.intercept_)
p=reg.predict(df1)
d=pd.DataFrame()
df1['prices']=p
# d['prices']=p
# d.to_csv["df1.csv"]
print(df1)

'''
area
1000
1500
2300
3540
4120
4560
5490
3460
4750
2300
9000
8600
7100
'''