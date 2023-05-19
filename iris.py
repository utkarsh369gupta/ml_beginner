from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 

iris=datasets.load_iris()
# print(iris.DESCR)

features=iris.data
label=iris.target

# print(features[0],label[0])

model=KNeighborsClassifier()
model.fit(features,label)
print(model.predict([[1,1,1,1]]))