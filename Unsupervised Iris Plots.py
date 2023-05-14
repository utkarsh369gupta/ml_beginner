from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load the iris dataset
iris = load_iris()
X = iris.data

# create a KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3)

# fit the model to the data
kmeans.fit(X)

# predict the cluster labels for each data point
y_pred = kmeans.predict(X)

# plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()


