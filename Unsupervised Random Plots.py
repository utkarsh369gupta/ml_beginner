import matplotlib.pyplot as plt
import numpy as np

np.random.seed(25)
X = np.random.rand(5, 10) * 100
print(X.shape)
print(X)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=123)
kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()



