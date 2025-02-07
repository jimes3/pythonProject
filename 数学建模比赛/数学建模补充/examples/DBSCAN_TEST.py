from Cluster import DBSCAN as dbscan
from sklearn.cluster import DBSCAN
import time
import matplotlib.pyplot as plt
from sklearn import datasets

X1, y1=datasets.make_circles(n_samples=5000, factor=.6, noise=.05)
trainData = X1[0:1000]

time_start2 = time.time()
clf2 = dbscan()
pred = clf2.train(trainData)
time_end2 = time.time()
print("Runtime of DBSCAN:", time_end2-time_start2)

time_start3 = time.time()
clf3 = DBSCAN(eps=0.1, min_samples=10)
clf3.fit(trainData)
pred3 = clf3.labels_
time_end3 = time.time()
plt.scatter(trainData[:, 0], trainData[:, 1], c=pred3)
plt.title('Sklearn DBSCAN')
plt.show()
print("Runtime of Sklearn DBSCAN:", time_end3-time_start3)




