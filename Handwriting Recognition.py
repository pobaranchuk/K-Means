import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import datasets

digits = datasets.load_digits()
#print(digits.DESCR)
#print(digits.data)
#print(digits.target)

#To visualize the data images, we need to use Matplotlib. Let’s visualize the image at index 1
#print(len(digits))
#plt.gray()  
#plt.matshow(digits.images[121]) 
#plt.show()

#Because there are 10 digits (0, 1, 2, 3, 4, 5, 6, 7, 8, and 9), there should be 10 clusters./
#So k, the number of clusters, is 10
model = KMeans(n_clusters = 10, random_state= 42)

model.fit(digits.data)
#Let’s visualize all the centroids!/
#Because data samples live in a 64-dimensional space, the centroids have values so they can be images!
fig = plt.figure(figsize=(8,3))
fig.suptitle(t="Visualizing after K-Means", fontsize = 14, fontweight = "bold")

#loop to displays each of the cluster_centers_
for i in range(10):
 
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
 
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
  
  
plt.show()