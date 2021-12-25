# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Donnees_projet_2021/jain.txt") # import des données

head = str(df.columns[0]).split('\t', 3)
df = df["0.85	17.45	2"].str.split('\t',3, expand = True)

df.loc[-1] = head  # adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index()  # sorting by index

newhead = ["Car 1", "Car2", "Cluster"]
for i in range(3):
    df = df.rename(columns = {i : newhead[i]}) 


data = np.array(df).astype(np.float64()) # Conversion en tableau np de float      

for i in range(len(data)): # affichage des vrais clusters
    if(data[i,2]==1):
        color = 'b'
    elif(data[i,2]==2):
        color = 'g'
    #plt.scatter(data[i,0],data[i,1], c = color)


from sklearn.preprocessing import StandardScaler

X = data
scaler = StandardScaler()
Z = scaler.fit_transform(X) #centrage/reduction des donnes

ari = [0,0,0,0,0]


import scipy.cluster.hierarchy as sch
import sklearn.metrics as sm

t = 18
link = sch.linkage(Z, method='ward', optimal_ordering=True,)
#dn = sch.dendrogram(link, color_threshold=t)
#plt.show()

clusters = sch.fcluster(link, t, criterion='distance' )
coef = sm.silhouette_score(Z, clusters)

for i in range(len(clusters)): # affichage des clusters par CAH
    if (clusters[i] == 1):
        color = 'b'
    elif(clusters[i] == 2):
        color = 'g'
    #plt.scatter(data[i,0], data[i,1], c = color)

clusters_reels = data[:,2]
ari[0] = sm.adjusted_rand_score(clusters, clusters_reels)



import sklearn.cluster as sc
OMP_NUM_THREADS=4

kmeans = sc.KMeans(n_clusters=2, init='k-means++', n_init=10).fit(X)
labels = kmeans.labels_
initia = kmeans.inertia_

for i in range(len(clusters)): # affichage des clusters par Kmeans
    if (labels[i] == 0):
        color = 'b'
    elif(labels[i] == 1):
        color = 'g'
    #plt.scatter(data[i,0], data[i,1], c = color)

ari[1] = sm.adjusted_rand_score(labels, clusters_reels)



from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components = 2, covariance_type= 'full', n_init=10).fit(Z)
gm_clusters = gm.predict(Z)

for i in range(len(clusters)): # affichage des clusters par Gaussian mixture
    if (gm_clusters[i] == 1):
        color = 'b'
    elif(gm_clusters[i] == 2):
        color = 'g'
    #plt.scatter(data[i,0], data[i,1], c = color)
    
ari[2] = sm.adjusted_rand_score(gm_clusters, clusters_reels)



dbscan = sc.DBSCAN(eps = 0.5, min_samples = 2,algorithm='auto').fit(Z)
db_clusters = dbscan.labels_

for i in range(len(clusters)): # affichage des clusters par DBSCAN
    if (db_clusters[i] == 0):
        color = 'b'
    elif(db_clusters[i] == 1):
        color = 'g'
    plt.scatter(data[i,0], data[i,1], c = color)
    
ari[3] = sm.adjusted_rand_score(db_clusters, clusters_reels)



spect = sc.SpectralClustering(n_clusters = 2, n_init = 10, affinity = 'rbf', n_neighbors=10).fit(Z)
spect_clusters = spect.labels_

for i in range(len(clusters)): # affichage des clusters par Spectral clustering
    if (spect_clusters[i] == 1):
        color = 'b'
    elif(spect_clusters[i] == 2):
        color = 'g'
    #plt.scatter(data[i,0], data[i,1], c = color)
    
ari[4] = sm.adjusted_rand_score(spect_clusters, clusters_reels)


print(ari)
