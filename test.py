# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

firststart = time.time()

df = pd.read_csv("Donnees_projet_2021/aggregation.txt") # import des données


head = str(df.columns[0]).split('\t', 3)
df = df["15.55	28.65	2"].str.split('\t',3, expand = True)

df.loc[-1] = head  # adding a row
df.index = df.index + 1  # shifting index
df = df.sort_index()  # sorting by index
    
newhead = ["Car 1", "Car2", "Cluster"]
for i in range(3):
    df = df.rename(columns = {i : newhead[i]}) 
    
    data = np.array(df).astype(np.float64()) #conversion en tableau np de float      
    
    for i in range(len(data)): # affichage des vrais clusters
        if(data[i,2]==1):
            color = 'b'
        elif(data[i,2]==2):
            color = 'g'
        elif(data[i,2]==3):
            color = 'm'
        elif(data[i,2]==4):
            color = 'y'
        elif(data[i,2]==5):
            color = 'r'
        elif(data[i,2]==6):
            color = 'c'
        elif(data[i,2]==7):
            color = 'k'
        #plt.scatter(data[i,0],data[i,1], c = color)
ari = [0,0,0,0,0]
t1 = [0,0,0,0,0]  
for k in range(100):
    
    from sklearn.preprocessing import StandardScaler
    
    X = data
    scaler = StandardScaler()
    Z = scaler.fit_transform(X) #centrage/reduction des donnes
    
    
    
    import scipy.cluster.hierarchy as sch
    import sklearn.metrics as sm
    
    t = 8
    link = sch.linkage(Z, method='ward', optimal_ordering=True,)
    #dn = sch.dendrogram(link, color_threshold=t)
    
    plt.show()
    
    start = time.time()
    clusters = sch.fcluster(link, t, criterion='distance' )
    coef = sm.silhouette_score(Z, clusters)
    
    for i in range(len(clusters)): # affichage des clusters par CAH
        if (clusters[i] == 1):
            color = 'b'
        elif(clusters[i] == 2):
            color = 'g'
        elif(clusters[i] == 3):
            color = 'm'
        elif(clusters[i] == 4):
            color = 'y'
        elif(clusters[i] == 5):
            color = 'r'
        elif(clusters[i] == 6):
            color = 'c'
        elif(clusters[i] == 7):
            color = 'k'
        #plt.scatter(data[i,0], data[i,1], c = color)
    
    clusters_reels = data[:,2]
    ari[0] += sm.adjusted_rand_score(clusters, clusters_reels)
    end = time.time()
    elapsed = end - start
    t1[0] += float(elapsed)
    
    import sklearn.cluster as sc
    OMP_NUM_THREADS=4
    
    start = time.time()
    kmeans = sc.KMeans(n_clusters=8, init='k-means++', n_init=10).fit(X)
    labels = kmeans.labels_
    initia = kmeans.inertia_
    
    for i in range(len(clusters)): # affichage des clusters par Kmeans
        if (labels[i] == 1):
            color = 'b'
        elif(labels[i] == 2):
            color = 'g'
        elif(labels[i] == 3):
            color = 'm'
        elif(labels[i] == 4):
            color = 'y'
        elif(labels[i] == 5):
            color = 'r'
        elif(labels[i] == 6):
            color = 'c'
        elif(labels[i] == 7):
            color = 'k'
        #plt.scatter(data[i,0], data[i,1], c = color)
    
    ari[1] += sm.adjusted_rand_score(labels, clusters_reels)
    end = time.time()
    elapsed = end - start
    t1[1] += float(elapsed)
    
    
    from sklearn.mixture import GaussianMixture
    
    start = time.time()
    gm = GaussianMixture(n_components = 7, covariance_type= 'full', n_init=10).fit(Z)
    gm_clusters = gm.predict(Z)
    
    for i in range(len(clusters)): # affichage des clusters par Gaussian mixture
        if (gm_clusters[i] == 1):
            color = 'b'
        elif(gm_clusters[i] == 2):
            color = 'g'
        elif(gm_clusters[i] == 3):
            color = 'm'
        elif(gm_clusters[i] == 4):
            color = 'y'
        elif(gm_clusters[i] == 5):
            color = 'r'
        elif(gm_clusters[i] == 6):
            color = 'c'
        elif(gm_clusters[i] == 0):
            color = 'k'
        #plt.scatter(data[i,0], data[i,1], c = color)
        
    ari[2] += sm.adjusted_rand_score(gm_clusters, clusters_reels)
    end = time.time()
    elapsed = end - start
    t1[2] += float(elapsed)
    
    
    start = time.time()
    dbscan = sc.DBSCAN(eps = 0.3, min_samples = 7,algorithm='auto').fit(Z)
    db_clusters = dbscan.labels_
    
    for i in range(len(clusters)): # affichage des clusters par DBSCAN
        if (db_clusters[i] == 1):
            color = 'b'
        elif(db_clusters[i] == 2):
            color = 'g'
        elif(db_clusters[i] == 3):
            color = 'm'
        elif(db_clusters[i] == 4):
            color = 'y'
        elif(db_clusters[i] == 5):
            color = 'r'
        elif(db_clusters[i] == 6):
            color = 'c'
        elif(db_clusters[i] == 0):
            color = 'k'
        #plt.scatter(data[i,0], data[i,1], c = color)
        
    ari[3] += sm.adjusted_rand_score(db_clusters, clusters_reels)
    end = time.time()
    elapsed = end - start
    t1[3] += float(elapsed)
    
    
    start = time.time()
    spect = sc.SpectralClustering(n_clusters = 7, n_init = 10, affinity = 'rbf', n_neighbors=10).fit(Z)
    spect_clusters = spect.labels_
    
    for i in range(len(clusters)): # affichage des clusters par Spectral clustering
        if (spect_clusters[i] == 1):
            color = 'b'
        elif(spect_clusters[i] == 2):
            color = 'g'
        elif(spect_clusters[i] == 3):
            color = 'm'
        elif(spect_clusters[i] == 4):
            color = 'y'
        elif(spect_clusters[i] == 5):
            color = 'r'
        elif(spect_clusters[i] == 6):
            color = 'c'
        elif(spect_clusters[i] == 0):
            color = 'k'
        #plt.scatter(data[i,0], data[i,1], c = color)
        
    ari[4] += sm.adjusted_rand_score(spect_clusters, clusters_reels)
    end = time.time()
    elapsed = end - start
    t1[4] += float(elapsed)
    print(k)


print(ari)
print(t1)
end = time.time()
elapsed = end - firststart

print(f'Temps d\'exécution : {elapsed:.2}s')
