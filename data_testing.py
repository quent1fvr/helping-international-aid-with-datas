# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:16:04 2021

@author: Ambroise
"""
#importation des différentes bibliothèques utiles 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.cluster as sc 
from sklearn import metrics  
from sklearn import cluster
import scipy.cluster.hierarchy as scp
from sklearn.mixture import GaussianMixture as Gauss
from sklearn.cluster import DBSCAN as DBS
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

coefs_Silhouette =[0,0,0,0,0] # cela nous permettra de comparer les différents algorithmes en terme de performance
#0° K-means
#1° CAH
#2° DBSCAN
#3° mélange de gaussienne
#4° Spectal clustering

df = pd.read_csv("Donnees_projet_2021/data.csv") # import des données
df2 = pd.read_csv("Donnees_projet_2021/data.csv") 

nom_pays = df.pop("country") # on retire la colonne contenant les pays, car non quantitative

caract = df.columns

#df.info()                          # Description des données
descr = df.describe()
df.hist()
#print("Shape : ")
#print(df.shape)
#print("Head : ")
#print(df.head())

data = np.array(df) # Conversion en tableau numpy

# PARTIE NETTOYAGE DE DONNEES

zero_values = np.argwhere(data == 0)    # Récuperation des indices données nulles
na_values = np.append(np.argwhere(np.isnan(data)),zero_values)  # Des données NaN
na_values = np.append(na_values, np.argwhere(data >= 100000))   # Des données aberrantes

for i in range(int(na_values.size/2)):  # Afficher quelles données sont à remplacer
    a = na_values[i*2]
    b = na_values[i*2 + 1]
    car = nom_pays[a] + " : " + caract[b]
    #print(car)
    
data_toadd = np.array([1.87, 31676, 6.82, 67294, 72.59, 51812, 115873, 0, 40284, 63543])    # Ajout des données manquantes

for i in range(int(na_values.size/2)):
    if(data_toadd[i] == 0):
        next
    else : 
        data[na_values[2*i],na_values[2*i+1]] = data_toadd[i]


df_cleaned = pd.DataFrame(data=data)    # Description des données nétoyées
df_cleaned.hist()
descr_cleaned = df_cleaned.describe()

cor = df_cleaned.corr(method = "pearson")
pd.plotting.scatter_matrix(df_cleaned)
plt.show()
#ANALYSE DES CORRELATIONS

#variables les plus corrélées négativement :  child mortality and life expectation 
#variables les plus corrélées positivement : GDP and income, total fertility and child mortality
#variables les moins corrélées : life expectation and imports, health and imports

# on comprend que le PIB et et le revenu sont corrélés car ils sont tout deux liés à la santé financière du pays
# de même, il est compréhensible que l'espérance de vie et la mortalité infantile sont corrélées, plus l'espérance de vie augmente 
# plus la mortalité infantile aura tendance à diminuer 


#PHASE ACP

X = data
scaler = StandardScaler()
Z = scaler.fit_transform(X) # Centrage/reduction des donnes
n=167
p=9

#on va déterminer le nombre de composantes principales idéales à conserver 

acp_test = PCA(svd_solver='full') #instanciation de l'ACP 

datacp_test = acp_test.fit_transform(Z)

#règle de Kaiser : 

print(acp_test.explained_variance_) 

# 4 vp>1 donc conservation de 4 composantes principales selon la règle de KAISER

# Méthode du coude : 

plt.plot(np.arange(0,9),acp_test.explained_variance_) 
plt.title("ACP :Variance expliquée pour chaque composante principale")
plt.show()

#Par méthode du coude, on peut décider de conserver 4 à 5 composantes principales


# #cumul de variance expliquée

plt.plot(np.arange(0,9),np.cumsum(acp_test.explained_variance_ratio_))
plt.title("ACP :Cumul de Variance expliquée en fonction du nombre de composantes")
plt.ylabel("cumul de variance expliquée")
plt.xlabel("nombre de composantes ")
plt.show()

#on remarque qu'en conservant 4 composantes principales, on peut conserver 90% de variance cumulée

# # on conserve donc finalement  4 axes principaux  :


acp= PCA(n_components=9, svd_solver='full')
datacp = acp.fit_transform(Z)



# # visualisation dans le premier plan principal : 
x0=datacp[:,0]
y0=datacp[:,1]

# #positionnement des individus dans le premier plan :
plt.scatter(x0,y0)
plt.title("premier plan principal de l'ACP")
for i in range(len(x0)):
     plt.annotate(nom_pays[i], xy=(x0[i], y0[i]))
plt.show()

#visualisation dans le second plan principal : 
x1=datacp[:,2]
y1=datacp[:,3]


plt.scatter(x1,y1)
plt.title("second plan principal de l'ACP")
for i in range(len(x1)):
     plt.annotate(nom_pays[i], xy=(x1[i], y1[i]))
plt.show()


#racine carrée des valeurs propres
eigval = ((n-1)/n)*acp.explained_variance_
sqrt_eigval = np.sqrt(eigval)
#print(sqrt_eigval)


#corrélation des variables avec les axes
corvar = np.zeros((9,9))
for k in range(9):
 corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]



#Variables expliquant le premier plan factoriel :


#cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

#affichage des étiquettes (noms des variables)
for j in range(p):
 plt.annotate(caract[j],(corvar[j,0],corvar[j,1]))
 

#ajouter les axes
plt.plot([-1,1],[0,0],linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],linestyle='-',linewidth=1)
plt.title("Variables expliquant le premier plan factoriel de l'ACP")

#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
#affichage
plt.show()

#Variables expliquant le second plan factoriel :

#cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

#affichage des étiquettes (noms des variables)
for j in range(p):
 plt.annotate(caract[j],(corvar[j,2],corvar[j,3]))
 

#ajouter les axes
plt.plot([-1,1],[0,0],linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],linestyle='-',linewidth=1)
plt.title("Variables expliquant le second plan factoriel de l'ACP")

#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
#affichage
plt.show()



# ANALYSE DE L'ACP

#Individus qui contribuent le plus à l’axe 1 : 

# sur la gauche de l'axe 1, on retrouve beaucoup de pays d'Afrique : Nigeria, Chad, Centrafrique, Niger, Mali, Congo, Guinée, Burkina Faso, Mozambique
# Mais également Haiti et Sierra Leone

# A l'inverse, sur la droite de l'axe, on retrouve le luxembourg, Singapour, le Qatar, La Suisse, Pays-Bas, Irelande, Malte, USA



#Individus qui contribuent le plus à l’axe 2:

# Tout en haut, on retrouve les pays qui n'ont pas d'accès à la mer ou  des Iles : Singapour, Luxembourg, Malte, les Seychelles 
#on retrouve en bas les Etats-Unis, le Japon et le Brésil. 


# variable expliquant l'axe 1 : 

#Sur le cercle des corrélations, on remarque que l'axe 1 est déterminé par le revenu, GDP, life expectation, soit les variables financières  pour la droite de l'axe
# et la mortalité infantile et la fertilité pour la gauche de l'axe 




#l'axe 2 quant à lui est déterminé par les importations/exportations en haut et aucune veritable variable en bas. 

# le second plan factoriel est inexploitable selon nous. On remarque seulement que l'axe 3 est déterminé à droite par l'inflation et la santé à gauche.
#l'axe 4 n'apporte aucune autre information 


# Analyse croisée de l'axe 1 : 

#on remarque que les pays D'Afrique avec Haiti et Sierra Leone sont ceux qui souffrent
# le plus de la mortalité infantile et qui ont les développements économiques les plus faibles. 
# Ce sont également des pays qui traditionnellement font plus d'enfants que les pays "développés du nord". 

# A l'inverse, les pays les plus à droite de l'axe sont ceux avec la meilleure santé économique, il n'est pas étonnant d'y retrouver la Suisse Et le Qatar


# Analyse croisée de l'axe 2 :  

#En haut de l'axe 2, on retrouve les pays qui dépendent de plus de l'importation (et dans une moindre mesure l'exportation ), 
# c'est pour ça que l'on retrouve les îles (Seychelles, Irelande) et les pays qui n'ont pas d'accés directs à la mer comme le luxembourg. 

#Conclusion de l'ACP : 

#L'axe 1 est le plus déterminant pour prioriser les pays nécessitant l'aide internationale, qui sont ceux les plus à gauche de l'axe 1. 
# Le monde devrait les aider notamment dans les infrastructures de santé pour limiter la mortalité infantile, et les aider également à se développer économiquement.




# # PARTIE 2.5 CLUSTERING DES DONNEES

# #— K-means Clustering:

# #utilisation du critère du coude pour déterminer le nombre de clusters à garder

resultat = np.arange(9,dtype="double")
for k in np.arange(9):
    
     test_kmean = cluster.KMeans(n_clusters=k+2)
     test_kmean.fit(X)
     
     resultat[k] = test_kmean.inertia_
#print(resultat)
plt.title(" K-means : Inertie en fonction du nombre de clusters")
plt.plot(np.arange(2,11,1),resultat)
plt.show()

# D'après la méthode du coude, il est intéressant de conserver entre 4 et 6 clusters 

#silhouette en fonction du nombre de clusters
resultat = np.arange(9,dtype="double")
for k in np.arange(9):
    
    test_kmean = cluster.KMeans(n_clusters=k+2)
    test_kmean.fit(X)
    resultat[k] = metrics.silhouette_score(X,test_kmean.labels_)
#print(resultat)


plt.title("K-means : coefficient de Silhouette en fonction du nombre de clusters")
plt.plot(np.arange(2,11,1),resultat)
plt.show()

 #Découper en 2 classes semblent être la meilleure partition des individus en observant l'évolution du coefficient de silhouette en fonction du nombre de clusters
 # on voit que le coefficient reaugmente légèrement pour k=8
 #   il peut être plus intéressant de visualiser 8  groupes de pays plutôt que 2 pour l'analyse des pays les plus en difficultés
 # On pourrait faire un lien avec "pays très développés", "pays développés" "pays en voie de développement +" , "pays en voie de développement -", "pays pauvres"  "pays très pauvres" : ceux qui ont besoin de l'aide internationale.
 
 #on conserve donc 8 clusters 

kmean = cluster.KMeans(n_clusters=8, init="k-means++") # on choisit donc 4 clusters, que l'on initialise avec la méthode fiable "kmeans++", on initialise 30fois pour limiter les erreurs
data_final = kmean.fit(X)

idk = np.argsort(kmean.labels_)


idk = np.argsort(kmean.labels_)
#print(nom_pays[idk],kmean.labels_[idk])

#on visualise le résultat du clustering dans le premier plan principal grâce à l'ACP :



plt.scatter(x0,y0, c=kmean.labels_,cmap="rainbow")
plt.title("Résultat après ACP de l'algorithme K-Means clustering avec K=8")
for i,label in enumerate(nom_pays):
     plt.annotate(nom_pays[i], xy=(x0[i], y0[i]))
coefs_Silhouette[0]=metrics.silhouette_score(X,kmean.labels_)
plt.show()

# le cluster contenant les pays les moins avancés  est composé : d'Haiti, Nigeria,  Angola, Chad, Soudan, république démocratique du congo, Mauritanie Ghana, soudan
#beaucoup de pays d'Afrique sont concernés par ce cluster


#— CAH : Clustering hiérarchique ascendant.

t=6
CAh_ward = scp.linkage(Z, method='ward', metric='euclidean',optimal_ordering="True") #on emploie la méthode de war connue pour être la plus efficace
Dendro_CAh_ward= scp.dendrogram(CAh_ward, color_threshold=t)
plt.title("Dendrogramme du Cah")
plt.show()


#En observant le dendrogramme, la meilleure option serait de conserver une dizaine de clusters . on décide de prendre t=6 clusters pour cibler les pays les plus en difficultés
#comme l'algorithme K-means

##print(fcluster)

groupes_cah = scp.fcluster(CAh_ward,t=6,criterion='distance')
#print(groupes_cah)
#index triés des groupes
idg = np.argsort(groupes_cah)
#affichage des observations et leurs groupes
#print(pd.DataFrame(groupes_cah[idg],nom_pays[idg]))


plt.scatter(x0,y0, c=groupes_cah,cmap="rainbow")
plt.title("Résultat après ACP de l'algorithme CAH clustering avec K=8")
for i,label in enumerate(nom_pays):
     plt.annotate(nom_pays[i], xy=(x0[i], y0[i]))
plt.show()
coefs_Silhouette[1]= metrics.silhouette_score(Z,groupes_cah)


# Sierra Leone, Centrafrique, Chad, Kenya, benin, cote d'ivoire République démocratique du Congo, Mali, Guinée, Burkina Faso, Malawi, Tanzanie, Rwanda, Togo, Sénégal, Comoros, Niger, uganda, burundi 
#Ce sont encore une fois beaucoup de pays d'Afrique qui aurait le plus besoin de l'aide internationale.
# les résultats obtenus sont très proches que pour le K-mean clustering






#— DBSCAN :


# la première étape va être de déterminer la distance optimale pour appliquer l'algorithme DBSCAN, on se base sur la recherche des plus
#proches voisins

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.title("DBSCAN, recherche de Epsilon optimal")
plt.plot(distances)
plt.show()

#on remarque que la courbure maximale est obtenue pour epsilon égal à 3200


db = DBS(eps = 3200, min_samples =5).fit(X) 
db_labels = db.labels_ 

plt.scatter(x0,y0, c=db_labels,cmap="rainbow")
plt.title("Résultat après ACP de l'algorithme DBSCAN clustering")
for i,label in enumerate(nom_pays):
     plt.annotate(nom_pays[i], xy=(x0[i], y0[i]))
plt.show()


coefs_Silhouette[2]= metrics.silhouette_score(X, db_labels)   

# en dépit d'avoir le coefficient de silhouette le plus satisfaisant, il ne donne pas des résultats 
#satisfaisants pour prendre une décision car il ne forme pas assez de clusters pour cibler les pays les plus démunis.

# MÉLANGE GAUSSIEN

gm = Gauss(n_components =10, random_state = 0).fit_predict(X)

plt.scatter(x0,y0, c=gm,cmap="rainbow")
plt.title("Résultat après ACP de l'algorithme dumodèle de mélange de Gaussiennes ")
for i,label in enumerate(nom_pays):
     plt.annotate(nom_pays[i], xy=(x0[i], y0[i]))
plt.show()
coefs_Silhouette[3]= metrics.silhouette_score(X,gm)   

# Il fournit des résultats assez similaires à K-means et Cah même si l'on obtient un cluster de pays les plus démunis assez conséquents 
    
    
    
    
    
    
    
#— Spectral clustering :

resultat = np.arange(9,dtype="double")
for k in np.arange(9):
    
    test_spectral = SpectralClustering(n_clusters=k+2)
    test_spectral.fit(X)
    resultat[k] = metrics.silhouette_score(X,test_spectral.labels_)
#print(resultat)


plt.title("Spectral clustering : Silhouette en fonction du nombre de clusters")
plt.xlabel("nombre de clusters")
plt.plot(np.arange(2,11,1),resultat)
plt.show()

clustering = SpectralClustering(n_clusters=9).fit(X)
labels=clustering.labels_

plt.scatter(x0,y0, c=labels,cmap="rainbow")
plt.title("Résultat après ACP de l'algorithme Spectral clustering avec K=8")
for i,label in enumerate(nom_pays):
     plt.annotate(nom_pays[i], xy=(x0[i], y0[i]))
plt.show()

# cet algorithme fournit le pire clustering, on ne distingue aucun cluster, c'est inexploitable

coefs_Silhouette[4]= metrics.silhouette_score(X, labels)



print(coefs_Silhouette)





#Au sens du coefficient de silhouette, le classement en terme de performance est le suivant : 
#1° DBSCAN le plus performant mais peu exploitable
#2° modèle de mélange de Gaussiennes, le plus convaincant au regard du coefficient de silhouette et des clusters obtenus
#3°K-means #résultat proche du mélange de gaussiennes
#4° CAh # résultat proche de K-means et du mélange de gaussiennes 
#5° Spectral clustering # le moins efficace, ne forme pas de clusters 

#En conservant les résultats du clustering obtenu grâce à l'algorithme DBSCAN, mais il ne fournit pas assez de clusters pour l'exploiter dans 
#la recherche du groupe de pays le plus démuni.
# 
#on dresse une liste des pays qui ont le plus besoin de l'aide internationale sur la base du mélange gaussien, confirmé par K-means et Cah 
# par ordre d'importance décroissant: 

# 1ER groupe HAITI, NIGERIA, CHAD, CENTRAFRIQUE, MALI, SIERRA LEONE, BURKINA FASO, NIGER, CONGO, BURKINA FASO  = groupe nécessitant le + l'aide internationale

#2ème groupe (dans le meme cluster): MOZAMBIQUE, GUINEE, ZAMBIE, MALAWI, AFGHANISTAN, CAMEROUN, TANZANIE, UGANDA,BURUNDI, BENIN, Gambie, 


