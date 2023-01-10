"""
Vous allez definir une classe pour chaque algorithme que vous allez développer,
votre classe doit contenit au moins les 3 méthodes definies ici bas, 
	* train 	: pour entrainer le modèle sur l'ensemble d'entrainement.
	* predict 	: pour prédire la classe d'un exemple donné.
	* evaluate 		: pour evaluer le classifieur avec les métriques demandées. 
vous pouvez rajouter d'autres méthodes qui peuvent vous etre utiles, mais la correction
se fera en utilisant les méthodes train, predict et evaluate de votre code.
"""

import numpy as np
from sklearn import neighbors


# le nom de votre classe
# BayesNaif pour le modèle bayesien naif
# Knn pour le modèle des k plus proches voisins

class Knn: #nom de la class à changer

	def __init__(self, k):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
		self.k = k
		
	def dist_euclidien(self, row1, row2): # calcule la distance euclidienne entre deux lignes
		dist = np.sqrt(np.sum((np.array(row1)-np.array(row2))**2))
		return dist

	def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		self.trainSet = train
		self.train_labels = train_labels

        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		distances = []
		for i in range(len(self.trainSet)):
			distance = self.dist_euclidien(self.trainSet[i], x)
			distances.append((self.train_labels[i], distance))
		distances.sort(key=lambda tuple: tuple[1]) # sort tuple ascending by distance 
		classes = []
		for i in range(self.k):
			classes.append(distances[i][0])
		pred = max(set(classes), key=classes.count)
		return pred

        
	def evaluate(self, X, y):
		"""
		c'est la méthode qui va evaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le mobre d'attribus (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		predictions = []
		for item in X:
			predictions.append(self.predict(item))

		#matrice de confusion
		C = len(np.unique(y)) # Number of classes 
		matrix = np.zeros((C, C))
		for i in range(len(y)):
			matrix[y[i]][predictions[i]] += 1

		# accuracy
		accuracy = round(((predictions == y).sum() / len(y))*100, 2 )

		# precision
		precision = [matrix[i][i]/sum(matrix[j][i] for j in range(matrix.shape[0])) for i in range(matrix.shape[0])]

		#recall 
		recall = [matrix[i][i]/sum(matrix[i][j] for j in range(matrix.shape[0])) for i in range(matrix.shape[0])]

		#F1 score
		f_score = [(2*p*r)/(p+r) for p,r in zip(precision, recall)]
		
		return {'matrice_confusion': matrix, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_score': f_score}
        
	
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.