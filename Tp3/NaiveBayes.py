import numpy as np
from sklearn import neighbors

class BayesNaif: #nom de la class à changer

	def __init__(self, **kwargs):
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""

		
	def normal_distrubution(self, x, mean, std):
		return (1/(np.sqrt(2*np.pi)*std)) * np.exp(-((x-mean)**2/(2*std**2)))

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
		classes = np.unique(train_labels)
		means = {}
		stds = {}
		classe_priors = {}
		for classe in classes:
			means[classe] = np.mean(train[train_labels == classe , :] , axis = 0)
			stds[classe] = np.std(train[train_labels == classe , :] , axis = 0)
			classe_priors[classe] = train[train_labels == classe , :].shape[0] / train.shape[0]

		self.trainSet = train
		self.train_labels = train_labels
		self.means = means
		self.stds = stds
		self.classe_priors = classe_priors

        
	def predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		predictions = {}
		for classe in np.unique(self.train_labels):
			poste = 1
			for i in range(len(x)):
				poste = poste * self.normal_distrubution(x[i], self.means[classe][i], self.stds[classe][i])
			predictions[classe] = self.classe_priors[classe] * poste
		return max(predictions, key=predictions.get)
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
		accuracy = round(((predictions == y).sum() / len(y))*100, 2)

		# precision
		precision = [matrix[i][i]/sum(matrix[j][i] for j in range(matrix.shape[0])) for i in range(matrix.shape[0])]

		#recall 
		recall = [matrix[i][i]/sum(matrix[i][j] for j in range(matrix.shape[0])) for i in range(matrix.shape[0])]

		#F1 score
		f_score = [(2*p*r)/(p+r) for p,r in zip(precision, recall)]
		
		return {'matrice_confusion': matrix, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_score': f_score}
	
	# Vous pouvez rajouter d'autres méthodes et fonctions,
	# il suffit juste de les commenter.
