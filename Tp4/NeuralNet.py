from matplotlib import pyplot as plt
import numpy as np



# le nom de votre classe
# DecisionTree pour l'arbre de décision
# NeuralNet pour le réseau de neurones

class NeuralNet(object):
	def __init__(self, inputSize, hiddenSize, outputSize, initialisation): 
		"""
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""       
		 # size of layers
		self.inputSize = inputSize
		self.outputSize = outputSize 
		self.hiddenSize = hiddenSize 
		self.initialisation = initialisation
		#notre reseau liste de dictionnaire
		self.network = []
		if self.initialisation == 'random':
			hidden_layer = [{'w':[np.random.uniform(-1, 1) for i in range(self.inputSize + 1)]} for i in range(self.hiddenSize)]
			self.network.append(hidden_layer)
			output_layer = [{'w':[np.random.uniform(-1, 1) for i in range(self.hiddenSize + 1)]} for i in range(self.outputSize)]
			self.network.append(output_layer)
		if self.initialisation == 'zeros':
			hidden_layer = [{'w':[0 for i in range(self.inputSize + 1)]} for i in range(self.hiddenSize)]
			self.network.append(hidden_layer)
			output_layer = [{'w':[0 for i in range(self.hiddenSize + 1)]} for i in range(self.outputSize)]
			self.network.append(output_layer)
	
	# pour le forward
	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))
	
	# pour le backward
	def sigmoid_prim(self, z):
		return self.sigmoid(z) * (1.0 - self.sigmoid(z))
	
	def forward(self, item):
		inputs = item
		for layer in self.network:
			new_inputs = []
			for neuron in layer:
				a = neuron['w'][-1]
				for i in range(len(neuron['w'])-1):
					a += neuron['w'][i] * inputs[i]
				neuron['output'] = self.sigmoid(a)
				new_inputs.append(neuron['output'])
			inputs = new_inputs
		return inputs

	def backward(self, expected):
		for i in reversed(range(len(self.network))):
			layer = self.network[i]
			errors = list()
			if i != len(self.network)-1:
				for j in range(len(layer)):
					error = 0
					for neuron in self.network[i + 1]:
						error += (neuron['w'][j] * neuron['d'])
					errors.append(error)
			else:
				for j in range(len(layer)):
					neuron = layer[j]
					errors.append(neuron['output'] - expected[j])
			for j in range(len(layer)):
				neuron = layer[j]
				neuron['d'] = errors[j] * self.sigmoid_prim(neuron['output'])

	def train(self, train, train_labels ,learning_rate, n_epoch):
		"""
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
		errors = []
		for epoch in range(n_epoch):
			error = 0
			for i in range(len(train)):
				outputs = self.forward(train[i])
				expected = [0 for i in range(self.outputSize)]
				expected[train_labels[i]] = 1
				error += sum([(expected[j] - outputs[j]) ** 2 for j in range(len(expected))])
				self.backward(expected)
				# mise a jours des poids
				for j in range(len(self.network)):
					inputs = train[i][:-1]
					if j != 0:
						inputs = [neuron['output'] for neuron in self.network[j - 1]]
					for neuron in self.network[j]:
						for k in range(len(inputs)):
							neuron['w'][k] -= learning_rate * neuron['d'] * inputs[k]
						neuron['w'][-1] -= learning_rate * neuron['d']
			errors.append(error/100)
			#print("Epoch ", epoch + 1 , "==== Erreur: ", error/100)
		#plt.plot(range(n_epoch), errors)
		#plt.xlabel('Number of epochs')
		#plt.ylabel('Error')
		#plt.title('Zeros initialisation')
		#plt.show()
	
	def item_predict(self, x):
		"""
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
		y_preds = self.forward(x)
		y_pred = list(y_preds).index(max(y_preds))
		return y_pred
	
	def predict(self, X):
		prediction = []
		for item in X:
			prediction.append(self.item_predict(item))
		return prediction

	def evaluate(self, X, y):
		"""
		c'est la méthode qui va évaluer votre modèle sur les données X
		l'argument X est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple de test dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		y : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		"""
		y_predict = self.predict(X)
        #matrice de confusion
		matrix = np.zeros([len(np.unique(y)),len(np.unique(y))])
		for i in range(len(y)):
			matrix[int(y[i])][int(y_predict[i])] += 1
		#accuracy 
		accuracy = round(sum(matrix[i][i] for i in range(matrix.shape[0]))/sum(matrix[i][j] for i in range(matrix.shape[0]) for j in range(matrix.shape[0])) , 2)
		#precision :  
		values_p = [matrix[i][i]/sum(matrix[j][i] for j in range(matrix.shape[0])) for i in range(matrix.shape[0])]
		precision = dict(list(enumerate(values_p)))
		#recall : 
		values_r = [matrix[i][i]/sum(matrix[i][j] for j in range(matrix.shape[0])) for i in range(matrix.shape[0])]
		recall = dict(list(enumerate(values_r)))
		#f1_score
		values_f1 = [(2* p * r) / (p + r) for p,r in zip(values_p, values_r)]
		f1_score = dict(list(enumerate(values_f1)))
		return {'confusion_matrix': matrix, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
