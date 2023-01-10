
import numpy as np

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTree():
    def __init__(self, max_depth=2): #Prune with max_depth
        """
		C'est un Initializer. 
		Vous pouvez passer d'autre paramètres au besoin,
		c'est à vous d'utiliser vos propres notations
		"""
        self.max_depth = max_depth
        self.root = None
    
    def entropy(self, target_col):
	    elements,counts = np.unique(target_col,return_counts = True)
	    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
	    return entropy

    def InfoGain(self, root, left, right):
        proba_left = len(left)/len(root)
        proba_right = len(right)/len(root)
        InfoG = self.entropy(root) - (proba_left * self.entropy(left) + proba_right * self.entropy(right))
        return InfoG
    
    def get_best_split(self, data, num_features):
        best_split = {}
        max_gain = -np.math.inf
        for feature_index in range(num_features):
            feature_values = data[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                data_left = np.array([row for row in data if row[feature_index] <= threshold])
                data_right = np.array([row for row in data if row[feature_index] > threshold])
                if len(data_left) > 0 and len(data_right) > 0:
                    y, left_y, right_y = data[:, -1], data_left[:, -1], data_right[:, -1]
                    curr_gain = self.InfoGain(y, left_y, right_y)
                    if curr_gain  > max_gain:   #choisir la meilleur feature qui a le plus grand gain d'information 
                        best_split["gain"] = curr_gain
                        best_split["feature_index"] = feature_index
                        best_split["data_left"] = data_left
                        best_split["data_right"] = data_right
                        best_split["threshold"] = threshold
                        max_gain = curr_gain
        return best_split
        
    def build_tree(self, data, depth=0):
        X, y = data[:,:-1], data[:,-1]
        num_features = np.shape(X)[1]
        if depth <= self.max_depth:
            best_split = self.get_best_split(data, num_features)
            if best_split["gain"] > 0:
                left_subtree = self.build_tree(best_split["data_left"], depth + 1)  #recursivité sur la partie gauche de l'arbre
                right_subtree = self.build_tree(best_split["data_right"], depth + 1)  #recursivité sur la partie droite de l'arbre
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["gain"])
        leaf_value = max(list(y), key=list(y).count)
        return Node(value=leaf_value)
    
    def visualize_tree(self, tree=None, alenea="---"): # les arguments ajouter pour un appel recursive à droite et à gauche de l'arbre
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("Colone: "+str(tree.feature_index), "seuil <=", tree.threshold, "gain d'information", round(tree.gain, 2))
            print("%s Gauche:" % alenea, end="")
            self.visualize_tree(tree.left, alenea*2)
            print("%s Droit:" % alenea, end="")
            self.visualize_tree(tree.right, alenea*2)
    
    def train(self, train, train_labels): #vous pouvez rajouter d'autres attributs au besoin
        """
		C'est la méthode qui va entrainer votre modèle,
		train est une matrice de type Numpy et de taille nxm, avec 
		n : le nombre d'exemple d'entrainement dans le dataset
		m : le nombre d'attributs (le nombre de caractéristiques)
		
		train_labels : est une matrice numpy de taille nx1
		
		vous pouvez rajouter d'autres arguments, il suffit juste de
		les expliquer en commentaire
		
		"""
        data = np.concatenate((train, train_labels), axis=1)
        self.root = self.build_tree(data)
    
    def item_prediction(self, x, tree): #l'argument tree est l'arbre construit par le modèle et pour faire une recuresivite sur les sous-arbres à droite et gauche
        """
		Prédire la classe d'un exemple x donné en entrée
		exemple est de taille 1xm
		"""
        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.item_prediction(x, tree.left)
        else:
            return self.item_prediction(x, tree.right)
    
    def predict(self, X):
        preditions = []
        for x in X:
            preditions.append(self.item_prediction(x, self.root))
        return preditions

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