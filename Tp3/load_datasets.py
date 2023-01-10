import numpy as np
import random

def load_iris_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Iris

    Args:
        train_ratio: le ratio des exemples qui vont etre attribués à l'entrainement,
        le reste des exemples va etre utilisé pour les tests.
        Par exemple : si le ratio est 50%, il y aura 50% des exemple (75 exemples) qui vont etre utilisés
        pour l'entrainement, et 50% (75 exemples) pour le test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.
    
    # Vous pouvez utiliser des valeurs numériques pour les différents types de classes, tel que :
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}
    
    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/bezdekIris.data', 'r')
    
    
    # TODO : le code ici pour lire le dataset
    
    # REMARQUE très importante : 
	  # remarquez bien comment les exemples sont ordonnés dans 
    # le fichier du dataset, ils sont ordonnés par type de fleur, cela veut dire que 
    # si vous lisez les exemples dans cet ordre et que si par exemple votre ration est de 60%,
    # vous n'allez avoir aucun exemple du type Iris-virginica pour l'entrainement, pensez
    # donc à utiliser la fonction random.shuffle pour melanger les exemples du dataset avant de séparer
    # en train et test.
    data = []
    for line in f:
      data.append(line.strip('\n').split(','))

    data_list = []
    for item in data:
      if item[4] == 'Iris-virginica':
        item[4] = conversion_labels['Iris-virginica']
      if item[4] == 'Iris-setosa':
        item[4] = conversion_labels['Iris-setosa']
      if item[4] == 'Iris-versicolor':
        item[4] = conversion_labels['Iris-versicolor']
      list_f = [float(x) for x in item[:-1]]
      list_f.append(item[4])
      data_list.append(list_f)

    np.random.shuffle(data_list)

    #split train and test
    data_train = data_list[:int(len(data_list)*train_ratio)]
    data_test = data_list[int(len(data_list)*train_ratio):]

    X_train = []
    y_train = []
    for item in data_train:
      X_train.append(item[:-1])
      y_train.append(item[-1])
    X_test = []
    y_test = []
    for item in data_test:
      X_test.append(item[:-1])
      y_test.append(item[-1])
    train = np.array(X_train)
    train_labels = np.array(y_train)
    test = np.array(X_test)
    test_labels = np.array(y_test)
    # Tres important : la fonction doit retourner 4 matrices (ou vecteurs) de type Numpy. 
    return(train, train_labels, test, test_labels)

	
	
def load_wine_dataset(train_ratio):
    """Cette fonction a pour but de lire le dataset Binary Wine quality

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels
		
        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.
		
        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]
        
        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.
		
        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    
    random.seed(1) # Pour avoir les meme nombres aléatoires à chaque initialisation.

    # Le fichier du dataset est dans le dossier datasets en attaché 
    f = open('datasets/binary-winequality-white.csv', 'r')

	
    # TODO : le code ici pour lire le dataset
    data = []
    for line in f:
      data.append(line.strip('\n').split(','))

    #str to float
    data_list = []
    for item in data:
      list_f = [float(x) for x in item[:-1]]
      list_f.append(int(float(item[-1])))
      data_list.append(list_f)

    np.random.shuffle(data_list)

    #split train and test
    data_train = data_list[:int(len(data_list)*train_ratio)]
    data_test = data_list[int(len(data_list)*train_ratio):]
    
    X_train = []
    y_train = []
    for item in data_train:
      X_train.append(item[:-1])
      y_train.append(item[-1])
    X_test = []
    y_test = []
    for item in data_test:
      X_test.append(item[:-1])
      y_test.append(item[-1])
    train = np.array(X_train)
    train_labels = np.array(y_train)
    test = np.array(X_test)
    test_labels = np.array(y_test)

	# La fonction doit retourner 4 structures de données de type Numpy.
    return (train, train_labels, test, test_labels)

def load_abalone_dataset(train_ratio):
    """
    Cette fonction a pour but de lire le dataset Abalone-intervalles

    Args:
        train_ratio: le ratio des exemples (ou instances) qui vont servir pour l'entrainement,
        le rest des exemples va etre utilisé pour les test.

    Retours:
        Cette fonction doit retourner 4 matrices de type Numpy: train, train_labels, test, et test_labels

        - train : une matrice numpy qui contient les exemples qui vont etre utilisés pour l'entrainement, chaque 
        ligne dans cette matrice représente un exemple d'entrainement.

        - train_labels : contient les labels (ou les étiquettes) pour chaque exemple dans train, de telle sorte
          que : train_labels[i] est l'etiquette pour l'exemple train[i]

        - test : une matrice numpy qui contient les exemples qui vont etre utilisés pour le test, chaque 
        ligne dans cette matrice représente un exemple de test.

        - test_labels : contient les étiquettes pour chaque exemple dans test, de telle sorte
          que : test_labels[i] est l'etiquette pour l'exemple test[i]
    """
    f = open('datasets/abalone-intervalles.csv', 'r') # La fonction doit retourner 4 matrices (ou vecteurs) de type Numpy.
    data = []
    for line in f:
      data.append(line.strip('\n').split(','))

    #str to float
    data_list = []
    for item in data:
      if item[0] == 'M':
        item[0] = 0
      if item[0] == 'F':
        item[0] = 1
      if item[0] == 'I':
        item[0] = 2
      list_f = [float(x) for x in item[:-1]]
      list_f.append(int(float(item[-1])))
      data_list.append(list_f)

    np.random.shuffle(data_list)

    #split train and test
    data_train = data_list[:int(len(data_list)*train_ratio)]
    data_test = data_list[int(len(data_list)*train_ratio):]
    
    X_train = []
    y_train = []
    for item in data_train:
      X_train.append(item[:-1])
      y_train.append(item[-1])
    X_test = []
    y_test = []
    for item in data_test:
      X_test.append(item[:-1])
      y_test.append(item[-1])
    train = np.array(X_train)
    train_labels = np.array(y_train)
    test = np.array(X_test)
    test_labels = np.array(y_test)
    
    return (train, train_labels, test, test_labels)
