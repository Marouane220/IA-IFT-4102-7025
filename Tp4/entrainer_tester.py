import numpy as np
import load_datasets
from DecisionTree import DecisionTree 
from NeuralNet import NeuralNet

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

from sklearn import metrics, tree
from sklearn import preprocessing

import matplotlib.pyplot as plt

"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entraîner votre classifieur
4- Le tester
"""

datasets = {'iris' : load_datasets.load_iris_dataset(train_ratio = 0.7),
           'wine' : load_datasets.load_wine_dataset(train_ratio = 0.7),
           'abalones' : load_datasets.load_abalone_dataset(train_ratio = 0.7)}

DecisionTree_classifier = DecisionTree(max_depth=3)


#Decision Tree
for dataset in ['iris', 'wine', 'abalones']:
	t1_dt = time.time()
	X_train, y_train = datasets[dataset][0], datasets[dataset][1].reshape(-1,1)
	X_test, y_test = datasets[dataset][2], datasets[dataset][3].reshape(-1,1)
	DecisionTree_classifier.train(X_train, y_train)
	print("\n\n Decision Tree de {}  est".format(dataset) )
	DecisionTree_classifier.visualize_tree()
	print('\n')
	t2_dt = time.time()
	evaluation_train = DecisionTree_classifier.evaluate(X_train, y_train)
	evaluation_test = DecisionTree_classifier.evaluate(X_test, y_test)
	print('Dataset : {} '.format(dataset),'\n')
	print("\n Sur les données train : \n")
	print('Notre Modèle : \n\n')
	print('Matrice de confusion : \n {}'.format(evaluation_train['confusion_matrix']))
	print('Accuracy : {} \n'.format(evaluation_train['accuracy']))
	print('Temps d’exécution : {} secondes \n'.format(round(t2_dt - t1_dt, 4)))
	classes = list(range(evaluation_train['confusion_matrix'].shape[0]))
	precision = list(evaluation_train['precision'].values())
	recall = list(evaluation_train['recall'].values())
	f1_score = list(evaluation_train['f1_score'].values())
	for i in range(len(precision)):
		print('\n precision de la classe ', i , ' est: ', precision[i])
	print('\n')
	for i in range(len(recall)):
		print('\n recall de la classe ', i , ' est: ', recall[i])
	print('\n')
	for i in range(len(f1_score)):
		print('\n f1_score de la classe ', i , ' est: ', f1_score[i])
	print('\n')

	#Sklearn Train
	clf = DecisionTreeClassifier(random_state=0)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_train)
	print('Scikit-learn: \n\n')
	print('Matrice de confusion : \n {}'.format(metrics.confusion_matrix(y_train, y_pred)))
	print(metrics.classification_report(y_train, y_pred))

	# Test
	print("Sur les données test : \n")
	print('Notre Modèle : \n\n')
	print('Matrice de confusion : \n {}'.format(evaluation_test['confusion_matrix']))
	print('Accuracy : {} \n'.format(evaluation_test['accuracy']))
	classes = list(range(evaluation_test['confusion_matrix'].shape[0]))
	precision = list(evaluation_test['precision'].values())
	recall = list(evaluation_test['recall'].values())
	f1_score = list(evaluation_test['f1_score'].values())
	for i in range(len(precision)):
		print('\n precision de la classe ', i , ' est: ', precision[i])
	print('\n')
	for i in range(len(recall)):
		print('\n recall de la classe ', i , ' est: ', recall[i])
	print('\n')
	for i in range(len(f1_score)):
		print('\n f1_score de la classe ', i , ' est: ', f1_score[i])
	print('\n')
	y_pred = clf.predict(X_test)
	print('\n Scikit-learn: \n')
	print('Matrice de confusion : \n {}'.format(metrics.confusion_matrix(y_test, y_pred)))
	print(metrics.classification_report(y_test, y_pred, digits=4))


fig, axes =  plt.subplots(3, 1, figsize=(10, 8))
fig.tight_layout(pad=5)
for dataset in ['iris', 'wine', 'abalones']:
	X_train, y_train = datasets[dataset][0], datasets[dataset][1].reshape(-1,1)
	X_test, y_test = datasets[dataset][2], datasets[dataset][3].reshape(-1,1)
	DecisionTree_classifier.train(X_train, y_train)
	training_errors = []
	test_errors = []
	line_space_train = [int(i) for i in np.linspace(0, X_train.shape[0], 11)][1:]
	line_space_test = [int(i) for i in np.linspace(0, X_test.shape[0], 11)][1:]
	y_pred_train = []
	y_pred_test = []
	for l1, l2 in zip(line_space_train, line_space_test):
		y_pred_train.append(accuracy_score(y_train[:l1], DecisionTree_classifier.predict(X_train[:l1])))
		y_pred_test.append(accuracy_score(y_test[:l2], DecisionTree_classifier.predict(X_test[:l2]))) 
	j = {'iris' : 0, 'wine': 1, 'abalones': 2}.get(dataset)
	axes[j].plot(np.arange(0, 101, 11), y_pred_train, c='r', label='Training')
	axes[j].plot(np.arange(0, 101, 11), y_pred_test, c='b', label='Test')
	axes[j].set_xlabel('Training Size')
	axes[j].set_ylabel('Score')
	axes[j].set_title('\n {} '.format(dataset), fontsize=11)
	axes[j].legend()
#plt.show()


#Neural Network
for dataset in ['iris','wine', 'abalones']:
	t1_rn = time.time()
	X_train, y_train = datasets[dataset][0], datasets[dataset][1]
	X_test, y_test = datasets[dataset][2], datasets[dataset][3]

	min_max_scaler = preprocessing.MinMaxScaler()
	X_train = min_max_scaler.fit_transform(X_train)
	X_test = min_max_scaler.fit_transform(X_test)

	inputs = X_train.shape[1]
	hiddens = 15
	outputs = len(np.unique(y_train))

	NeuralNet_classifier = NeuralNet(inputs, hiddens, outputs, 'random')

	NeuralNet_classifier.train(X_train, y_train, 0.01, 300)
	print('\n')
	t2_rn = time.time()

	evaluation_train = NeuralNet_classifier.evaluate(X_train, y_train)
	evaluation_test = NeuralNet_classifier.evaluate(X_test, y_test)

	print('Dataset : {} '.format(dataset),'\n')
	print("\n Sur les données train : \n")
	print('Notre Modèle : \n\n')
	print('Matrice de confusion : \n {}'.format(evaluation_train['confusion_matrix']))
	print('Accuracy : {} \n'.format(evaluation_train['accuracy']))
	print('Temps d’exécution : {} secondes \n'.format(round(t2_rn - t1_rn, 4)))
	classes = list(range(evaluation_train['confusion_matrix'].shape[0]))
	precision = list(evaluation_train['precision'].values())
	recall = list(evaluation_train['recall'].values())
	f1_score = list(evaluation_train['f1_score'].values())
	for i in range(len(precision)):
		print('\n precision de la classe ', i , ' est: ', precision[i])
	print('\n')
	for i in range(len(recall)):
		print('\n recall de la classe ', i , ' est: ', recall[i])
	print('\n')
	for i in range(len(f1_score)):
		print('\n f1_score de la classe ', i , ' est: ', f1_score[i])
	print('\n')

	# Test
	print("Sur les données test : \n")
	print('Notre Modèle : \n\n')
	print('Matrice de confusion : \n {}'.format(evaluation_test['confusion_matrix']))
	print('Accuracy : {} \n'.format(evaluation_test['accuracy']))
	classes = list(range(evaluation_test['confusion_matrix'].shape[0]))
	precision = list(evaluation_test['precision'].values())
	recall = list(evaluation_test['recall'].values())
	f1_score = list(evaluation_test['f1_score'].values())
	for i in range(len(precision)):
		print('\n precision de la classe ', i , ' est: ', precision[i])
	print('\n')
	for i in range(len(recall)):
		print('\n recall de la classe ', i , ' est: ', recall[i])
	print('\n')
	for i in range(len(f1_score)):
		print('\n f1_score de la classe ', i , ' est: ', f1_score[i])
	print('\n')

#plot l'initialisation des poids
fig, axes =  plt.subplots(3, 2, figsize=(15, 10))
fig.tight_layout(pad=5.0)
for dataset in ['iris', 'wine', 'abalones']:
    for init in ['random', 'zero']:
        X_train= datasets[dataset][0]
        y_train = datasets[dataset][1]
        inputs = X_train.shape[1]
        hiddens = 5
        outputs = len(np.unique(y_train))
        NeuralNet_classifier = NeuralNet(inputs, hiddens, outputs, init)
        training_accs = []
        for l in [int(i) for i in np.linspace(0, X_train.shape[0], 11)][1:]:
            model = NeuralNet_classifier.train(X_train[:l], y_train[:l], 0.08, 40)
            y_train_predict = NeuralNet_classifier.predict(X_train[:l])
            train_acc = metrics.accuracy_score(y_train[:l], y_train_predict)
            training_accs.append(train_acc)
        i = {'iris' : 0, 'wine' : 1, 'abalones' : 2}.get(dataset)
        j = {'random' : 1, 'zero' : 0}.get(init)
        axes[i][j].plot(np.arange(10, 101, 10), training_accs,
        label='training set', marker='*')
        axes[i][j].set_title('Dataset : {}, Initialisation: {}'.format(dataset, init), fontsize=15)
        axes[i][j].set_xlabel('Size of training set')
#plt.show()


params = {'nbr_neurones': range(5, 46, 5),
          'nbr_couches': range(1, 9, 2)}

erreur_n = {'iris' : [],
            'wine': [],
            'abalones' : []}

# Recherche du meilleur nombre de neurones dan la couche cachée pour chaque dataset
fig, axes =  plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5)
for dataset in ['iris', 'wine', 'abalones']:
    score_t = []
    for n in params['nbr_neurones']:
        X_train = datasets[dataset][0]
        y_train = datasets[dataset][1]
        kf = KFold(n_splits=10)
        clf = MLPClassifier(hidden_layer_sizes=(n,), max_iter=300, random_state=1)
        scores = []
        for train_indices, test_indices in kf.split(X_train):
            clf.fit(X_train[train_indices], y_train[train_indices])
            score = clf.score(X_train[test_indices], y_train[test_indices])
            scores.append(score)
        score_t.append((n, np.mean(scores)))
        erreur_n[dataset].append(round(np.mean([1-x for x in scores]), 3))
    i = {'iris' : 0, 'wine' : 1, 'abalones' : 2}.get(dataset)
    axes[i].plot(params['nbr_neurones'], erreur_n[dataset],label='training set', marker='*')
    axes[i].set_xlabel("Nombre de neurones dans la couche cachée")
    axes[i].set_ylabel('Erreur ')
    axes[i].set_title('Dataset : {} \n'.format(dataset), fontsize=15)
    axes[i].legend()
#plt.show()

best_N = {'iris' : (35,),
          'wine': (35,),
          'abalones' : (40,)}

erreur_h = {'iris' : [],
            'wine': [],
            'abalones' : []}

# Recherche du meilleur nombre des couches cachées pour chaque dataset
fig, axes =  plt.subplots(3, 1, figsize=(10, 10))
fig.tight_layout(pad=5)
for dataset in ['iris', 'wine', 'abalones']:
    score_t = []
    for n in params['nbr_couches']:
        X_train = datasets[dataset][0]
        y_train = datasets[dataset][1]
        kf = KFold(n_splits=10)
        clf = MLPClassifier(hidden_layer_sizes = n*best_N[dataset], max_iter=100, random_state=1)
        scores = []
        for train_indices, test_indices in kf.split(X_train):
            clf.fit(X_train[train_indices], y_train[train_indices])
            score = clf.score(X_train[test_indices], y_train[test_indices])
            scores.append(score)
        score_t.append((n, np.mean(scores)))
        erreur_h[dataset].append(round(np.mean([1-x for x in scores]), 3))
    i = {'iris' : 0, 'wine' : 1, 'abalones' : 2}.get(dataset)
    axes[i].plot(params['nbr_couches'], erreur_h[dataset],label='training set', marker='*')
    axes[i].set_xlabel("Nombre de couches cachées")
    axes[i].set_ylabel('Erreur ')
    axes[i].set_title('Dataset : {} \n'.format(dataset), fontsize=15)
    axes[i].legend()
#plt.show()

best_H = {'iris' : 35,
          'wine': 35,
          'abalones' : 40}


for dataset in ['iris','wine', 'abalones']:
	X_train, y_train = datasets[dataset][0], datasets[dataset][1]
	X_test, y_test = datasets[dataset][2], datasets[dataset][3]

	min_max_scaler = preprocessing.MinMaxScaler()
	X_train = min_max_scaler.fit_transform(X_train)
	X_test = min_max_scaler.fit_transform(X_test)

	inputs = X_train.shape[1]
	hiddens = best_H[dataset]
	outputs = len(np.unique(y_train))

	NeuralNet_classifier = NeuralNet(inputs, hiddens, outputs, 'random')

	NeuralNet_classifier.train(X_train, y_train, 0.01, 100)
	print('\n')

	evaluation_test = NeuralNet_classifier.evaluate(X_test, y_test)

	print('Dataset : {} '.format(dataset),'\n')
	print("\n Sur les données test : \n")
	print('Notre Modèle avec les hyperparamètres optimales: \n\n')
	print('Matrice de confusion : \n {}'.format(evaluation_test['confusion_matrix']))
	print('Accuracy : {} \n'.format(evaluation_test['accuracy']))
	classes = list(range(evaluation_test['confusion_matrix'].shape[0]))
	precision = list(evaluation_test['precision'].values())
	recall = list(evaluation_test['recall'].values())
	f1_score = list(evaluation_test['f1_score'].values())
	for i in range(len(precision)):
		print('\n precision de la classe ', i , ' est: ', precision[i])
	print('\n')
	for i in range(len(recall)):
		print('\n recall de la classe ', i , ' est: ', recall[i])
	print('\n')
	for i in range(len(f1_score)):
		print('\n f1_score de la classe ', i , ' est: ', f1_score[i])
	print('\n')