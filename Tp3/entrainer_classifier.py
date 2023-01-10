import load_datasets
from sklearn import metrics
#import NaiveBayes # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
from sklearn.naive_bayes import GaussianNB
import NaiveBayes


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les paramètres que vous avez utilisés
En gros, vous allez :
1- Initialiser votre classifieur avec ses paramètres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""
def cross_validation_split(data, folds=10):
    if len(data) < folds:
        return None
    index_split = []
    data_copy = data.tolist()
    fold_size = int(len(data) / folds)
    intervalle = list(range(len(data_copy)))
    for i in range(folds):
        fold = []
        while len(fold) < fold_size:
            index = np.random.choice(intervalle)
            fold.append(index)
            intervalle.remove(index)
        index_split.append(fold)
    return index_split

# Initialiser les paramètres
#K = range(1, 24, 3)
K = [5]

# Charger/lire les datasets 
iris = load_datasets.load_iris_dataset(0.7)
X_train_i = iris[0]
y_train_i = iris[1]
X_test_i = iris[2]
y_test_i = iris[3]
indexes_iris = cross_validation_split(X_train_i, 10)

wine = load_datasets.load_wine_dataset(0.7)
X_train_w = wine[0]
y_train_w = wine[1]
X_test_w = wine[2]
y_test_w = wine[3]
indexes_wine = cross_validation_split(X_train_w, 10)

abalones = load_datasets.load_abalone_dataset(0.7)
X_train_a = abalones[0]
y_train_a = abalones[1]
X_test_a = abalones[2]
y_test_a = abalones[3]
indexes_abalone = cross_validation_split(X_train_a, 10)

# choisir le meilleur K pour IRIS
scores = []
X_axe = []
Y_axe = []
for k in K:
    scores_kfold = []
    for index in indexes_iris:
        train_indexes = [y for x in indexes_iris for y in x if x!=index]
        X_train = X_train_i[train_indexes]
        y_train = y_train_i[train_indexes]
        X_valid = X_train_i[index]
        y_valid = y_train_i[index]
        knn_model = Knn.Knn(k)
        knn_model.train(X_train, y_train)
        metric = knn_model.evaluate(X_valid, y_valid)
        scores_kfold.append(metric['accuracy'])
    X_axe.append(k)
    Y_axe.append(np.mean(scores_kfold))
    scores.append((k, np.mean(scores_kfold)))
best_k_iris = max(scores,key=lambda item:item[1])[0]
print('le meilleur k pour IRIS est : ', best_k_iris)
#plt.plot(X_axe, Y_axe)
#plt.xlabel('Nombre de K')
#plt.ylabel('Score moyen par Kfold')
#plt.show()

# choisir le meilleur K pour Wine
scores = []
X_axe = []
Y_axe = []
for k in K:
    scores_kfold = []
    for index in indexes_wine:
        train_indexes = [y for x in indexes_wine for y in x if x!=index]
        X_train = X_train_w[train_indexes]
        y_train = y_train_w[train_indexes]
        X_valid = X_train_w[index]
        y_valid = y_train_w[index]
        knn_model = Knn.Knn(k)
        knn_model.train(X_train, y_train)
        predictions = []
        for item in X_valid:
            predictions.append(knn_model.predict(item))
        accuracy = round(((predictions == y_valid).sum() / len(y_valid))*100, 2 )
        scores_kfold.append(accuracy)
    X_axe.append(k)
    Y_axe.append(np.mean(scores_kfold))
    scores.append((k, np.mean(scores_kfold)))
best_k_wine = max(scores,key=lambda item:item[1])[0]
print('le meilleur k pour Wine est : ', best_k_wine)
#plt.plot(X_axe, Y_axe)
#plt.xlabel('Nombre de K')
#plt.ylabel('Score moyen par Kfold')
#plt.show()

# choisir le meilleur K pour Abalones
scores = []
X_axe = []
Y_axe = []
for k in K:
    scores_kfold = []
    for index in indexes_abalone:
        train_indexes = [y for x in indexes_abalone for y in x if x!=index]
        X_train = X_train_a[train_indexes]
        y_train = y_train_a[train_indexes]
        X_valid = X_train_a[index]
        y_valid = y_train_a[index]
        knn_model = Knn.Knn(k)
        knn_model.train(X_train, y_train)
        predictions = []
        for item in X_valid:
            predictions.append(knn_model.predict(item))
        accuracy = round(((predictions == y_valid).sum() / len(y_valid))*100, 2 )
        scores_kfold.append(accuracy)
    X_axe.append(k)
    Y_axe.append(np.mean(scores_kfold))
    scores.append((k, np.mean(scores_kfold)))
best_k_abalones = max(scores,key=lambda item:item[1])[0]
print('le meilleur k pour Abalone est : ', best_k_abalones)
#plt.plot(X_axe, Y_axe)
#plt.xlabel('Nombre de K')
#plt.ylabel('Score moyen par Kfold')
#plt.show()



print('********* TRAINING WITH KNN *********')

# Entrainez votre classifieur knn sur les donnees IRIS
knn_model_iris = Knn.Knn(best_k_iris)
t1_iris = time.time()
knn_model_iris.train(X_train_i, y_train_i)
t2_iris = time.time()
t_total_iris = t2_iris - t1_iris

    #Evaluation du modèle sur test
print('I-1-2) Les métriques avec notre modèle KNN avec les données de test IRIS:')
metrics_iris_test = knn_model_iris.evaluate(X_test_i, y_test_i)
#matrice de confusion
print("la matrice de confusion est: \n ", metrics_iris_test['matrice_confusion'])
#accuracy
print("------- l'accuracy du modèle est: ", metrics_iris_test["accuracy"], '%')
#precision
for i in range(len(metrics_iris_test["precision"])):
    print("+++++++ Precision de la classe ", i ," est: ", round(metrics_iris_test["precision"][i]*100, 2), '%')
#recall
for i in range(len(metrics_iris_test["recall"])):
    print("******* Recall de la classe ", i ," est: ", round(metrics_iris_test["recall"][i]*100, 2), '%')
#f1_score
for i in range(len(metrics_iris_test["f_score"])):
    print("####### f1 score de la classe ", i ," est: ", round(metrics_iris_test["f_score"][i]*100, 2), '%')
print("temps d'execution de KNN sur Iris est: ", t_total_iris ,' S')

# Sklearn KNN iris 
model_iris = KNeighborsClassifier(n_neighbors=best_k_iris)
model_iris.fit(X_train_i, y_train_i)
y_pred = model_iris.predict(X_test_i)
print('I) Scikit-learn KNN pour IRIS: \n')
print('Matrice de confusion : \n {}'.format(metrics.confusion_matrix(y_test_i, y_pred)))
print(metrics.classification_report(y_test_i , y_pred))



# Entrainez votre classifieur knn sur les donnees WINE
knn_model_wine = Knn.Knn(best_k_wine)
t1_wine = time.time()
knn_model_wine.train(X_train_w, y_train_w)
t2_wine = time.time()

t_total_wine = t2_wine - t1_wine

    #Evaluation du modèle sur test
metrics_wine_test = knn_model_wine.evaluate(X_test_w, y_test_w)
print('I-2-2) Les métriques avec notre modèle KNN avec les données de test WINE:')
#matrice de confusion
print("la matrice de confusion est: \n ", metrics_wine_test['matrice_confusion'])
#accuracy
print("------- l'accuracy du modèle est: ", metrics_wine_test["accuracy"], '%')
#precision
for i in range(len(metrics_wine_test["precision"])):
    print("+++++++ Precision de la classe ", i ," est: ", round(metrics_wine_test["precision"][i]*100, 2), '%')
#recall
for i in range(len(metrics_wine_test["recall"])):
    print("******* Recall de la classe ", i ," est: ", round(metrics_wine_test["recall"][i]*100, 2), '%')
#f1_score
for i in range(len(metrics_wine_test["f_score"])):
    print("####### f1 score de la classe ", i ," est: ", round(metrics_wine_test["f_score"][i]*100, 2), '%')
print("temps d'execution de KNN sur Wine est: ", t_total_wine ,' S')

# Sklearn KNN Wine
model_wine = KNeighborsClassifier(n_neighbors=best_k_wine)
model_wine.fit(X_train_w, y_train_w)
y_pred = model_wine.predict(X_test_w)
print('I) Scikit-learn KNN pour Wine: \n')
print('Matrice de confusion : \n {}'.format(metrics.confusion_matrix(y_test_w, y_pred)))
print(metrics.classification_report(y_test_w , y_pred))



# Entrainez votre classifieur knn sur les donnees ABALONES
knn_model_abalones = Knn.Knn(best_k_abalones)
t1_abalones = time.time()
knn_model_abalones.train(X_train_a, y_train_a)
t2_abalones = time.time()

t_total_abalones = t2_abalones - t1_abalones

    #Evaluation du modèle sur test
metrics_abalones_test = knn_model_abalones.evaluate(X_test_a, y_test_a)
print('I-3-2) Les métriques avec notre modèle KNN avec les données de test ABALONES:')
#matrice de confusion
print("la matrice de confusion est: \n ", metrics_abalones_test['matrice_confusion'])
#accuracy
print("------- l'accuracy du modèle est: ", metrics_abalones_test["accuracy"], '%')
#precision
for i in range(len(metrics_abalones_test["precision"])):
    print("+++++++ Precision de la classe ", i ," est: ", round(metrics_abalones_test["precision"][i]*100, 2), '%')
#recall
for i in range(len(metrics_abalones_test["recall"])):
    print("******* Recall de la classe ", i ," est: ", round(metrics_abalones_test["recall"][i]*100, 2), '%')
#f1_score
for i in range(len(metrics_abalones_test["f_score"])):
    print("####### f1 score de la classe ", i ," est: ", round(metrics_abalones_test["f_score"][i]*100, 2), '%')

print("temps d'execution de KNN sur Abalones est: ", t_total_abalones ,' S')

# Sklearn KNN Abalones

model_abalones = KNeighborsClassifier(n_neighbors=best_k_abalones)
model_abalones.fit(X_train_a, y_train_a)
y_pred = model_abalones.predict(X_test_a)
print('I) Scikit-learn KNN pour ABALONES: \n')
print('Matrice de confusion : \n {}'.format(metrics.confusion_matrix(y_test_a, y_pred)))
print(metrics.classification_report(y_test_a , y_pred))



# Entrainez votre classifieur bayes naif sur les donnees IRIS
bayes_model = NaiveBayes.BayesNaif()
print('********* TRAINING WITH BAYES NAIF *********')
t1_iris = time.time()
bayes_model.train(X_train_i, y_train_i)
t2_iris = time.time()

t_total_iris = t2_iris - t1_iris

    #Evaluation du modèle sur test
print('I-1-2) Les métriques avec notre modèle bayes naif avec les données de test IRIS:')
metrics_iris_test = bayes_model.evaluate(X_test_i, y_test_i)
#matrice de confusion
print("la matrice de confusion est: \n ", metrics_iris_test['matrice_confusion'])
#accuracy
print("------- l'accuracy du modèle est: ", metrics_iris_test["accuracy"], '%')
#precision
for i in range(len(metrics_iris_test["precision"])):
    print("+++++++ Precision de la classe ", i ," est: ", round(metrics_iris_test["precision"][i]*100, 2), '%')
#recall
for i in range(len(metrics_iris_test["recall"])):
    print("******* Recall de la classe ", i ," est: ", round(metrics_iris_test["recall"][i]*100, 2), '%')
#f1_score
for i in range(len(metrics_iris_test["f_score"])):
    print("####### f1 score de la classe ", i ," est: ", round(metrics_iris_test["f_score"][i]*100, 2), '%')
print("temps d'execution de BAYES NAIF sur Iris est: ", t_total_iris ,' S')

#Sklearn BN Iris
model_iris = GaussianNB()
model_iris.fit(X_train_i, y_train_i)
y_pred = model_iris.predict(X_test_i)
print('I) Scikit-learn BAYES NAIF pour IRIS: \n')
print('Matrice de confusion : \n {}'.format(metrics.confusion_matrix(y_test_i, y_pred)))
print(metrics.classification_report(y_test_i , y_pred))


# Entrainez votre classifieur bayes naif sur les donnees WINE
t1_wine = time.time()
bayes_model.train(X_train_w, y_train_w)
t2_wine = time.time()

t_total_wine = t2_wine - t1_wine

    #Evaluation du modèle sur test
metrics_wine_test = bayes_model.evaluate(X_test_w, y_test_w)
print('I-2-2) Les métriques avec notre modèle bayes naif avec les données de test WINE:')
#matrice de confusion
print("la matrice de confusion est: \n ", metrics_wine_test['matrice_confusion'])
#accuracy
print("------- l'accuracy du modèle est: ", metrics_wine_test["accuracy"], '%')
#precision
for i in range(len(metrics_wine_test["precision"])):
    print("+++++++ Precision de la classe ", i ," est: ", round(metrics_wine_test["precision"][i]*100, 2), '%')
#recall
for i in range(len(metrics_wine_test["recall"])):
    print("******* Recall de la classe ", i ," est: ", round(metrics_wine_test["recall"][i]*100, 2), '%')
#f1_score
for i in range(len(metrics_wine_test["f_score"])):
    print("####### f1 score de la classe ", i ," est: ", round(metrics_wine_test["f_score"][i]*100, 2), '%')
print("temps d'execution de BAYES NAIF sur Wine est: ", t_total_wine ,' S')

#Sklearn BN Wine
model_wine = GaussianNB()
model_wine.fit(X_train_w, y_train_w)
y_pred = model_wine.predict(X_test_w)
print('I) Scikit-learn BAYES NAIF pour Wine: \n')
print('Matrice de confusion : \n {}'.format(metrics.confusion_matrix(y_test_w, y_pred)))
print(metrics.classification_report(y_test_w , y_pred))

# Entrainez votre classifieur bayes naif sur les donnees ABALONES
t1_abalones = time.time()
bayes_model.train(X_train_a, y_train_a)
t2_abalones = time.time()

t_total_abalones = t2_abalones - t1_abalones

    #Evaluation du modèle sur test
metrics_abalones_test = bayes_model.evaluate(X_test_a, y_test_a)
print('I-3-2) Les métriques avec notre modèle bayes naif avec les données de test ABALONES:')
#matrice de confusion
print("la matrice de confusion est: \n ", metrics_abalones_test['matrice_confusion'])
#accuracy
print("------- l'accuracy du modèle est: ", metrics_abalones_test["accuracy"], '%')
#precision
for i in range(len(metrics_abalones_test["precision"])):
    print("+++++++ Precision de la classe ", i ," est: ", round(metrics_abalones_test["precision"][i]*100, 2), '%')
#recall
for i in range(len(metrics_abalones_test["recall"])):
    print("******* Recall de la classe ", i ," est: ", round(metrics_abalones_test["recall"][i]*100, 2), '%')
#f1_score
for i in range(len(metrics_abalones_test["f_score"])):
    print("####### f1 score de la classe ", i ," est: ", round(metrics_abalones_test["f_score"][i]*100, 2), '%')
print("temps d'execution de BAYES NAIF sur Abalones est: ", t_total_abalones ,' S')
#Sklearn BN Abalones
model_abalones = GaussianNB()
model_abalones.fit(X_train_a, y_train_a)
y_pred = model_abalones.predict(X_test_a)
print('I) Scikit-learn BAYES NAIF pour ABALONES: \n')
print('Matrice de confusion : \n {}'.format(metrics.confusion_matrix(y_test_a, y_pred)))
print(metrics.classification_report(y_test_a , y_pred))