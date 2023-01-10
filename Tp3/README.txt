Dans la partie KNN dans le code K=5, si vous voulez implémenter la validation croisée vous pouvez enlever le commentaire dans la ligne 40 (#K = range(1, 24, 3)), cela parce que le code prend un grand temps pour trouver les k optimaux de chaque jeu de donnée.

Exemple d'output:

le meilleur k pour IRIS est :  10
le meilleur k pour Wine est :  1
le meilleur k pour Abalone est :  19
********* TRAINING WITH KNN *********
I-1-2) Les métriques avec notre modèle KNN avec les données de test IRIS:
la matrice de confusion est: 
  [[16.  0.  0.]
 [ 0. 14.  0.]
 [ 0.  2. 13.]]
------- l'accuracy du modèle est:  95.56 %
+++++++ Precision de la classe  0  est:  100.0 %
+++++++ Precision de la classe  1  est:  87.5 %
+++++++ Precision de la classe  2  est:  100.0 %
******* Recall de la classe  0  est:  100.0 %
******* Recall de la classe  1  est:  100.0 %
******* Recall de la classe  2  est:  86.67 %
####### f1 score de la classe  0  est:  100.0 %
####### f1 score de la classe  1  est:  93.33 %
####### f1 score de la classe  2  est:  92.86 %
temps d'execution de KNN sur Iris est:  1.71661376953125e-05  S
I) Scikit-learn KNN pour IRIS: 

Matrice de confusion : 
 [[16  0  0]
 [ 0 14  0]
 [ 0  2 13]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       0.88      1.00      0.93        14
           2       1.00      0.87      0.93        15

    accuracy                           0.96        45
   macro avg       0.96      0.96      0.95        45
weighted avg       0.96      0.96      0.96        45

I-2-2) Les métriques avec notre modèle KNN avec les données de test WINE:
la matrice de confusion est: 
  [[415.  81.]
 [ 74. 241.]]
------- l'accuracy du modèle est:  80.89 %
+++++++ Precision de la classe  0  est:  84.87 %
+++++++ Precision de la classe  1  est:  74.84 %
******* Recall de la classe  0  est:  83.67 %
******* Recall de la classe  1  est:  76.51 %
####### f1 score de la classe  0  est:  84.26 %
####### f1 score de la classe  1  est:  75.67 %
temps d'execution de KNN sur Wine est:  1.9073486328125e-06  S
I) Scikit-learn KNN pour Wine: 

Matrice de confusion : 
 [[415  81]
 [ 74 241]]
              precision    recall  f1-score   support

           0       0.85      0.84      0.84       496
           1       0.75      0.77      0.76       315

    accuracy                           0.81       811
   macro avg       0.80      0.80      0.80       811
weighted avg       0.81      0.81      0.81       811

I-3-2) Les métriques avec notre modèle KNN avec les données de test ABALONES:
la matrice de confusion est: 
  [[ 67.  59.   0.]
 [ 25. 952.   6.]
 [  0. 128.  17.]]
------- l'accuracy du modèle est:  82.62 %
+++++++ Precision de la classe  0  est:  72.83 %
+++++++ Precision de la classe  1  est:  83.58 %
+++++++ Precision de la classe  2  est:  73.91 %
******* Recall de la classe  0  est:  53.17 %
******* Recall de la classe  1  est:  96.85 %
******* Recall de la classe  2  est:  11.72 %
####### f1 score de la classe  0  est:  61.47 %
####### f1 score de la classe  1  est:  89.73 %
####### f1 score de la classe  2  est:  20.24 %
temps d'execution de KNN sur Abalones est:  1.9073486328125e-06  S
I) Scikit-learn KNN pour ABALONES: 

Matrice de confusion : 
 [[ 67  59   0]
 [ 25 952   6]
 [  0 128  17]]
              precision    recall  f1-score   support

           0       0.73      0.53      0.61       126
           1       0.84      0.97      0.90       983
           2       0.74      0.12      0.20       145

    accuracy                           0.83      1254
   macro avg       0.77      0.54      0.57      1254
weighted avg       0.81      0.83      0.79      1254

********* TRAINING WITH BAYES NAIF *********
I-1-2) Les métriques avec notre modèle bayes naif avec les données de test IRIS:
la matrice de confusion est: 
  [[16.  0.  0.]
 [ 0. 13.  1.]
 [ 0.  0. 15.]]
------- l'accuracy du modèle est:  97.78 %
+++++++ Precision de la classe  0  est:  100.0 %
+++++++ Precision de la classe  1  est:  100.0 %
+++++++ Precision de la classe  2  est:  93.75 %
******* Recall de la classe  0  est:  100.0 %
******* Recall de la classe  1  est:  92.86 %
******* Recall de la classe  2  est:  100.0 %
####### f1 score de la classe  0  est:  100.0 %
####### f1 score de la classe  1  est:  96.3 %
####### f1 score de la classe  2  est:  96.77 %
temps d'execution de BAYES NAIF sur Iris est:  0.0009028911590576172  S
I) Scikit-learn BAYES NAIF pour IRIS: 

Matrice de confusion : 
 [[16  0  0]
 [ 0 13  1]
 [ 0  0 15]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        16
           1       1.00      0.93      0.96        14
           2       0.94      1.00      0.97        15

    accuracy                           0.98        45
   macro avg       0.98      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45

I-2-2) Les métriques avec notre modèle bayes naif avec les données de test WINE:
la matrice de confusion est: 
  [[395. 101.]
 [ 42. 273.]]
------- l'accuracy du modèle est:  82.37 %
+++++++ Precision de la classe  0  est:  90.39 %
+++++++ Precision de la classe  1  est:  72.99 %
******* Recall de la classe  0  est:  79.64 %
******* Recall de la classe  1  est:  86.67 %
####### f1 score de la classe  0  est:  84.67 %
####### f1 score de la classe  1  est:  79.25 %
temps d'execution de BAYES NAIF sur Wine est:  0.0011990070343017578  S
I) Scikit-learn BAYES NAIF pour Wine: 

Matrice de confusion : 
 [[393 103]
 [ 42 273]]
              precision    recall  f1-score   support

           0       0.90      0.79      0.84       496
           1       0.73      0.87      0.79       315

    accuracy                           0.82       811
   macro avg       0.81      0.83      0.82       811
weighted avg       0.83      0.82      0.82       811

I-3-2) Les métriques avec notre modèle bayes naif avec les données de test ABALONES:
la matrice de confusion est: 
  [[120.   6.   0.]
 [163. 521. 299.]
 [  3.  73.  69.]]
------- l'accuracy du modèle est:  56.62 %
+++++++ Precision de la classe  0  est:  41.96 %
+++++++ Precision de la classe  1  est:  86.83 %
+++++++ Precision de la classe  2  est:  18.75 %
******* Recall de la classe  0  est:  95.24 %
******* Recall de la classe  1  est:  53.0 %
******* Recall de la classe  2  est:  47.59 %
####### f1 score de la classe  0  est:  58.25 %
####### f1 score de la classe  1  est:  65.82 %
####### f1 score de la classe  2  est:  26.9 %
temps d'execution de BAYES NAIF sur Abalones est:  0.0008690357208251953  S
I) Scikit-learn BAYES NAIF pour ABALONES: 

Matrice de confusion : 
 [[120   6   0]
 [163 521 299]
 [  3  73  69]]
              precision    recall  f1-score   support

           0       0.42      0.95      0.58       126
           1       0.87      0.53      0.66       983
           2       0.19      0.48      0.27       145

    accuracy                           0.57      1254
   macro avg       0.49      0.65      0.50      1254
weighted avg       0.74      0.57      0.61      1254