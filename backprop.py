#-*- coding: utf-8 -*-
import numpy as np
import random
from mnist import MNIST

np.random.seed(1) # pour que l'exécution soit déterministe

# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie

N, D_in, D_h, D_out = 30, 2, 10, 3
lr = 0.01 # learning rate

def forward(X, W, b):
	I = X.dot(W)+b
	O = 1/(1+np.exp(-I))
	print X.shape
	print I.shape
	print O.shape
	print W.shape
	return O

def backward(X, Y, e, W, b, lr):
	new_W = W + (lr*Y*(1-Y)*e) * X
	new_b = b + (lr*Y*(1-Y)*e)
	new_e = e*W
	return new_W, new_b, new_e

if __name__=='__main__':

	# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
	X = np.random.random((N, D_in))
	Y = np.random.random((N, D_out))

	# Initialisation aléatoire des poids du réseau
	W1 = 2 * np.random.random((D_in, D_h)) - 1
	b1 = np.zeros((1,D_h))
	W2 = 2 * np.random.random((D_h, D_out)) - 1
	b2 = np.zeros((1,D_out))

	# Passe avant : calcul de la sortie prédite Y_pred #

	# I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
	# O1 = 1/(1+np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)

	O1 = forward(X, W1, b1)

	# I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
	# O2  =  1/(1+np.exp(-I2))  #  Sortie  de  la  couche  de  sortie  (fonction  d'activation  de  type sigmoïde)
	# Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie

	Y_pred = forward(O1, W2, b2)

	# Calcul et affichage de la fonction perte de type MSE #
	loss = np.square(Y_pred -Y).sum() / 2
	print(loss)

	# Passe arrière

	W2, b2, e1 = backward(O1, Y_pred, loss, W2, b2, lr)

	W1, b1, e0 = backward(X, O1, e1, W1, b1, lr)


	

