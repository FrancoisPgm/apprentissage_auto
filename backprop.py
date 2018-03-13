#-*- coding: utf-8 -*-
import numpy as np
import random
# from mnist import MNIST

np.random.seed(1) # pour que l'exécution soit déterministe

# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie

N, D_in, D_h, D_out = 30, 2, 10, 3
lr = 0.001 # learning rate

def forward(X, W, b):
	I = X.dot(W)+b
	O = 1/(1+np.exp(-I))
	return (O)

def backward(X, Y, e, W, b, lr):
	new_W = W + (lr*Y*(1-Y)*e) * X.T
	new_b = b + (lr*Y*(1-Y)*e)
	new_e = e*W
	return new_W, new_b, new_e

if __name__=='__main__':

	# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
	X = np.random.random((N, D_in))
	Y = np.random.random((N, D_out))

	# Initialisation aléatoire des poids du réseau
	W1 = 2 * np.random.random((D_in, D_h)) - 1
	b1 = np.zeros((1, D_h))
	W2 = 2 * np.random.random((D_h, D_out)) - 1
	b2 = np.zeros((1, D_out))

	for ii in range (100000):
		# Passe avant : calcul de la sortie prédite Y_pred #

		I1 = X.dot(W1) + b1 # Potentiel d'entrée de la couche cachée
		O1 = 1/(1+np.exp(-I1)) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)


		I2 = O1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
		O2  =  1/(1+np.exp(-I2))  #  Sortie  de  la  couche  de  sortie  (fonction  d'activation  de  type sigmoïde)
		Y_pred = O2 # Les valeurs prédites sont les sorties de la couche de sortie

		# Calcul et affichage de la fonction perte de type MSE #
		loss = np.square(Y_pred -Y).sum() / 2
		if not ii%25000:
			print(loss)

		l2_error=Y-Y_pred
		l2_delta=l2_error*(Y_pred*(1-Y_pred))

		# print(l2_delta.shape)
		l1_error=l2_delta.dot(W2.T)
		l1_delta=l1_error*(O1* (1 - O1))

		W2 += lr * O1.T.dot(l2_delta)
		b2 += lr * l2_delta.sum(axis=0)
		O1 += lr * l2_delta.dot(W2.T)

		b1 += lr * l1_delta.sum(axis=0)
		W1 += lr * X.T.dot(l1_delta)



		# b2+=lr*O1.T.dot(l2_delta)
		# W2+=lr*O1.T.dot(l2_delta)





		# new_b2=b2+lr*(Y_pred*(1-Y_pred))*loss
		# new_W2=W2+lr*O1.T.dot(Y_pred*(1-Y_pred)*loss)
		# new_O1=O1+lr*(Y_pred*(1-Y_pred)*loss).dot(W2.T)
        #
		# b2 = new_b2
		# W2 = new_W2
		# O1=new_O1
        #
		# new_b1 = b1 + lr * (O1 * (1 - O1)*loss)
		# new_W1 = W1 + lr * loss * X.T.dot(O1 * (1 - O1)*loss)
        #
		# b1=new_b1
		# W1=new_W1





