#-*- coding: utf-8 -*-
import numpy as np
import random
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

# from Crypto.Util.number import size
from mnist import MNIST

def lecture_mnist(chemin):

	mdata = MNIST(chemin)

	DataApp, LabelApp = mdata.load_training()
	DataTest, LabelTest = mdata.load_testing()

	Xapp = np.array(DataApp, dtype=np.float32)
	Yapp = np.array(LabelApp, dtype=np.float32)
	Xtest = np.array(DataTest, dtype=np.float32)
	Ytest = np.array(LabelTest, dtype=np.float32)

	return Xapp, Yapp, Xtest, Ytest

def decoupage_donnees(X, Y):
	ratio = 0.8

	indices = range(X.shape[0])
	random.shuffle(indices)
	nb_app = int(X.shape[0]*ratio)

	Xapp = [X[indices[i]] for i in range(nb_app)]
	Yapp = [Y[indices[i]] for i in range(nb_app)]

	Xtest = [X[indices[i]] for i in range(nb_app, X.shape[0])]
	Ytest = [Y[indices[i]] for i in range(nb_app, X.shape[0])]

	return Xapp, Yapp, Xtest, Ytest

def kppv_distances(Xtest, Xapp):

	Mtest2 = np.matrix([np.sum(np.square(Xtest), axis=1)]*Xapp.shape[0]).T
	Mapp2 = np.matrix([np.sum(np.square(Xapp), axis=1)]*Xtest.shape[0])

	MtestMapp = np.matrix(Xtest)*np.matrix(Xapp).T

	return np.array(Mtest2 + Mapp2 - 2*MtestMapp)

def kppv_predict(Dist, Yapp, K):
	Ypred=[]
	for i in range(Dist.shape[0]):
		numeros = sorted(range(len(Dist[i])), key=lambda k: Dist[i][k])
		predicts = [Yapp[numeros[k]] for k in range(K)]
		best_predict = max(set(predicts), key=predicts.count)
		Ypred.append(best_predict)
	return Ypred

def evaluation_classifieur(Ypred, Ytest):
	return [Ypred[i]==Ytest[i] for i in range(len(Ypred))].count(True)/float(len(Ypred))*100

def moyenne(tableau):
    return sum(tableau, 0.0) / len(tableau)

def variance(tableau):
    m=moyenne(tableau)
    return moyenne([(x-m)**2 for x in tableau])

def ecartype(tableau):
    return variance(tableau)**0.5

def cross_validation(n, X, Y, K):

	indices=range(X.shape[0])
	random.shuffle(indices)

	size_fold=X.shape[0]/n

	X=[X[i] for i in indices]
	Y=[Y[i] for i in indices]

	X_folds=[]
	Y_folds=[]
	evals=[]

	for i in range(n):
		X_folds.append([X[k+i*size_fold] for k in range(size_fold)])
		Y_folds.append([Y[k+i*size_fold] for k in range(size_fold)])

	for i in tqdm(range(n)):
		Xapp=[]
		Yapp=[]

		for k in range(1, n):
			Xapp+=X_folds[(i+k)%n]
			Yapp+=Y_folds[(i+k)%n]

		dist = kppv_distances(np.array(X_folds[i]), np.array(Xapp))
		Ypred = kppv_predict(dist, np.array(Yapp), K)
		evals.append(evaluation_classifieur(Ypred, Y_folds[i]))

	print ("perf moyenne : "+str(moyenne(evals)))
	print ("ecart-type : "+str(ecartype(evals)))


def computeHOG(image):
	# image = image[:, :, 0]
	# print(image.shape)
	hog_image = hog(image, orientations=9, pixels_per_cell=(4,4),
					cells_per_block=(2, 2))
	return hog_image

if __name__=='__main__':

	print('Loading MNIST')
	Xapp, Yapp, Xtest, Ytest = lecture_mnist('MNIST-data')
	print('Done')
	data=Xapp[:18000]
	labels=Yapp[:18000]

	if(True):
		data2=[]
		for ii in tqdm(range(len(data)),desc='HOG Computation'):
			data2.append(np.asarray(computeHOG(data[ii].reshape((28,28)))))

		data=np.asarray(data2)

	cross_validation(60, data, labels, 3)

