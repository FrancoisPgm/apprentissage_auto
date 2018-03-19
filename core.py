import numpy as np
import matplotlib.pyplot as plt


def forwardPass(X, W, b):
    I = X.dot(W)+b
    O = 1/(1+np.exp(-I))
    return (O)

def backwardPass(X, Y, e, W, b, lr):
    new_W = W + (lr*Y*(1-Y)*e) * X.T
    new_b = b + (lr*Y*(1-Y)*e)
    new_e = e*W
    return new_W, new_b, new_e

def printLoss(ii,lLoss,gt,guess):
    if not ii%250:
        lLoss.append(np.square(gt -guess).sum() / 2)
    return lLoss

def sigmoidDiff(Y):
    return Y*(1-Y)

def declareLayer(w,h):
    W = 2 * np.random.random((w,h)) - 1
    b = np.zeros((1, h))
    return W,b

def updateLayer(delta,I,W,b,lr,bFirstLayer):
    new_W = W + lr * I.T.dot(delta)
    b+= lr * delta.sum(axis=0)
    if not bFirstLayer:
        I = lr * delta.dot(W.T)
    return new_W,b,I


def backProgHiddenLayer(sup_delta,sup_W,O):
    error=sup_delta.dot(sup_W.T)
    delta=error*sigmoidDiff(O)
    return error,delta

def backProgLastLayer(Y,Y_pred):
    error=Y-Y_pred
    delta=error*sigmoidDiff(Y_pred)
    return error,delta

def showLearning(lLoss):
    plt.figure()
    plt.plot(lLoss)
    plt.grid(True)
    plt.show()






















