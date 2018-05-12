# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses


# Computes the Aii = a(i) distance weight matrix
def compute_distance_weight_vec(test_datum, tau, N, X):
    A_norm = -l2(test_datum, X)
    Ai = (A_norm / (2 * (tau**2)))
    sum_Aj = logsumexp(Ai)
    A = np.exp(Ai - sum_Aj) * np.identity(N)

    return A


#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    N = len(x_train)
    I = np.identity(d)
    A = compute_distance_weight_vec(test_datum, tau, N, x_train)
    lam_I = lam * I
    X_trans = np.transpose(x_train)
    XTA = np.matmul(X_trans, A)
    XTAX = np.matmul(XTA, x_train)

    XTAy = np.matmul(XTA, y_train)

    # w*((XTAX + λI)^-1) = (XTAy)
    w_opt = np.linalg.solve(XTAX + lam_I, XTAy)

    # Apply our optimal weights to our test vector, yielding out result
    y_hat = np.matmul(w_opt, test_datum[0])
    return y_hat


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''

    losses_arr = []
    batch_size = N//k
    for index in range(k):
        start_index = index * batch_size
        # Choose a block of size N/<k> from the training data
        x_test = x[start_index: start_index + batch_size]
        y_test = y[start_index: start_index + batch_size]

        # Extract training data excluding previously defined test data
        x_train = np.concatenate([
            x[0: start_index],
            x[start_index + batch_size:]
        ])
        y_train = np.concatenate([
            y[0: start_index],
            y[start_index + batch_size:]
        ])

        losses = run_on_fold(x_test, y_test, x_train, y_train, taus)
        losses_arr.append(losses)

    # Return the column-wise average of our losses
    return np.average(losses_arr, axis=0)


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.figure(figsize=(15, 5))
    plt.plot(taus, losses)
    plt.xlabel('τ')
    plt.ylabel('average loss')
    plt.tight_layout()
    plt.show()
    print(losses)
    print("min loss = {}".format(losses.min()))
