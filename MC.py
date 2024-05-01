import numpy as np
from numpy.linalg import eig
import os
import cyclehunter
from cyclehunter import *
from scipy.optimize import minimize
from scipy.linalg import eig
import matplotlib.pyplot as plt
import time

def weight_zeta(n, p, musq) :
    x = p[n] 
    T = len(x)
    Lam = stab(x, musq)
    w = np.absolute(T / (Lam - 1))

    return w

def stab(cycle, musq) :
    ret = 1
    # find the Jacobian matrix for a given cycle and musqr
    Jac = np.array ([[1, 0], [0, 1]])
    for x in cycle :
        tmp = np.array ([[- 3 * musq * (x ** 2) + musq + 2, -1], [1, 0]])
        Jac = np.matmul (tmp, Jac)
        w, v = eig (Jac)
        #find expanding eigenvalue
        for l in np.absolute(w) :
            if l > 1 :
                ret *= l

    return (ret)

# Metropolis-Hastings algorithm for MCMC sampling
def metropolis_hastings_zeta(x0, n_samples, p, musq, result_list, tm):
    samples = [p[x0]]
    x_current = x0
    n = 0
    while n < n_samples :
        # Propose a new sample from a Gaussian distribution centered at the current sample
        x_proposed = np.random.randint(0, len(p))
        
        # Calculate the acceptance probability
        acceptance_prob = min(1, weight_zeta(x_proposed, p, musq) / weight_zeta(x_current, p, musq))
        
        # Accept or reject the proposed sample based on the acceptance probability
        if np.random.rand() < acceptance_prob:
            x_current = x_proposed
        
        samples.append(p[x_current])
        n += 1
    
    stability = -1 * np.log(Hill_det (samples, musq))
    # Expectation of period
    T = np.array([len(x) for x in samples])
    # Calculate the expectation value of x
    expectation_value = np.mean(stability / T)

    # For parallelization purpose
    result_list.append (expectation_value)
    tm.append(np.log(len(p)))
    print (len(result_list))
    
    return expectation_value

# QR decomposition 
def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    det = 1
    
    for i in range(n):
        det *= R[i, i]
    
    #det *= np.linalg.det(Q)
        
    return Q, R, det

# Calculate Hill determinant
def Hill_det (sample, musq) :
    J = []
    for cycle in sample:
        A = [[] for j in range (len(cycle))]

        for j in range (len(cycle)):
            for k in range (len(cycle)):
                if (j == k):
                    A[j].append (musq + 2 - 3 * musq * cycle[j]**2 )
                elif (k == j+1 or k == j-1 or k == j+len(cycle)-1 or k == j-len(cycle)+1):
                    A[j].append (-1)
                else: 
                    A[j].append (0)

        B = np.array(A)
        Q, R, det = qr_decomposition(B)
        c = det
        J.append(c)

    return (np.array(J))

# MCMC simulation with distribution given by partition function
def metropolis_hastings_part(x0, n_samples, A, musq):
    samples = [A[x0]]
    x_current = x0
    n = 0
    while n < n_samples :
        # Propose a new sample from a Gaussian distribution centered at the current sample
        x_proposed = np.random.randint(0, len(A))
        
        # Calculate the acceptance probability
        acceptance_prob = min(1, weight_part(x_proposed, A, musq) / weight_part(x_current, A, musq))
        
        # Accept or reject the proposed sample based on the acceptance probability
        if np.random.rand() < acceptance_prob:
            x_current = x_proposed
        
        samples.append(A[x_current])
        n += 1
    
    return samples

# Weight calculation for partition function (periodic states instead of orbits)
def weight_part (n, A, musq) :
    x = A[n]
    J = Hill_det([x], musq)

    return (1/J)

def take_first_n_terms(dictionary, n):
    # Initialize an empty dictionary to store the result
    new_dict = {}
    
    # Iterate over the original dictionary and add the first n key-value pairs
    for i, (key, value) in enumerate(dictionary.items()):
        if i < n:
            new_dict[key] = value
        else:
            break
    
    return new_dict