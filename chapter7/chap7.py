#!/usr/bin/python
#-*- encoding: utf-8 -*-
"""
Copyright (c) 2015 @myuuuuun
https://github.com/myuuuuun/NumericalCalculation

This software is released under the MIT License.
"""
from __future__ import division, print_function
import math
import numpy as np
import scipy as sc
import scipy.linalg as scl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ti
EPSIRON = 1.0e-8


def ldl_decomposition(matrix):
    p, L, U = scl.lu(matrix)
    size = matrix.shape[0]
    D = np.zeros((size, size), dtype=np.float64)
    divisor = np.ones(size, dtype=np.float64)
    for i in range(size):
        D[i][i] = U[i][i]
        divisor = U[i][i]
        U[i] /= divisor

    return L, D, U

    
def power_method(matrix, initial_vector, repeat):
    if repeat == 0:
        y = matrix.dot(initial_vector).flatten()
        x = initial_vector.flatten()
        argmax_x = np.argmax(np.abs(x))
        return y[argmax_x] / x[argmax_x]

    y = matrix.dot(initial_vector)
    x = y / np.linalg.norm(y)
    print("*"*30)
    print(repeat, "回目")
    print(y)
    print(x)
    return power_method(matrix, x, repeat-1)


def power_method_rayleigh(matrix, initial_vector, repeat):
    if repeat == 0:
        y = matrix.dot(initial_vector).flatten()
        x = initial_vector.flatten()
        x_t = x.transpose()
        return x_t.dot(matrix).dot(x) / x_t.dot(x)

    y = matrix.dot(initial_vector)
    x = y / np.linalg.norm(y)
    return power_method_rayleigh(matrix, x, repeat-1)


def inverse_iteration(approx, matrix, initial_vector=None, repeat=2):
    size = matrix.shape[0]
    shifted_matrix = matrix - np.identity(size) * approx

    if initial_vector is None:
        initial_vector = np.ones((size, 1))

    for i in range(repeat):
        y = np.linalg.solve(shifted_matrix, initial_vector)
        initial_vector = y / np.linalg.norm(y)

    y = y.flatten()
    x = initial_vector.flatten()
    argmax_x = np.argmax(np.abs(x))
    print(x[argmax_x])
    print(y[argmax_x])
    return approx + (x[argmax_x] / y[argmax_x])


if __name__ == '__main__':
    """
    A = np.array([[5, 1, -1], [2, 4, -2], [1, -1, 3]])
    x = np.array([[1], [1], [1]])
    """
    A = np.array([[2, 1, 0, 0],
                  [1, 2, 1, 0],
                  [0, 1, 2, 1],
                  [0, 0, 1, 2]
                ], dtype=np.float64)

    x = np.array([[1, 1, 1, 1]], dtype=np.float64).transpose()
    """
    a = power_method(A, x, 3)
    b = power_method_rayleigh(A, x, 3)
    print(a)
    print(b)
    """

    #c = inverse_iteration(3.61765, A, x)
    #print(c)
    L, D, U = ldl_decomposition(A)

    print(A)
    print(L)
    print(D)
    print(U)

    print(L.dot(D).dot(U))

