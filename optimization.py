#-*- encoding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(precision=6)
EPS = 1.0e-12
EPS2 = 1.0e-6

# 黄金分割法
# minimize optimization for 1 variable unimodal function (in given interval).
def golden_section(func, lower_b=None, upper_b=None, **kwargs):
    if lower_b is None:
        lower_b = -1 * pow(2, 30)

    if upper_b is None:
        upper_b = pow(2, 30)

    if upper_b - lower_b < 0:
        raise ValueError("Empty interval.")

    loop = kwargs.get("loop", 1000)
    epsilon = kwargs.get("epsilon", EPS)

    f1 = (3 - math.sqrt(5)) / 2
    f2 = (math.sqrt(5) - 1) / 2
    interval = upper_b - lower_b
    x_lower = lower_b + f1 * interval
    x_upper = lower_b + f2 * interval

    for i in range(loop):
        f_lower = func(x_lower)
        f_upper = func(x_upper)
        if f_lower == float('Inf') or f_upper == float('Inf'):
            raise ValueError("function value is not bounded below in the interval.")

        if f_lower < f_upper:
            upper_b = x_upper
            x_upper = x_lower
            interval = upper_b - lower_b
            x_lower = lower_b + f1 * interval

        else:
            lower_b = x_lower
            x_lower = x_upper
            interval = upper_b - lower_b
            x_upper = lower_b + f2 * interval

        if math.fabs(interval) < epsilon:
            return (x_lower+x_upper)/2, (f_lower+f_upper)/2

    raise ValueError("Algorithm did not finish in {0} loops.".format(loop))


# 最急降下法
# minimize optimization
def steepest_descent(grad_func, start, step=0.1, **kwargs):
    loop = kwargs.get("loop", 1000)
    epsilon = kwargs.get("epsilon", EPS2)
    start = np.asarray(start)

    for i in range(loop):
        grad = np.asarray(grad_func(start))
        new = start - step * grad
        if np.linalg.norm(start - new) < epsilon:
            print(i)
            return new
        start = new

    raise ValueError("Algorithm did not finish in {0} loops.".format(loop))


# 最適勾配法
# minimize optimization
def optimal_grad(func, grad_func, start, **kwargs):
    loop = kwargs.get("loop", 1000)
    epsilon = kwargs.get("epsilon", 0.0000001)
    start = np.asarray(start)

    for i in range(loop):
        grad = np.asarray(grad_func(start))
        linear_search = lambda step: func(start + step * grad)
        step = golden_section(linear_search, -10, 10)[0]
        new = start + step * grad
        if np.linalg.norm(start - new) < epsilon:
            print(i)
            return new
        start = new

    raise ValueError("Algorithm did not finish in {0} loops.".format(loop))


if __name__ == '__main__':

    # 最急降下法
    func = lambda x: -1 * (-1 * x[0]**2 + 16*x[0] -2*x[1]**2 + 16*x[1])
    grad = lambda x: np.array([2*x[0] - 16, 4*x[1] - 16])

    start = [2.0, 1.0]
    step = 0.25

    result1 = steepest_descent(grad, start, step)
    print("最急降下法", result1, -func(result1))

    result2 = optimal_grad(func, grad, start)
    print("最適勾配法", result2, -func(result2))

    """
    # 黄金分割法
    func = lambda x: pow(x, 2) + 10
    result = golden_section(func, loop=1000, epsilon=0.001)
    print( "x={0:.5f} のとき minf = {1:.5f}".format(result[0], result[1]) )
    """





