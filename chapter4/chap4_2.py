#!/usr/bin/python
#-*- encoding: utf-8 -*-
"""
Copyright (c) 2015 @myuuuuun
https://github.com/myuuuuun/NumericalCalculation

This software is released under the MIT License.
"""
from __future__ import division, print_function
import math
import functools
import numpy as np
import matplotlib.pyplot as plt
EPSIRON = 1.0e-8


"""
自動微分モジュール（Forward Mode）
"""
# AutoDiffオブジェクト同士で演算が行われているかをチェックするデコレータ
# 演算の引数に定数が与えられた場合、新しく導関数値が0のAutoDiffオブジェクトを作って計算
def valuecheck(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, value in enumerate(args[1:]):
            if value.__class__.__name__ != "AutoDiff":
                if isinstance(value, (int, float)):
                    args[i+1] = AutoDiff(value, 0.0)
                else:
                    raise ValueError("AutoDiffオブジェクト同士で演算してください")

        rst = func(*tuple(args), **kwargs)
        return rst
    return wrapper


class AutoDiff():
    def __init__(self, x, dx):
        self.x = 1.0 * x
        self.dx = 1.0 * dx

    def __repr__(self):
        return "関数値: {0:.5f}, 導関数値: {1:.5f}".format(self.x, self.dx)

    # 関数値: f(x)+g(x), 導関数値: f'(x)+g'(x)
    @valuecheck
    def __add__(self, value):
        x = self.x + value.x
        dx = self.dx + value.dx
        return AutoDiff(x, dx)

    # 関数値: f(x)-g(x), 導関数値: f'(x)-g'(x)
    @valuecheck
    def __sub__(self, value):
        x = self.x - value.x
        dx = self.dx - value.dx
        return AutoDiff(x, dx)

    # 関数値: f(x)*g(x), 導関数値: f'(x)g(x)+f(x)g'(x)
    @valuecheck
    def __mul__(self, value):
        x = self.x * value.x
        dx = self.dx * value.x + self.x * value.dx
        return AutoDiff(x, dx)

    # 関数値: f(x)/g(x), 導関数値: {f'(x)g(x)-f(x)g'(x)} / {g(x)}^2
    @valuecheck
    def __truediv__(self, value):
        x = self.x / value.x
        dx = (self.dx * value.x - self.x * value.dx) / (value.x ** 2)
        return AutoDiff(x, dx)

    # 関数値: sin(f(x)), 導関数値: f'(x)*cos(f(x))
    @classmethod
    @valuecheck
    def sin(cls, value):
        print(value)
        x = np.sin(value.x)
        dx = value.dx * np.cos(value.x)
        return AutoDiff(x, dx)

    # 関数値: cos(f(x)), 導関数値: -f'(x)*sin(f(x))
    @classmethod
    @valuecheck
    def cos(cls, value):
        x = np.cos(value.x)
        dx = value.dx * -1.0 * np.sin(value.x)
        return AutoDiff(x, dx)

    # 関数値: tan(f(x)), 導関数値: f'(x)/{cos(f(x))}^2
    @classmethod
    @valuecheck
    def tan(cls, value):
        x = np.tan(self.x)
        dx = self.dx / (np.cos(self.x) ** 2)
        return AutoDiff(x, dx)

    # 関数値: exp(f(x)), 導関数値: f'(x)*exp(f(x))
    @classmethod
    @valuecheck
    def exp(cls, value):
        x = np.exp(value.x)
        dx = value.dx * value.x
        return AutoDiff(x, dx)

    # 関数値: log(|f(x)|), 導関数値: f'(x)*exp(f(x))
    @classmethod
    @valuecheck
    def log(cls, value):
        if value.x <= 0:
            raise ValueError("logの中身に負の数はとれません")
        x = np.log(value.x)
        dx = value.dx / value.x
        return AutoDiff(x, dx)

    # 関数値: {f(x)}^{g(x)}, 導関数値: {g(x)*logf(x)}' * {f(x)}^{g(x)}
    @classmethod
    @valuecheck
    def pow(self, base, exponent):
        x = math.pow(base.x, exponent.x)
        log_base = AutoDiff.log(base)
        dx = (exponent * log_base).dx * x
        return AutoDiff(x, dx)

    # 関数値: √f(x), 導関数値: 1/(2*√f(x))
    @classmethod
    @valuecheck
    def sqrt(self, value):
        x = math.sqrt(value.x)
        dx = value.dx * 0.5 / math.sqrt(value.x)
        return AutoDiff(x, dx)


if __name__ == '__main__':
    func = lambda x: 100 * np.cos(100 * x)
    true_diff = lambda x: -10000 * np.sin(100 * x)
    step1 = math.pow(10, -9)
    step2 = math.pow(10, -2)
    point_size = 21
    point_x = np.linspace(-0.001, 0.001, point_size)
    point_y0 = point_y1 = point_y2 = point_y3 = np.zeros(point_size, dtype=np.float64)

    for i, x in enumerate(point_x):
        # 真の導関数値
        point_y0[i] = true_diff(x)
        # 自動微分を用いた導関数値
        obj = AutoDiff.cos(AutoDiff(x, 1.0) * 100) * 100
        point_y3[i] = obj.dx

        
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.plot(point_x, point_y0, color='k', linewidth=2, label="Correct Diff")
    plt.plot(point_x, point_y3, color='pink', linewidth=1, label="Auto Diff")

    plt.legend()
    plt.show()




