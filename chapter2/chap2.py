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
import functools
import sys
import types
import matplotlib.pyplot as plt
import matplotlib.cm as cm
EPSIRON = 1.0e-8


"""
グラフを描画し、その上に元々与えられていた点列を重ねてプロットする

INPUT:
points: 与えられた点列のリスト[[x_0, f_0], [x_1, f_1], ..., [x_n, f_n]]
x_list: 近似曲線を描写するxの範囲・密度
f_list: 上のxに対応するfの値
"""
def points_on_func(points, x_list, f_list, **kwargs):
    title = kwargs.get('title', "与えられた点と近似曲線")
    xlim = kwargs.get('xlim', False)
    ylim = kwargs.get('ylim', False)
    
    fig, ax = plt.subplots()
    plt.title(title)

    plt.plot(x_list, f_list, color='b', linewidth=1, label="近似曲線")

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    plt.plot(points_x, points_y, 'o', color='r', label="点列")
    
    plt.xlabel("x")
    plt.ylabel("f")
    if xlim:
        ax.set_xlim(xlim)
    
    if ylim:
        ax.set_ylim(ylim)
    plt.legend()
    plt.show()


"""
元々の関数と、（いくつかの）近似関数を重ねてプロットする

INPUT:
x_list: それぞれの近似曲線を描写するxの範囲・密度
f: 元々の関数のf
f_lists: 「それぞれの関数の、上のxに対応するfの値」の配列
"""
def funcs(x_list, f, f_lists, **kwargs):
    title = kwargs.get('title', "元の関数と近似関数")
    xlim = kwargs.get('xlim', False)
    ylim = kwargs.get('ylim', False)
    labels = kwargs.get('labels', False)
    points = kwargs.get('points', False)
    axis = kwargs.get('axis', False)

    if not isinstance(f_lists, list):
        f_lists = [f_lists]
    
    # 近似曲線の本数
    num = len(f_lists)
    
    fig, ax = plt.subplots()
    plt.title(title)

    if axis:
        plt.axvline(x=axis, color='#444444')

    plt.plot(x_list, f, color='#444444', linewidth=3, label="元の曲線")

    for i in range(num):
        if labels:
            plt.plot(x_list, f_lists[i], linewidth=1, color=cm.gist_rainbow(i*1.0/num), label=labels[i])
        else:
            plt.plot(x_list, f_lists[i], linewidth=1, color=cm.gist_rainbow(i*1.0/num))
    
    if points:
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        plt.plot(points_x, points_y, 'o', color='#000000')

    plt.xlabel("x")
    plt.ylabel("f")
    if xlim:
        ax.set_xlim(xlim)
    
    if ylim:
        ax.set_ylim(ylim)
    plt.legend()
    plt.show()


"""
式(2.5)の実装
n+1個の点列を入力し、逆行列を解いて、補間多項式を求め、
n次補間多項式の係数行列[a_0, a_1, ..., a_n]を返す

INPUT
points: n+1個の点列[[x_0, f_0], [x_1, f_1], ..., [x_n, f_n]]

OUTPUT
n次補間多項式の係数行列[a_0, a_1, ..., a_n]を返す
"""
def lagrange(points):
    # 次元数
    dim = len(points) - 1

    # [x_0^0, x_0^1, ..., x_0^n]     [a_0]     [f_0]
    # [            .           ]     [ . ]     [ . ]
    # [            .           ]  *  [ . ]  =  [ . ]
    # [            .           ]     [ . ]     [ . ]
    # [x_n^0, x_n^1, ..., x_n^n]     [a_n]     [f_n]
    #
    # なので、A = X^-1 * F を計算する

    # matrix Xをもとめる（ヴァンデルモンドの行列式）
    x_matrix = np.array([[pow(point[0], j) for j in range(dim + 1)] for point in points])

    # matrix Fをもとめる
    f_matrix = np.array([point[1] for point in points])
    
    # 線形方程式 X * A = F を解く
    a_matrix = np.linalg.solve(x_matrix, f_matrix)

    return a_matrix


"""
係数行列[a_0, a_1, ..., a_n] から、n次多項式 a_0 + a_1 * x + ... + a_n * x^n
を生成して返す（関数を返す）
"""
def make_polynomial(a_matrix):
    def __func__(x):
        f = 0
        for n, a_i in enumerate(a_matrix):
            f += a_i * pow(x, n)

        return f

    return __func__


"""
式(2.7)の実装
補間多項式を変形した式から、逆行列の計算をすることなく、ラグランジュの補間多項式を求める
ただし、今回は補間多項式の係数行列を返すのではなく、具体的なxの値のリストに対して、
補間値のリストを生成して返す

# INPUT
# points: 与えられた点列を入力
# x_list: 補間値を求めたいxのリストを入力

# OUTPUT
# f_list: x_listの各要素に対する補間値のリスト
"""
def lagrange2(points, x_list=np.arange(-5, 5, 0.1)):
    dim = len(points) - 1

    f_list = []
    for x in x_list:
        L = 0
        for i in range(dim + 1):
            Li = 1
            for j in range(dim + 1):
                if j != i:
                    Li *= (x - points[j][0]) / (points[i][0] - points[j][0])

            Li *= points[i][1]
            L += Li

        f_list.append(L)

    return f_list


"""
ネヴィルの算法の実装

# INPUT
# points: 与えられた点列を入力
# x: 補間多項式による近似値f(x)を求めたい点

# OUTPUT
# pointsのうち、xに近い何点かを使って近似した多項式におけるf(x)の値
"""
def neville(points, x, **kwargs):
    # 使用する点の最大数
    size = kwargs.get('size', len(points))
    size = size if len(points) >= size else len(points)

    # 途中でf(x)の値の変化が小さくなったら、その時点で値を返すか
    useallpoints = kwargs.get('useallpoints', True)

    # xから近い順に点列を並び替え
    ordered = sorted(points, key=lambda point: pow(point[0] - x, 2))

    # DP用メモを初期化
    table = [[0 for j in range(i+1)] for i in range(size)]

    # xに近い点から1つずつとりだしてループ。
    # f(x)の値が適当に収束するか、点を全て使い切ったら終了
    # (使った点の個数 - 1)次の多項式で関数を近似した時の、f(x)の値を返す
    for i in range(size):
        table[i][0] = ordered[i][1]

        for j in range(1, i+1):
            table[i][j] = ((ordered[i][0] - x) * table[i-1][j-1] - (ordered[i-j][0] - x) * table[i][j-1]) / (ordered[i][0] - ordered[i-j][0])

        print(table[i][i])
        if not useallpoints and math.fabs(table[i][i] - table[i-1][i-1]) < EPSIRON:
            print("途中終了しました。全部で {0} 個の点を使いました。".format(i+1))
            return table[i][i]

    print("最大限の点を用いて近似しました。全部で {0} 個の点を使いました。".format(i+1))
    return table[size-1][size-1]


def spline(points, x_list, step, **kwargs):
    flag = kwargs.get('flag', False)

    size = len(points)
    a_matrix = np.zeros((size-2, size-2))
    f_matrix = np.zeros(size-2)

    # 小さい順に点列を並び替え
    points = sorted(points, key=lambda point: point[0])
    print(points)

    for i in range(size-2):
        for j in range(size-2):
            if j == i:
                a_matrix[i][j] = 4
            elif j == i-1 or j == i+1:
                a_matrix[i][j] = 1
            else:
                a_matrix[i][j] = 0

        f_matrix[i] = (points[i][1] - 2 * points[i+1][1] + points[i+2][1]) * 6.0 / pow(step, 2)
    
    dx2_matrix = np.r_[[0], np.linalg.solve(a_matrix, f_matrix), [0]]
    print(a_matrix)
    print(f_matrix)
    print(dx2_matrix)

    f_list = []
    def __expression(p, q, u, v, x, fv, fu):
        return ((q-p)*pow(x-u, 3))/(6*(v-u)) + (p * pow(x-u, 2))/2 + ( (fv-fu)/(v-u) - (q+2*p)*(v-u)/6 )*(x-u) + fu
    

    for x in x_list:
        if x < points[0][0] or x > points[-1][0]:
            f_list.append(0)

        else:
            for i in range(size-1):
                if points[i][0] <= x <= points[i+1][0]:
                    fu = points[i][1]
                    fv = points[i+1][1]
                    p = dx2_matrix[i]
                    q = dx2_matrix[i+1]
                    u = points[i][0]
                    v = points[i+1][0]
                    f_list.append(__expression(p, q, u, v, x, fv, fu))
                    if flag:
                        print(p, q)
                        x3 = (q-p) / (6*(v-u))
                        x2 = -3 * u * (q-p)/(6*(v-u)) + p/2
                        x1 = 3 * pow(u, 2) * (q-p) / (6*(v-u)) - u*p + ( (fv-fu)/(v-u) - (q+2*p)*(v-u)/6 )
                        x0 = -1*pow(u, 3)*(q-p)/(6*(v-u)) + p/2*pow(u, 2) + ( (fv-fu)/(v-u) - (q+2*p)*(v-u)/6 )*-1*u + fu
                        print("x = {0:.4f} の時、{1:.3f} x^3 + {2:.3f} x^2 + {3:.3f} x + {4:.3f}".format(x, x3, x2, x1, x0))
                    break

    return f_list



if __name__ == "__main__":
    """
    # lagrange()で求めた補間多項式と、元の点列をプロットしてみる
    # 与えられた点列のリスト
    points = [[1, 1], [2, 2], [3, 1], [4, 1], [5, 3]]

    # ラグランジュの補間多項式の係数行列を求める
    a_matrix = lagrange(points)

    # 係数行列を多項式に変換
    func_lagrange = make_polynomial(a_matrix)

    # 0から8まで、0.1刻みでxとfの値のセットを求める
    x_list = np.arange(0, 8, 0.1)
    f_list = func_lagrange(x_list)

    # プロットする
    points_on_func(points, x_list, f_list)
    """

    """
    # lagrange2でも同じことが出来る
    points = [[1, 1], [2, 2], [3, 1], [4, 1], [5, 3]]
    a_matrix = lagrange2(points, np.arange(-10, 8, 0.1))
    points_on_func(points, np.arange(0, 8, 0.1), a_matrix, title="lagrange2で求めた補間多項式と元の点列をプロット")
    """

    """
    # 例5を試す
    points = [[1, 1], [2, 2], [3, 1], [4, 1], [5, 3]]
    print(neville(points, 2.2))
    """


    """
    # 知っているsinの値を用いて、sin26°を求める
    points_theta = [-90, -60, -45, -30, 0, 30, 45, 60, 90]
    points_x = [theta * math.pi / 180 for theta in points_theta]
    points_f = np.sin(points_x)
    points = [[x, f] for x, f in zip(points_x, points_f)]

    print(neville(points, 26 * math.pi / 180, useallpoints=False))
    print(np.sin(26 * math.pi / 180))
    """
    


    """
    # 任意の点の数を用いて、sin26°を求める
    rad_26 = 26 * math.pi / 180
    points_theta = [-90, -60, -45, -30, 0, 30, 45, 60, 90]
    points_x = [theta * math.pi / 180 for theta in points_theta]
    points_f = np.sin(points_x)
    points = [[x, f] for x, f in zip(points_x, points_f)]
    ordered = sorted(points, key=lambda point: pow(point[0] - rad_26, 2))

    x_list = np.linspace(-math.pi, math.pi, 1000)
    f_list = np.sin(x_list)
    f2 = lagrange2(ordered[0:2], x_list)
    f3 = lagrange2(ordered[0:3], x_list)
    f4 = lagrange2(ordered[0:4], x_list)
    f5 = lagrange2(ordered[0:5], x_list)
    f6 = lagrange2(ordered[0:6], x_list)

    funcs(x_list, f_list, [f2, f3, f4, f5, f6], labels=["2点", "3点", "4点", "5点", "6点"], points=points, axis=rad_26)
    """

    """
    func = lambda x: 1 / (25 * x**2 + 1)

    points_x = np.linspace(-2, 2, 0.5)
    points_f = func(points_x)
    points = [[x, f] for x, f in zip(points_x, points_f)]


    x_list = np.linspace(-2, 2, 1000)
    f_list = func(x_list)
    f1 = lagrange2(points[0:5], x_list)
    f2 = lagrange2(points[0:7], x_list)
    f3 = lagrange2(points[0:9], x_list)
    funcs(x_list, f_list, [f1, f2, f3], labels=["5点", "7点", "9点"], points=points, axis=0)
    """


    points = [[1, 1], [2, 1], [3, 2], [4, 3]]
    x_list = np.arange(1, 5, 0.01)
    f_list = spline(points, x_list, 1, flag=True)
    points_on_func(points, x_list, f_list, xlim=[0, 8], ylim=[0, 5])


    








