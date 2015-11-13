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
import matplotlib.cm as cm
import matplotlib.ticker as ti
EPSIRON = 1.0e-8


def forward_diff(func, point, step=None):
    if step is None:
        step = step
    return (func(point+step) - func(point)) / step


def central_diff(func, point, step):
    return (func(point+step) - func(point-step)) / (2 * step)
