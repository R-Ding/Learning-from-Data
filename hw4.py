import matplotlib.pyplot as plt
import numpy as np
import random
import math
from sklearn.linear_model import LinearRegression

def generatePoints(d):
    ''' 
    Generates points x_n in [-1, 1].

    Args:
        d (int): number of points

    Returns:
        array: the points
    '''
    return [[random.uniform(-1, 1)] for i in range(d)]

def f(x):
    ''' 
    Target function f(x).

    Args:
        x (array): x coordinates

    Returns:
        array: y coordinates
    '''
    return np.sin(np.pi * x)

def ahat(n):
    ''' 
    Calculate average slope of the hypotheses.

    Args:
        n (int): number of points

    Returns:
        float: average slope
    '''
    slopes = 0
    for i in range(n):
        pts = np.array(generatePoints(2))
        y = np.array([f(pts[0]), f(pts[1])])
        # Calculate slope of g(x)
        slope = LinearRegression(fit_intercept=False).fit(pts, y).coef_[0]
        slopes += slope
    return slopes / n

def bias(n, ahat):
    ''' 
    Calculates approximate bias.

    Args:
        n (int): the number of data points
        ahat (float): average slope of hypotheses

    Returns:
        float: bias
    '''
    x = np.array(generatePoints(n))
    f_x = f(x)
    gbar_x = ahat * x
    return np.sum(np.power(gbar_x - f_x, 2)) / n

def variance(N, n, ahat):
    ''' 
    Calculates approximate variance.

    Args:
        N (int): the number of trials
        n (int): the number of datapoints
        ahat (float): average slope of hypotheses

    Returns:
        float: variance
    '''
    var = 0
    for i in range(N):
        pts = np.array(generatePoints(2))
        y = np.array([f(pts[0]), f(pts[1])])
        slope = LinearRegression(fit_intercept=False).fit(pts, y).coef_[0]
        x = np.array(generatePoints(n))
        gbar_x = ahat * x
        g_x = slope * x
        var += np.sum(np.power(gbar_x - g_x, 2)) / n
    return var / N

def errorSquaredFunc(N, n, intercept=False):
    ''' 
    Calculates expected value of out-of-sample error for hypotheses
    of the form h(x) = ax^2 + b.

    Args:
        N (int): the number of trials
        n (int): the number of datapoints
        intercept (bool): false if b = 0

    Returns:
        float: out-of-sample error
    '''
    e_out = 0
    for i in range(N):
        pts = generatePoints(2)
        x1, x2 = pts[0][0], pts[1][0]
        y1, y2 = [f(x1), f(x2)]
        a = (x1 ** 2 * y1 + x2 ** 2 * y2) / (x1 ** 4 + x2 ** 4)
        b = 0
        if intercept:
            b = y1 - a * x1 ** 2
        x = np.array(generatePoints(n))
        e_out += np.sum(np.power((a * x ** 2 + b) - f(x), 2)) / n
    return e_out / N

N = 1000
n = 100000
ahat = ahat(n)
bias = bias(n, ahat)
variance = variance(N, n, ahat)
e_out_no_intercept = errorSquaredFunc(N, n)
e_out_intercept = errorSquaredFunc(N, n, intercept=True)



# Hw answers
# 1) d     
# 2) d
# 3) c
# 4) e      # output = 1.4256103248940748
# 5) b      # output = 0.2684476174624181
# 6) a      # output = 0.23459119466234046
# 7) b
# 8) c
# 9) b
# 10) e