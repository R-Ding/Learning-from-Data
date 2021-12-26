import numpy as np
import random

def errorFunc(u, v):
    ''' 
    The nonlinear error surface.

    Args:
        u (float): u coordinate
        v (float): v coordinate

    Returns:
        float: the error E(u, v)
    '''
    return (u * np.exp(v) - 2 * v * np.exp(-1 * u)) ** 2

def errorGradient(u, v):
    ''' 
    Calculates change in u and v coordinates.

    Args:
        u (float): u coordinate
        v (float): v coordinate

    Returns:
        array: change in u and v coordinates
    '''
    du = 2 * (u * np.exp(v) - 2 * v * np.exp(-1 * u)) * (np.exp(v) + 2 * v * np.exp(-1 * u))
    dv = 2 * (u * np.exp(v) - 2 * v * np.exp(-1 * u)) * (u * np.exp(v) - 2 * np.exp(-1 * u))
    return [du, dv]

def gradientDescent(eta, u, v, final_err):
    ''' 
    Performs gradient descent in the uv space.

    Args:
        eta (float): learning rate
        u (float): u coordinate
        v (float): v coordinate
        final_err: The error for which once reached, iteration will stop

    Returns:
        int: the number of iterations
        u: the final u coordinate
        v: the final v coordinate
    '''
    iterations = 0
    error = errorFunc(u, v)
    while error > final_err:
        du, dv = errorGradient(u, v)
        u -= eta * du
        v -= eta * dv
        error = errorFunc(u, v)
        iterations += 1
    return iterations, u, v

def coordinateDescent(eta, u, v):
    ''' 
    Performs coordinate descent by moving along the u coordinate
    followed by the v coordinate.

    Args:
        eta (float): learning rate
        u (float): u coordinate
        v (float): v coordinate

    Returns:
        float: the error E(u, v)
    '''
    for i in range(15):
        du = errorGradient(u, v)[0]
        u -= eta * du
        error = errorFunc(u, v)
        dv = errorGradient(u, v)[1]
        v -= eta * dv
        error = errorFunc(u, v)
    return error

def generateData(n):
    ''' 
    Generates points x_n = [1, x1, x2], where x1 and x2 are in 
    [-1, 1].

    Args:
        n (int): number of points

    Returns:
        np array: the points
    '''
    return np.array([[1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)] for i in range(n)])

def targetFunction(p1, p2, x1):
    ''' 
    Given two points, calculates the line through both and evaluates
    the line at the given x1 coordinate.

    Args:
        p1 (array): x1 and x2 coordinates for a random point
        p2 (array): x1 and x2 coordinates for a random point
        x1 (array): x1 coordinates to evaluate target function at.

    Returns:
        np array: the x2 coordinates
    '''
    m = (p2[2] - p1[2]) / (p2[1] - p1[1])
    b = p1[2] - m * p1[1]
    return m * x1 + b

def generateLabels(x, p1, p2):
    ''' 
    Generates labels y_n that are -1 or 1, depending on whether the
    points in x are above or below the target function.

    Args:
        x (array): the data points
        p1, p2 (arrays): the points to create the target function with.

    Returns:
        np array: the labels (y) for the data points
    '''
    return np.sign(x[:,2] - targetFunction(p1, p2, x[:,1]))

def err_out(w, p1, p2):
    ''' 
    Estimates cross entropy error.

    Args:
        w (array): the weight vector resulting from logistic regression
        p1 (array): 1st point for target function
        p2 (array): 2nd point for target function

    Returns:
        float: the cross entropy error E_out
    '''
    x = generateData(1000)
    y = generateLabels(x, p1, p2)
    error = 0
    for i in range(1000):
        error += np.log(1 + np.exp(-1 * y[i] * np.dot(np.transpose(w), x[i])))
    return error / 1000


# Problems 5, 6
iterations, u, v = gradientDescent(0.1, 1, 1, 10 ** (-14))
# Problem 7
error = coordinateDescent(0.1, 1, 1)

# Problems 8, 9
N = 100
n = 100
epochs = 0
E_out = 0
# Trials
for i in range(N):
    # Generate datapoints and labels
    x = generateData(n)
    p1, p2 = generateData(2)
    y = generateLabels(x, p1, p2)
    # initialize weight vector
    w = np.array([1.0, 1.0, 1.0])
    new_w = np.array([0.0, 0.0, 0.0])
    # Continue iterating until stop condition
    while np.linalg.norm(new_w - w) >= 0.01:
        w = new_w
        # shuffle data
        perm = np.random.permutation(n)
        for i in perm:
            # Calculate gradient descent for a data point (stochastic)
            grad =  (-1 * y[i] * x[i]) / (1 + np.exp(y[i] * np.dot(w, x[i])))
            # update weight vector by logistic regression
            new_w = new_w - 0.01 * grad
        epochs += 1
    E_out += err_out(new_w, p1, p2)
epochs = epochs / N
E_out = E_out / N


# Hw answers
# 1) c   
# 2) d
# 3) c
# 4) e      
# 5) d      # output = 10
# 6) e      # output = (0.04473629039778207, 0.023958714099141746)
# 7) a      # output = 0.13981379199615315
# 8) d      # output = 0.10412320807178446
# 9) a      # output = 333.62
# 10) e