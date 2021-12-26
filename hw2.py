import matplotlib.pyplot as plt
import numpy as np
import random


def flipCoins(runs):
    coins = 1000
    c_1 = 0
    c_rand = 0
    c_min = 1
    v_1 = 0
    v_rand = 0
    v_min = 0
    for i in range(runs):
        for j in range(coins):
            c = np.random.randint(2, size = 10)
            c_rand_idx = random.randint(0, coins - 1)
            heads = np.count_nonzero(c) / 10
            if j == 0:
                c_1 = heads
            if j == c_rand_idx:
                c_rand = heads
            if heads < c_min:
                c_min = heads
        v_1 += c_1
        v_rand += c_rand
        v_min += c_min
    ave_v_1 = v_1 / runs
    ave_v_rand = v_rand / runs
    ave_v_min = v_min / runs

    results = [ave_v_1, ave_v_rand, ave_v_min]
    print(results)

# flipCoins(100000)


def generateData(n):
    ''' 
    Generates points x_n = [1, x1, x2], where x1 and x2 are in 
    [-1, 1].

    Args:
        n (int): number of points

    Returns:
        np array: the points
    '''
    x = []
    for i in range(n):
        x.append([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
    return np.array(x)

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
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
    return m * x1 + b

def generateLabels(x, p1, p2):
    ''' 
    Generates labels y_n that are -1 or 1, depending on whether the
    points in x are above or below the target function.

    Args:
        x (array): the points

    Returns:
        np array: the labels
    '''
    y = []
    y = np.sign(x[:,2] - targetFunction(p1, p2, x[:,1]))
    return y

def getMisclassified(x, y, w, n):
    ''' 
    Gets array of misclassified points

    Args:
        x (array): array of points to be classified with w
        y (array): true classifications of x (labels)
        w (array): 1 x 3 weight matrix
        n (int): number of points

    Returns:
        np array: the misclassified points
    '''
    misclassified_pts = []
    for i in range(n):
        misclassified = np.sign(np.dot(np.transpose(w), x[i,:])) != y[i]
        if misclassified:
            misclassified_pts.append([x[i,:], y[i]])
    return misclassified_pts


def perceptron(n, w):
    # Generate inputs x_n of the data set as random points
    x = generateData(n)

    # Generate two random points to create the target function
    p1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    p2 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    # Evaluate the target function on each x_n to get the corresponding outputs y_n
    y = generateLabels(x, p1, p2)

    iterations = 0
    bool = True
    while bool:
        # Perceptron Learning Algorithm
        misclassified_pts = getMisclassified(x, y, w, n)
        # Choose a point randomly from the set of misclassified points
        if (len(misclassified_pts) != 0):
            idx = random.choice(range(len(misclassified_pts)))
            pt = misclassified_pts[idx]
            # update the weight vector
            w = w + pt[1] * pt[0]
            iterations += 1
        else:
            bool = False
    return iterations, w, p1, p2


def LinearRegression(n, X, y):
    # Compute the pseudo-inverse
    X_t = np.transpose(X)
    X_dagger = np.dot(np.linalg.inv(np.dot(X_t, X)), X_t)
    w = np.dot(X_dagger, y)

    # Evaluate the in-sample error 
    E_in = len(getMisclassified(X, y, w, n)) / n

    return E_in, w


# Experiments
runs = 1000
n = 100

# # Problems 5 and 6
E_in = 0
E_out = 0
for i in range(runs):
    # Generate inputs x_n of the data set as random points
    X = generateData(n)
    # Generate two random points to create the target function
    p1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    p2 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    # Evaluate the target function on each x_n to get the corresponding outputs y_n
    y = generateLabels(X, p1, p2)

    # Apply Linear Regression
    error_in, w = LinearRegression(n, X, y)
    E_in += error_in

    # Generate 1000 fresh points
    x = generateData(1000)
    # Estimate out of sample error
    y = generateLabels(x, p1, p2)
    E_out += len(getMisclassified(x, y, w, 1000)) / 1000

ave_E_in = E_in / runs
ave_E_out = E_out / runs

# print(ave_E_in)
# print(ave_E_out)


# Problem 7
n = 10
iterations = 0
for i in range(runs):
    # Generate inputs x_n of the data set as random points
    x = generateData(n)
    # Generate two random points to create the target function
    p1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    p2 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    # Evaluate the target function on each x_n to get the corresponding outputs y_n
    y = generateLabels(x, p1, p2)

    # Apply Linear Regression to find weights
    _, w = LinearRegression(n, x, y)

    # Use weights as initial for PLA
    iters, _, _, _, = perceptron(n, w)
    iterations += iters
ave_iterations = iterations / runs
print(ave_iterations)

def targetFunction2(x, n):
    ''' 
    Calculate target function f(x_1, x_2) = sign(x_1^2, x_2^2 - 0.6)

    Args:
        x (array): (1, x1, x2) data points.
        
    Returns:
        np array: the resulting labels calculated by the target function
    '''
    x1 = x[:,1]
    x2 = x[:,2]
    y = np.sign(np.add(np.power(x1, 2), np.power(x2, 2)) - 0.6)
    # randomly flip 10% of the output
    size = int(0.1 * n)
    idxs = random.sample(range(n), k=size)
    for i in idxs:
        y[i] = -1 * y[i]
    return y

def transform(x):
    x_prime = []
    for j in range(n):
        x1 = x[j][1]
        x2 = x[j][2]
        x_prime.append([1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2])
    x_prime = np.array(x_prime)
    return x_prime

# Problem 8
E_in = 0
E_out = 0
for i in range(runs):
    # Generate inputs x_n of the data set as random points
    x = generateData(n)
    # Transform training data into nonlinear feature vectors
    x_prime = transform(x)
    # Evaluate the target function on each x_n to get the corresponding outputs y_n
    y = targetFunction2(x, n)
    y_prime = targetFunction2(x_prime, n)

    # Apply Linear Regression
    error_in, w = LinearRegression(n, x, y)
    _, w_prime = LinearRegression(n, x_prime, y_prime)
    E_in += error_in

    # Generate 1000 fresh points and transform them
    x = generateData(1000)
    x_prime = transform(x)
    # Estimate out of sample error
    y_prime = targetFunction2(x_prime, n)
    E_out += len(getMisclassified(x_prime, y_prime, w_prime, n)) / 1000

ave_E_in = E_in / runs
ave_E_out = E_out / runs
# print(ave_E_in)
# print(ave_E_out)




# Hw answers
# 1) b      output: 0.01
# 2) d
# 3) e
# 4) b      
# 5) c      output: 0.04037, 0.0293
# 6) c      output: 0.04992, 0.04847
# 7) a      output: 7.828, 7.706
# 8) d      output: 0.50308
# 9) a      output: [-1.02389139 -0.09481602  0.00881944  0.0120234   1.59266596  1.54646615]
# 10) b     output: 0.125894
