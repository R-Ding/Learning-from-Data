import numpy as np
import random
from sklearn import svm

def error(x, y, w, n):
    ''' 
    Gets ratio of misclassified points to total points

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
    return misclassified_pts, len(misclassified_pts) / n

def LinearRegression(n, X, y):
    ''' 
    Perform linear regression.

    Args:
        n (int): number of data points
        x (array): data points
        y (array): classifications of x (labels)        

    Returns:
        float: the in sample error
        np array: the weight vector
    '''
    # Compute the pseudo-inverse
    X_t = np.transpose(X)
    X_dagger = np.dot(np.linalg.inv(np.dot(X_t, X)), X_t)
    w = np.dot(X_dagger, y)

    # Evaluate the in-sample error 
    _, E_in = error(X, y, w, n)

    return E_in, w

def transform(x, k):
    ''' 
    Nonlinearly transform points in X space to Z space

    Args:
        x (array): array of points to be transformed
        k (int): transform x to phi_0 through phi_k

    Returns:
        np array: the transformed points
    '''
    Z = []
    for i in range(x.shape[0]):
        x1 = x[i][0]
        x2 = x[i][1]
        Z.append([1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)][:k+1])
    Z = np.array(Z)
    return Z

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
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
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

def data(n):
    ''' 
    Generates data d including x_0 = 1 and x which doesn't, and labels y_n that 
    are -1 or 1, depending on whether the points in x are above or below 
    the target function.

    Args:
        n (int): number of data points

    Returns:
        np array: the data points including x_0
        np array: data points excluding x_0
        np array: labels for data points
        np array: the points to create the target function with.
    '''
    # Generate inputs x_n of the data set as random points
    d = generateData(n)
    x = d[:,1:3]
    # Generate two random points to create the target function
    p1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    p2 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    # Evaluate the target function on each x_n to get the corresponding outputs y_n
    y = generateLabels(d, p1, p2)
    return d, x, y, p1, p2

def perceptron(n, w, x, y):
    iterations = 0
    bool = True
    while bool:
        # Perceptron Learning Algorithm
        misclassified_pts, _ = error(x, y, w, n)
        # Choose a point randomly from the set of misclassified points
        if (len(misclassified_pts) != 0):
            idx = random.choice(range(len(misclassified_pts)))
            pt = misclassified_pts[idx]
            # update the weight vector
            w = w + pt[1] * pt[0]
            iterations += 1
        else:
            bool = False
    return iterations, w


# Problems 1-5
in_data = np.loadtxt('in.dta.txt')
x = in_data[:,0:2]
k = 3
reverse = True
for k in range(3, 8):
    z = transform(x, k)
    if reverse:
        z_val = z[0:25]
        z_train = z[25:36]
        y_val = in_data[0:25,2]
        y_train  = in_data[25:36,2]
    else:
        z_train = z[0:25]
        z_val = z[25:36]
        y_train = in_data[0:25,2]
        y_val = in_data[25:36,2]
    E_in, w = LinearRegression(z_train.shape[0], z_train, y_train)

    _, E_val = error(z_val, y_val, w, z_val.shape[0])

    out_data = np.loadtxt('out.dta.txt')
    z_test = transform(out_data[:,0:2], k)
    y_test = out_data[:,2]
    _, E_out = error(z_test, y_test, w, z_test.shape[0])

    print(E_val)

# Problem 6
e = 0
for i in range(1000):
    e1 = np.random.uniform()
    e2 = np.random.uniform()
    e += min(e1, e2)
print(e / 1000)




# Problems 8-10
N = 100
runs = 1000
better = 0
n_vectors = 0
for i in range(runs):
    d, x, y, p1, p2 = data(N)
    if np.max(y) == np.min(y):
        runs += 1
        continue
    w = [0.0, 0.0, 0.0]
    _, w = perceptron(N, w, d, y)

    # Generate 1000 fresh points
    x_test = generateData(1000)
    # Estimate out of sample error
    y_test = generateLabels(x_test, p1, p2)
    _, E_pla = error(x_test, y_test, w, 1000)

    clf = svm.SVC(kernel='linear', C = 9999999999999999999999999999)
    clf.fit(x, y)
    n_vectors += len(clf.support_)
    x_test = x_test[:,1:3]
    y_predict = clf.predict(x_test)
    misclassified = 0
    for j in range(1000):
        if y_predict[j] != y_test[j]:
            misclassified += 1
    E_svm = misclassified / 1000

    if (E_svm < E_pla):
        better += 1
print(better / runs)


# Hw answers
# 1) d     # output = 0.3, 0.5, 0.2, 0.0, 0.1
# 2) e     # output = 0.42, 0.416, 0.188, 0.084, 0.072
# 3) d     # output = 0.28, 0.36, 0.2, 0.08, 0.12
# 4) d     # output = 0.396, 0.388, 0.284, 0.192, 0.196
# 5) b     # output = 0.084, 0.192
# 6) d     # output = 0.32
# 7) c
# 8) c     # output = 0.5384615384615384
# 9) d     # output = 0.5886454183266933
# 10) b    # output = 2.980059820538385