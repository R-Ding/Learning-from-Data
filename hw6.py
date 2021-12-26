import numpy as np

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
    E_in = len(getMisclassified(X, y, w, n)) / n

    return E_in, w

def LinRegWeightDecay(n, Z, y, lmbda):
    ''' 
    Perform linear regression with weight decay (regulation).

    Args:
        n (int): number of data points
        Z (array): data points in Z space
        y (array): classifications of Z (labels)        

    Returns:
        float: the in sample error
        np array: the weight vector
    '''
     # weight vector with regulation (w_reg)
    Z_t = np.transpose(Z)
    w = np.dot(Z_t, Z)
    w = np.linalg.inv(w + lmbda * np.identity(w.shape[0]))
    w = np.dot(np.dot(w, Z_t), y)
    
    # Evaluate the in-sample error 
    E_in = len(getMisclassified(Z, y, w, n)) / n

    return E_in, w

def transform(x):
    ''' 
    Nonlinearly transform points in X space to Z space

    Args:
        x (array): array of points to be transformed

    Returns:
        np array: the transformed points
    '''
    Z = []
    for i in range(x.shape[0]):
        x1 = x[i][0]
        x2 = x[i][1]
        Z.append([1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)])
    Z = np.array(Z)
    return Z

# Problem 2
# Load training data and transform it
in_data = np.loadtxt('in.dta.txt')
x1 = in_data[:,0:2]
y1 = in_data[:,2]
n1 = x1.shape[0]
Z1 = transform(x1)

# Apply Linear Regression with training data
E_in, w = LinearRegression(n1, Z1, y1)

# Load test data and transform it
out_data = np.loadtxt('out.dta.txt')
x2 = out_data[:,0:2]
y2 = out_data[:,2]
n2 = x2.shape[0]
Z2 = transform(x2)

# Estimate out of sample error
E_out = len(getMisclassified(Z2, y2, w, n2)) / n2

# print(E_in)
# print(E_out)

# Problem 3-5
lmbda = 10 ** (-2)
# Perform linear regression with weight decay and get errors
E_in2, w_2 = LinRegWeightDecay(n1, Z1, y1, lmbda)
E_out2 = len(getMisclassified(Z2, y2, w_2, n2)) / n2
# print(E_in2)
# print(E_out2)

# Problem 6
# Find min out of sample error
min_E_out = 1
for k in range(-20, 20):
    lmbda = 10 ** (k)
    _, w_2 = LinRegWeightDecay(n1, Z1, y1, lmbda)
    E_out = len(getMisclassified(Z2, y2, w_2, n2)) / n2
    if E_out < min_E_out:
        min_E_out = E_out
# print(min_E_out)

# Hw answers
# 1) b   
# 2) a      # output = 0.02857142857142857, 0.084
# 3) d      # output = 0.02857142857142857, 0.08
# 4) e      # output = 0.37142857142857144, 0.436
# 5) d      # output = 0.228, 0.124, 0.092, 0.056, 0.084
# 6) b      # output = 0.056
# 7) c 
# 8) d
# 9) a
# 10) e