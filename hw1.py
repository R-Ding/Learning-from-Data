import matplotlib.pyplot as plt
import numpy as np
import random

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
    
def perceptron(n):
    # Generate inputs x_n of the data set as random points
    x = generateData(n)

    # Generate two random points to create the target function
    p1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    p2 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    # Evaluate the target function on each x_n to get the corresponding outputs y_n
    y = generateLabels(x, p1, p2)

    # plot data and target function line
    # plt.plot([x[:,1]], [x[:,2]], 'ro')
    # x1 = np.array([-1, 1])
    # plt.plot(x1, targetFunction(p1, p2, x1))
    # plt.show()

    # Start with the weight vector being all zeros
    w = np.array([0, 0, 0])

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

runs = 1000
n = 100

# Calculate the number of iterations that PLA takes to converge to g
total_iters = 0
for i in range(runs):
    iterations, w, _, _ = perceptron(n)
    total_iters += iterations
ave_iters = total_iters / runs
print(ave_iters)

# Calculate the disagreement between f and g
prob = 0
num_pts = 1000
for i in range(1000):
    _, w, p1, p2 = perceptron(n)
    # Generate 1000 points
    points = generateData(num_pts)
    f = generateLabels(points, p1, p2)
    g = []
    for j in range(num_pts):
        g.append(np.sign(np.dot(np.transpose(w), points[j,:])))
    h = f - g
    n_diff = len(np.where(h != 0)[0])
    prob += n_diff
    
ave_prob = prob / (runs * num_pts)
print(ave_prob)

# Output
# n = 10:  iterations = 10.262, 9.956
#          P = 0.110715 , 0.3712630000000003
# n = 100: iterations = 109.429, 104.504
#          P = 0.37574400000000047, 0.38187699999999997

# Hw answers
# 1) d
# 2) a
# 3) d
# 4) b
# 5) c
# 6) e
# 7) b
# 8) c
# 9) b
# 10) b