import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score


def one_versus_all(X, y, digit):
    # binary classifier sets digit to +1 and all others to -1
    y = np.where(y == digit, 1, -1)

    # Implement SVM with soft margin and polynomial kernel
    clf = svm.SVC(C = C, kernel = 'poly', degree = Q, gamma = 1, coef0 = 1)
    clf.fit(X, y)

    # Calculate E_in
    y_predict = clf.predict(X)
    E_in = np.mean(y_predict != y)
    return clf, E_in, len(clf.support_vectors_)

def one_versus_one(X, y, dig1, dig2, C, Q):
    # binary classifier sets one digit to +1 and another digit to -1
    idxs = [i for i in range(len(y)) if y[i] in [dig1, dig2]]
    y = np.array([a for a in y if a in [dig1, dig2]])
    y = np.where(y == dig1, 1, -1)

    # Disregard all other digits and their data
    X = X[idxs]

    # Implement SVM with soft margin and polynomial kernel
    clf = svm.SVC(C = C, kernel = 'poly', degree = Q, gamma = 1, coef0 = 1)
    clf.fit(X, y)

    # Calculate E_in
    y_predict = clf.predict(X)
    E_in = np.mean(y_predict != y)
    return clf, E_in, len(clf.support_vectors_)

def ovo_cross_val(X, y, dig1, dig2, C, Q):
    # binary classifier sets one digit to +1 and another digit to -1
    idxs = [i for i in range(len(y)) if y[i] in [dig1, dig2]]
    y = np.array([a for a in y if a in [dig1, dig2]])
    y = np.where(y == dig1, 1, -1)

    # Disregard all other digits and their data
    X = X[idxs]

    # Implement SVM with soft margin and polynomial kernel
    clf = svm.SVC(C = C, kernel = 'poly', degree = Q, gamma = 1, coef0 = 1)
    errors = 1 - cross_val_score(clf, X, y, cv=10)
    return np.mean(errors)


# Problems 2 and 3
C = 0.01
Q = 2

# Read and parse the given training and testing data
train_data = np.loadtxt('features.train.txt')
X_train = train_data[:, 1:]
y_train = train_data[:, 0:1].flatten()
test_data = np.loadtxt('features.test.txt')
X_test = test_data[:, 1:]
y_test = test_data[:, 0:1].flatten()

for digit in [1, 3, 5, 7, 9]:
    _, E_in, _ = one_versus_all(X_train, y_train, digit)
    print(E_in)

# Problem 4
_, _, num_SV_0 = one_versus_all(X_train, y_train, 0)
_, _, num_SV_1 = one_versus_all(X_train, y_train, 1)
print(num_SV_0 - num_SV_1)

# Problem 5
for C in [0.001, 0.01, 0.1, 1]:
    _, E_in, num_SV = one_versus_one(X_train, y_train, 1, 5, C, Q)
    print(E_in)

# Problem 6
for C in [0.001]:
    for Q in [2, 5]:
        _, E_in, num_SV = one_versus_one(X_train, y_train, 1, 5, C, Q)
        print(num_SV)

# Problem 7
runs = 100
select = []
for i in range(runs):    
    # shuffle data
    temp = list(zip(X_train, y_train))
    random.shuffle(temp)
    X_train, y_train = zip(*temp)
    X_train = np.array(X_train)

    e = []
    for C in [0.0001, 0.001, 0.01, 0.1, 1]:
        errors = ovo_cross_val(X_train, y_train, 1, 5, C, 2)
        e.append(errors)
    # Keep track of the index of the C with lowest error after testing all 5
    select.append(e.index(min(e)))
# output the most frequent C in array of Cs with lowest error
print(max(set(select), key=select.count))

# Problem 8
e = []
for i in range(runs):
    e.append(ovo_cross_val(X_train, y_train, 1, 5, 0.01, 2))
# output average cross val error for C = 0.01
print(np.mean(e))

# Problems 9 and 10
def ovo_rbf(X, y, dig1, dig2, C):
    # binary classifier sets one digit to +1 and another digit to -1
    idxs = [i for i in range(len(y)) if y[i] in [dig1, dig2]]
    y = np.array([a for a in y if a in [dig1, dig2]])
    y = np.where(y == dig1, 1, -1)

    # Disregard all other digits and their data
    X = X[idxs]

    # Implement SVM with soft margin and polynomial kernel
    clf = svm.SVC(C = C, kernel = 'rbf', gamma = 1)
    clf.fit(X, y)

    # Calculate E_in
    y_predict = clf.predict(X)
    E_in = np.mean(y_predict != y)
    return clf, E_in, len(clf.support_vectors_)

# prepare test data for one vs one
idxs = [i for i in range(len(y_test)) if y_test[i] in [1, 5]]
y_test = np.array([a for a in y_test if a in [1, 5]])
y_test = np.where(y_test == 1, 1, -1)

# Disregard all other digits and their data
X_test = X_test[idxs]

for C in [0.01, 1, 100, 10 ** 4, 10 ** 6]:
    clf, E_in, _ = ovo_rbf(X_train, y_train, 1, 5, C)
    # print(E_in)
    y_predict = clf.predict(X_test)
    E_out = np.mean(y_predict != y_test)
    # print(E_out)


# Hw answers
# 1) d   
# 2) a      # output = 0.10588396653408312, 0.10026059525442327, 0.08942531888629818, 0.09107118365107666, 0.07433822520916199
# 3) a      # 0.014401316691811822, 0.09024825126868742, 0.07625840076807022, 0.08846523110684405, 0.08832807570977919
# 4) c      # output = 1793
# 5) d      # output = 0.004484304932735426, 0.004484304932735426, 0.004484304932735426, 0.0032030749519538757
# 6) b      # output = 76, 25
# 7) b      # output = 1
# 8) c      # output = 0.004487179487179471
# 9) e      # output = 0.003843689942344651, 0.004484304932735426, 0.0032030749519538757, 0.0025624599615631004, 0.0006406149903907751
# 10) c     # output = 0.02358490566037736, 0.02122641509433962, 0.018867924528301886, 0.02358490566037736, 0.02358490566037736