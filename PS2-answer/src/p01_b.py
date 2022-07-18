# This code is used to test the problem found in datasets B.
import util
import numpy as np
import matplotlib.pyplot as plt
def plot(x,y):
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=2)
    plt.show()
Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
plot(Xa,Ya)
plot(Xb,Yb)