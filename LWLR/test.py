# %%
from CCKNN.predict import Predicter
import numpy as np
from multiprocessing import Process, Queue
from numpy import mat
from numpy.linalg import linalg
import datetime as dt
import random
from numpy import *
from math import exp

def calRest(xMat, weights, yMat, q):
    rest = xMat.T * (weights * yMat)
    q.put(rest)

def calInverse(xMat, weights):
    xTx = xMat.T * (weights * xMat)
    inv = xTx.I
    return inv


if __name__ == '__main__':
    predicter = Predicter('./LWLR/dataset/CASP.csv', True)
    predicter.pred()

    q = Queue()
    x_train = np.array(predicter.dataloader.X_train)
    y_train = np.array(predicter.dataloader.Y_train)
    yArr = y_train.astype(np.float64)
    xMat = mat(x_train)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    testPoint = mat(np.array(predicter.dataloader.X_train)[random.randint(0,9), :])

    starTime3 = dt.datetime.now()
    print(starTime3)
    weights = mat(eye((m)))
    for j in range(m):  # next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]  #
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * 1.0 ** 2))
    invMat = calInverse(xMat, weights)
    p1 = Process(target=calRest, args=(xMat, weights, yMat, q))
    p1.start()
    p1.join()
    rest = q.get()
    w = invMat * rest      
    ws = invMat * rest
    endTime3 = dt.datetime.now()
    print(endTime3)
    print('MPtime: %f ms' % ((endTime3 - starTime3).microseconds / 1000))


# %%
