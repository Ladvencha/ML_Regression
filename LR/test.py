# %%
from CCKNN.predict import Predicter
import numpy as np
from multiprocessing import Process, Queue
from numpy import mat
from numpy.linalg import linalg
import datetime as dt

def calRest(xArr, yArr, q):
    rest = xArr.T * yArr
    q.put(rest)

def calInverse(xArr):
    xTx = xArr.T * xArr
    inv = xTx.I
    return inv


if __name__ == '__main__':
    predicter = Predicter('./LR/dataset/train.csv', True)
    predicter.pred()
    q = Queue()
    x_train = np.array(predicter.dataloader.X_train)
    y_train = np.array(predicter.dataloader.Y_train)
    y_train = y_train.astype(np.float64)
    xMat = mat(x_train)
    yMat = mat(y_train).T
    
    starTime3 = dt.datetime.now()
    invMat = calInverse(xMat)
    p1 = Process(target=calRest, args=(xMat, yMat, q))
    p1.start()
    p1.join()
    rest = q.get()
    w = invMat * rest
    endTime3 = dt.datetime.now()
    print('myMPtime: %f ms' % ((endTime3 - starTime3).microseconds / 1000))



# %%
