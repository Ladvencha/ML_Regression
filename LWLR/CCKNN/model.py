import numpy as np
import operator
from ICCStandard.ICCStandard import IModel
from numpy import mat
from numpy.linalg import linalg
from sklearn.linear_model import LinearRegression
import time
import datetime as dt
from numpy import *
from math import exp

class Model(IModel):
    def __init__(self):
        self.load_model()
    
    '''
    加载外部模型主体, 完成模型配置
    '''
    def load_model(self):
        self.model = self.knn
    
    '''
    获取最终模型主体
    '''
    def get_model(self):
        return self.model
    
    @staticmethod
    def knn(testPoint, xArr, yArr, k=1.0):
        yArr = yArr.astype(np.float64)
        xMat = mat(xArr);
        yMat = mat(yArr).T
        testPoint = mat(testPoint)
        m = shape(xMat)[0]
        print(xMat.shape)
        print(yMat.shape)
        print(testPoint.shape)
        print(xMat[5, :].shape)

        # mytime
        starTime2 = dt.datetime.now()
        print(starTime2)
        weights = mat(eye((m)))
        for j in range(m):  # next 2 lines create weights matrix
            diffMat = testPoint - xMat[j, :]  #
            weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        xTx = xMat.T * (weights * xMat)
        
        if linalg.det(xTx) == 0.0:
            print
            "This matrix is singular, cannot do inverse"
            return
        
        ws = xTx.I * (xMat.T * (weights * yMat))
        endTime2 = dt.datetime.now()
        print(endTime2)
        print('mytime: %f ms' % ((endTime2 - starTime2).microseconds / 1000))
        
        # return testPoint * ws