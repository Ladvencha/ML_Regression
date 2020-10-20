import numpy as np
import operator
from ICCStandard.ICCStandard import IModel
from numpy import mat
from numpy.linalg import linalg
import datetime as dt
from numpy import *
from sklearn.model_selection import train_test_split
import io
import sys

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

class Model(IModel):
    def __init__(self):
        self.load_model()
    
    '''
    加载外部模型主体, 完成模型配置
    '''
    def load_model(self):
        self.model = self.local_weight_linear_regression
    
    '''
    获取最终模型主体
    '''
    def get_model(self):
        return self.model
    
    @staticmethod
    def local_weight_linear_regression(testPoint, xArr, yArr, k):
        
        xMat = mat(xArr)
        yMat = mat(yArr).T
        testPoint = mat(testPoint)
        m = shape(xMat)[0]
        # print(xMat.shape)
        # print(yMat.shape)
        # print(testPoint.shape)
        # print(xMat[5, :].shape)

        # 开始
        weights = mat(eye((m)))
        for j in range(m):  # next 2 lines create weights matrix
            diffMat = testPoint - xMat[j, :]  #
            weights[j, j] = np.exp((diffMat * diffMat.T)[0, 0] / (-2.0 * k ** 2))
            # print((diffMat * diffMat.T)[0, 0] / (-2.0 * k ** 2))
        xTx = xMat.T * (weights * xMat)
        # print(weights)
        # print(np.exp(-2407.10397111125))
        
        if linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        
        ws = xTx.I * (xMat.T * (weights * yMat))
        
        return testPoint * ws