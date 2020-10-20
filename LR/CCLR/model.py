import numpy as np
import io
import sys
import operator
from ICCStandard.ICCStandard import IModel
from numpy.linalg import linalg
from sklearn.linear_model import LinearRegression
import datetime as dt
from sklearn.model_selection import train_test_split

sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

def calMSE(matW, xArr, yArr):
        matTest_X = np.mat(xArr)
        matTest_Y = np.mat(yArr).T
        pre_X = matTest_X * matW
        diff = pre_X - matTest_Y
        # print(diff)
        sum = 0
        for i in range(matTest_X.shape[0]):
            sum = sum + diff[i, 0] ** 2
        MSE = sum / matTest_X.shape[0]
        return MSE


class Model(IModel):
    def __init__(self):
        self.load_model()
    
    '''
    加载外部模型主体, 完成模型配置
    '''
    def load_model(self):
        self.model = self.Linear_Regression
    
    '''
    获取最终模型主体
    '''
    def get_model(self):
        return self.model

    
    @staticmethod # 线性回归实现代码
    def Linear_Regression(x_train, y_train, test_split):

        # 划分数据集
        X_train, X_test, Y_train, Y_test = train_test_split(x_train,  y_train, test_size=test_split, random_state=5)

        #  开始训练
        xMat = np.mat(X_train)
        yMat = np.mat(Y_train).T
        xTx = xMat.T * xMat
        if linalg.det(xTx) == 0.0:
            raise Exception("选取的数据集矩阵无法求逆，请重新划分数据集")
            return
        ws = xTx.I * (xMat.T * yMat)
        # print(ws)
        # 评估模型
        # ---------计算训练误差
        print("训练集/测试集 = ", (1 - test_split) * 10,'/', (test_split) * 10)
        print("训练集均方误差：", calMSE(ws, X_train, Y_train))
        # ---------计算测试误差
        print("测试集均方误差：", calMSE(ws, X_test, Y_test))
        return ws