import numpy as np
from ICCStandard.ICCStandard import IPredict
from CCLWLR.dataloader import DataLoader
from CCLWLR.model import Model
from sklearn.model_selection import train_test_split

class Predicter(IPredict):
    def __init__(self, train_file_name, ignore_first_row=False):
        self.model_init()
        self.dataloader = DataLoader(train_file_name, ignore_first_row)
    
    def model_init(self):
        self.model = Model().get_model()

    def data_process(self):
        return 0

    def pred(self, prePoint = None, test_split = 0.3, k = 10.0):
        X_train, X_test, Y_train, Y_test = train_test_split(self.dataloader.X_train, self.dataloader.Y_train, test_size=test_split, random_state=5)
        
        # print(X_test[0,:])
        sum = 0
        # print(self.model(X_test[20, :], X_train, Y_train, k))
        # print(Y_train[20])
        # print(X_test[1, :])
        for i in range(prePoint.shape[0]):
            print("第",i+1,"个点的预测值为：", self.model(prePoint[i, :], X_train, Y_train, k))

