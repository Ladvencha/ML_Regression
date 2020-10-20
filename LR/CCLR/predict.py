import numpy as np
from ICCStandard.ICCStandard import IPredict
from CCLR.dataloader import DataLoader
from CCLR.model import Model
import io
import sys

np.set_printoptions(threshold=1e6)
np.set_printoptions(suppress=True)
# sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

class Predicter(IPredict):
    def __init__(self, train_file_name, ignore_first_row=False):
        self.ws = None
        self.model_init()
        self.dataloader = DataLoader(train_file_name, ignore_first_row)
    
    def model_init(self):
        self.model = Model().get_model()

    def data_process(self):
        return 0

    def pred(self, prePoint = None, test_split = 0.3):
        ws =  self.model(self.dataloader.X_train, self.dataloader.Y_train, test_split)
        # print(ws)
        # print(prePoint.shape)
        matPre = np.mat(prePoint)
        predict = matPre * ws
        print("预测值分别是：", '\n', predict)



        