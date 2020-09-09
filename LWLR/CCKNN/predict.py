import numpy as np
from ICCStandard.ICCStandard import IPredict
from CCKNN.dataloader import DataLoader
from CCKNN.model import Model
import random

class Predicter(IPredict):
    def __init__(self, train_file_name, ignore_first_row=False):
        self.model_init()
        self.dataloader = DataLoader(train_file_name, ignore_first_row)
    
    def model_init(self):
        self.model = Model().get_model()

    def data_process(self):
        return 0

    def pred(self):
        return self.model(testPoint=np.array(self.dataloader.X_train)[random.randint(0,9), :], xArr=np.array(self.dataloader.X_train), yArr=np.array(self.dataloader.Y_train), k=1.0)