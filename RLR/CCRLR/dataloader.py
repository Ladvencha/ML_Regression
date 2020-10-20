from ICCStandard.ICCStandard import IDataLoader
import numpy as np
import io
import sys

np.set_printoptions(threshold=1e6)
np.set_printoptions(suppress=True)
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

class DataLoader(IDataLoader):
    def __init__(self, file_name, ignore_first_row=False):
        self.read_data_set(file_name, ignore_first_row)
        self.verify_data()
        self.process_data()

    '''
    读取数据
    '''
    def read_data_set(self, file_name, ignore_first_row=False):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n') # 去掉空格
        if ignore_first_row:
            ori_list = ori_list[1:] # 如果忽略第一行，就从第二行开始
        self.train_list = []
        for line in ori_list: # 循环每一行
            line = line.strip().split(',') # 去掉逗号
            for i in range(len(line) - 1):
                line[i] = float(line[i])
            self.train_list.append(line)  # 将文件中的所有数据读取并以float格式存储到train_list列表中，其中每个列表元素以一行为单位
    
    '''
    验证数据
    '''
    def verify_data(self):
        if len(self.train_list) == 0:
            raise Exception("Train data is empty.") # 如果列表为空则引发异常
        col_num = len(self.train_list[0]) # 每一行的列数
        if col_num < 2:
            raise Exception("文件列数大小小于2(数据集至少要有一个属性列一个标签列) at line 0.")
              # 如果小于2，引发异常
        for idx, item in enumerate(self.train_list):
            if len(item) != col_num:
                raise Exception("第{}行的列数与第一行的不匹配, 应该是{}行但却是{}行.".format(idx, col_num, len(item)))

    '''
    处理数据
    '''
    def process_data(self):
        self.X_train = []
        self.Y_train = []
        for item in self.train_list:
            self.X_train.append(item[:len(item) - 1]) # 建立特征列表
            self.Y_train.append(item[len(item) - 1]) # 建立标签列表
        
        self.X_train = np.array(self.X_train).astype(np.float64)
        self.Y_train = np.array(self.Y_train).astype(np.float64)

        train_ones = np.ones((1, self.X_train.shape[0]))
        self.X_train = np.c_[self.X_train, train_ones.T]

        # print(self.X_train)
        # print(self.Y_train.shape)