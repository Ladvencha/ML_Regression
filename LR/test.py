# %%
from CCLR.predict import Predicter
import numpy as np

predicter = Predicter('./LR/dataset/boston_house_prices.csv', True) # 读取数据，初始化模型
predicter.pred(prePoint = np.array([[0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98, 1],
                                   [0.02731, 0, 7.07, 0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.9, 9.14, 1]]).astype(np.float64)) 



# %%
  