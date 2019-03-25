import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('test.csv')
# create 独立变量vector
X = dataset.iloc[:, :-1].values # 第一个冒号是所有列（row），第二个是所有行（column）除了最后一个(Purchased)
Y = dataset.iloc[:, 3].values # 只取最后一个column作为依赖变量。

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])   # (inclusive column 1, exclusive column 3, means col 1 & 2)
X[:, 1:3] = imputer.transform(X[:, 1:3]) # 将imputer 应用到数据