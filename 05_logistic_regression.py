"""
逻辑回归——多分类任务——手写数字识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'..\data\handwritten_digits.csv')
# 划分特征、标签
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]
# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征工程：归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义模型：逻辑回归模型
model = LogisticRegression(max_iter=10000)
# 训练模型
model.fit(X_train, y_train)
# 看一下模型训练的效果
print(model.score(X_test, y_test))

# 预测
y_pred = model.predict(X_test[30].reshape(1, -1))  # 注意这里需要reshape将一维数据转换为二维
print(y_pred)
# 查看实际数字
plt.imshow(X_test[30].reshape(28, -1), cmap="Grays_r")
plt.show()
