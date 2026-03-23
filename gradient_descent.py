"""
模型求解——梯度下降法
"""
import numpy as np
import matplotlib.pyplot as plt


# 模拟损失函数
def loss(w):
    return (w ** 2 - 2) ** 2


# 初始化参数与超参数
w = 1.4
alpha = 0.1

w_list = []
y_list = []
for i in range(10):
    y = loss(w)
    w_list.append(w)
    y_list.append(y)
    print(f"第{i + 1}次迭代，参数：{w}，损失函数值：{y}")
    # 计算梯度（损失函数求偏导）
    gra = 4 * w ** 3 - 8 * w
    # 更新参数
    w = w - alpha * gra

# 绘图
w_ori = np.arange(1.4, 1.425, 0.001)
plt.plot(w_ori, loss(w_ori), c='black')
plt.plot(w_list, y_list, c='r')
plt.scatter(w_list, y_list, c='r')
plt.show()
