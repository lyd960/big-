#波士顿房价预测  
import numpy as np#函数计算用
import pandas as pd#数据库变形用
import matplotlib.pyplot as plt
import random
#data_path='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'#想把网页的数据直接分别按照表格赋值=feature_data
#获取数据
import requests#获取数据库用
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
feature_data = pd.read_csv(url, header=None, delim_whitespace=True)

#增加表头
feature_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
feature_data.columns = feature_names
print(feature_data.head())  # 打印前5行数据
print(feature_data.shape)  # 打印数据形状 检验 目前到这里都正确
#数据变形
data = feature_data.to_numpy()
print(data.shape)

#划分数据集 80%训练 20%测试
from sklearn.model_selection import train_test_split
# 假设 X 是特征数据，y 是标签数据
# 划分特征和标签
X = feature_data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']] # 特征数据
y = feature_data['MEDV']# 标签数据
# 划分数据集：80% 训练集，20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n训练集特征：\n", X_train)
print("测试集特征：\n", X_test)
print("训练集标签：\n", y_train)
print("测试集标签：\n", y_test)
print("\n训练集特征：\n", X_train.head())
print("测试集特征：\n", X_test.head())
print("训练集标签：\n", y_train.head())
print("测试集标签：\n", y_test.head())
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# 6. 预测
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)

# 7. 模型评估
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)  # R^2 分数

print("\n模型评估结果：")
print("均方误差 (MSE):", mse)
print("R^2 分数:", r2)

# 创建包含预测结果和真实值的DataFrame
results_df = pd.DataFrame({
    'True_Value': y_test.values,
    'Predicted_Value': y_pred,
    'Index': y_test.index  # 保留原始索引以便排序
})

# 按索引排序，使数据点按原始顺序排列
results_df = results_df.sort_index()

# 绘制对比图
plt.figure(figsize=(12, 6))
plt.plot(results_df['True_Value'], 'b-', label='True Values (MEDV)', marker='o')
plt.plot(results_df['Predicted_Value'], 'r--', label='Predicted Values', marker='x')

plt.title('True vs Predicted Values Comparison')
plt.xlabel('Data Point Index')
plt.ylabel('MEDV Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
x = range(len(results_df))  # 创建x轴位置
width = 0.35  # 柱状图宽度
plt.bar([i - width/2 for i in x], results_df['True_Value'], width, label='True Values (MEDV)', color='b', alpha=0.7)
plt.bar([i + width/2 for i in x], results_df['Predicted_Value'], width, label='Predicted Values', color='r', alpha=0.7)
plt.title('True vs Predicted Values Comparison (Bar Chart)')
plt.xlabel('Data Point Index')
plt.ylabel('MEDV Value')
plt.xticks(x, results_df.index, rotation=45)  # 设置x轴刻度标签并旋转45度防止重叠
plt.legend()
plt.grid(True)
plt.show()
