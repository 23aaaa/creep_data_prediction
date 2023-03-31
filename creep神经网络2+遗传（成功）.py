import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import keras.backend as K
import random
import tensorflow as tf

# 设置随机数种子以保证可重复性
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 定义自定义 RMSE 评估函数
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# 将数据集从 CSV 文件中读取到 Pandas DataFrame 中
df = pd.read_csv(r'C:/Users/10239/Desktop/creepdata.csv')

# 从 DataFrame 中分离输入特征 X 和目标变量 y
X = df.drop('y', axis=1)
y = df['y']

# 使用 MinMaxScaler 将输入特征 X 和目标变量 y 进行归一化
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 使用 train_test_split 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 使用 Keras 的 Sequential API 定义神经网络结构
model = Sequential()
model.add(Dense(950, input_dim=X_train.shape[1], activation='relu'))  # 第一层隐藏层，包含 950 个神经元，激活函数为 ReLU
model.add(Dropout(0.2))
model.add(Dense(280, activation='relu'))  # 第二层隐藏层，包含 280 个神经元，激活函数为 ReLU
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))  # 输出层，包含 1 个神经元，激活函数为线性

# 使用 Adam 优化器和均方误差损失函数编译模型
optimizer = Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae', rmse])

# 在训练集上训练模型，并添加验证集
model.fit(X_train, y_train, epochs=450, batch_size=39, verbose=1, validation_split=0.1)

# 在测试集上评估模型性能
loss, mse, mae, rmse = model.evaluate(X_test, y_test)
print('均方误差：', mse)
print('平均绝对误差：', mae)
print('均方根误差：', rmse)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print('R² 分数：', r2)

# 在测试集上进行预测并将结果转换为原始比例
y_pred_actual = scaler_y.inverse_transform(y_pred)
