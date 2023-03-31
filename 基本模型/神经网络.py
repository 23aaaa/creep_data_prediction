import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

# 固定随机种子
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# 读取数据集到 DataFrame
df = pd.read_csv(r'C:/Users/10239/Desktop/creepdata.csv')

# 将 DataFrame 分为输入特征 X 和目标值 y
X = df.drop('y', axis=1).values
y = df['y'].values

# 归一化输入特征 X 和目标值 y
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
model = Sequential()
model.add(Conv1D(filters=91, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001712)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# 训练模型
model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=1800, batch_size=32, verbose=1)

# 在训练集和测试集上进行预测，并对预测结果进行反归一化
y_train_pred = scaler_y.inverse_transform(model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1)).reshape(-1, 1)).ravel()
y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
y_test_pred = scaler_y.inverse_transform(model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).reshape(-1, 1)).ravel()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# 计算R2, MAE, RMSE
r2_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# 输出结果
print("Training set evaluation:")
print("R2: {:.16f}".format(r2_train))
print("MAE: {:.16f}".format(mae_train))
print("RMSE: {:.16f}".format(rmse_train))
print("\nTest set evaluation:")
print("R2: {:.16f}".format(r2_test))
print("MAE: {:.16f}".format(mae_test))
print("RMSE: {:.16f}".format(rmse_test))
