import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import random
import tensorflow as tf

# 固定随机数种子
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# 自定义 rmse 评估函数
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# 读取数据
df = pd.read_csv(r'C:/Users/10239/Desktop/creepdata.csv')
X = df.drop('y', axis=1)
y = df['y']

# 数据归一化
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 定义神经网络模型
model = Sequential()
model.add(Dense(950, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(280, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mse', 'mae', rmse])

# 训练神经网络模型
model.fit(X_train, y_train, epochs=450, batch_size=39, verbose=1)

# 使用神经网络模型预测测试集
nn_pred = scaler_y.inverse_transform(model.predict(X_test))

# 定义XGBoost模型，并设置默认参数
xgb_model = xgb.XGBRegressor()

# 训练XGBoost模型
xgb_model.fit(X_train, y_train.ravel())

# 使用XGBoost模型预测测试集
xgb_pred = xgb_model.predict(X_test)

# 将神经网络和XGBoost的预测结果组合在一起
ensemble_pred = 0.3 * nn_pred.ravel() + 0.7 * xgb_pred

# 计算R2分数
r2 = r2_score(y_test, ensemble_pred)
print('Ensemble R² Score:', r2)
