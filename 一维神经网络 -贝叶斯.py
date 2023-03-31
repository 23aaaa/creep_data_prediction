import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import tensorflow as tf
import matplotlib.pyplot as plt
from skopt import Optimizer
from skopt.space import Real, Integer
import os
import random
from skopt.utils import use_named_args

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




# 定义超参数空间
space = [
    Real(0.0001, 0.1, name='learning_rate'),
    Integer(64, 256, name='filters'),
    Integer(3, 4, name='kernel_size'),
    Integer(500, 4000, name='epochs')
]

# 在 create_model 函数中获取学习率参数
def create_model(params, X_train):
    learning_rate = float(params['learning_rate'])
    filters = int(params['filters'])
    kernel_size = int(params['kernel_size'])
    seed = 42

    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=(kernel_size,), activation='relu', input_shape=(X_train.shape[1], 1), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
    model.add(Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        run_eagerly=False
    )
    return model

i = 1

# 在 objective 函数中获取学习率参数
def objective(params):
    return -fitness_func(params, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# 在 fitness_func 函数中获取学习率参数
def fitness_func(params, X_train, y_train, X_test, y_test):
    model = create_model(params, X_train)
    epochs = params['epochs']
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=epochs, batch_size=32, verbose=0)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).ravel()
    mse = mean_squared_error(y_test, y_pred)
    fitness = 1 / (mse + 1e-8)
    return fitness

# 创建贝叶斯优化器实例
n_calls = 80  # 可根据需要调整
opt = Optimizer(space)

best_fitness_history = []

for i in range(n_calls):
    x = opt.ask()
    x_dict = {space[j].name: x[j] for j in range(len(space))}
    fitness = objective(x_dict)  # 不要使用 **x_dict
    result = opt.tell(x, fitness)
    best_fitness_history.append(-np.min(opt.yi))
    best_params_index = np.argmin(opt.yi)
    best_learning_rate, best_filters, best_kernel_size, best_epochs = opt.Xi[best_params_index]
    remaining_iterations = n_calls - (i + 1)
    print(f"Iteration {i + 1}: Best fitness = {-np.min(opt.yi)}, Remaining iterations: {remaining_iterations}")

plt.plot(best_fitness_history, '-o')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.show()

# 获取最佳超参数
best_params_index = np.argmin(opt.yi)
best_learning_rate, best_filters, best_kernel_size, best_epochs = opt.Xi[best_params_index]

# 使用最佳超参数创建并训练模型
best_params = {'learning_rate': best_learning_rate, 'filters': best_filters, 'kernel_size': best_kernel_size}
best_model = create_model(best_params, X_train)
best_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=best_epochs, batch_size=39, verbose=1)

# 在测试集上进行评估
y_pred = scaler_y.inverse_transform(best_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).reshape(-1, 1)).ravel()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  # 计算 MAE
print("Best Parameters - Learning rate: {:.6f}, Filters: {}, Kernel size: {}, Epochs: {}".format(best_learning_rate, best_filters, best_kernel_size, best_epochs))
print("Optimized Model - R-squared: {:.6f}, Mean Squared Error: {:.6f}, Mean Absolute Error: {:.6f}".format(r2, mse, mae))
