import numpy as np
import pandas as pd
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import pygad.kerasga
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import random

# 超参数设置的注释
'''
num_generations: 遗传算法的最大迭代次数。   num_parents_mating: 在每一代中被选为父母的染色体数量。  
fitness_func: 定义染色体适应度的函数。   sol_per_pop: 种群中的染色体数量。   num_genes: 每个染色体包含的基因数量。  
gene_space: 每个基因可能的取值范围，是一个包含字典的列表，每个字典表示一个基因，包含low和high两个键，分别表示基因的最小值和最大值。   
init_range_low: 初始种群的取值下界，是一个包含初始种群每个基因下界的列表。   
init_range_high: 初始种群的取值上界，是一个包含初始种群每个基因上界的列表。   
parent_selection_type: 父选择的方法，可以是"sss"、"rws"、"sus"、"random"之一，
分别表示随机同样选择、轮盘赌选择、随机不重复选择、随机选择。   
crossover_type: 交叉的方式，可以是"single_point"、"two_points"、"uniform"之一，
分别表示单点交叉、双点交叉、均匀交叉。   
mutation_type: 变异的方式，可以是"random"、"swap"、"scramble"、"inversion"之一，
分别表示随机变异、交换变异、扰动变异、翻转变异。   
mutation_percent_genes: 变异概率，表示每个基因变异的概率。   
ga_instance: 传递一个KerasGA对象，表示使用Keras模型进行遗传算法优化。   
'''

# 固定随机种子
# 固定 numpy 随机数种子，保证随机数序列是可重复的
np.random.seed(42)
# 固定 Python 随机数种子，保证随机数序列是可重复的
random.seed(42)
# 固定 TensorFlow 随机数种子，保证随机数序列是可重复的
tf.random.set_seed(42)
# 固定 CPU 和 GPU 配置，保证每次运行程序时生成的随机数序列是相同的
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

# 导入所需库并定义一个Sequential模型
from keras.models import Sequential
model = Sequential()


# 添加  卷积层，filters参数指定卷积核的数量，kernel_size参数指定卷积核大小，activation参数指定激活函数，input_shape参数指定输入数据的形状
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
# 添加  最大池化层，pool_size参数指定池化窗口大小
model.add(MaxPooling1D(pool_size=2))
# 添加  Dropout层，参数为0.2表示在训练过程中每次更新时随机丢弃20%的神经元
model.add(Dropout(0.2))
# 添加  Flatten层，将多维数据展平为一维数据
model.add(Flatten())
# 添加  全连接层，Dense表示全连接层，100为神经元数量，activation参数指定激活函数
model.add(Dense(100, activation='relu'))
# 添加  输出层，Dense表示全连接层，1为神经元数量，activation参数指定激活函数
model.add(Dense(1, activation='linear'))


# 编译模型，loss参数指定损失函数，optimizer参数指定优化器
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型，X_train为输入数据，y_train为输出数据，epochs参数指定训练轮数，batch_size参数指定每次训练的样本数量，verbose参数为1表示打印训练过程信息
model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=1000, batch_size=39, verbose=1)

# 定义适应度函数
def fitness_func(solution, sol_idx):
    learning_rate, filters, kernel_size, epochs = decode(solution)

    model = create_model(learning_rate, filters, kernel_size)
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=epochs, batch_size=39, verbose=1)

    y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).ravel()
    mse = mean_squared_error(y_test, y_pred)
    fitness = 1 / (mse + 1e-8)
    return fitness

# 解码染色体
def decode(solution):
    learning_rate = solution[0]
    filters = int(solution[1])
    kernel_size = int(solution[2])
    epochs = int(solution[3])
    return learning_rate, filters, kernel_size, epochs

# 创建模型
def create_model(learning_rate, filters, kernel_size):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


# 创建一个kerasga对象，它封装了神经网络模型和遗传算法（GA）对象。
keras_ga = pygad.kerasga.KerasGA(model=create_model(0.001, 64, 3), num_solutions=10)

# 创建一个GA实例，指定需要迭代50代，每代的种群大小为10，每个个体有4个基因
# 每个基因都有其自己的取值范围。此外，还设置了父代选择，交叉和变异类型以及一些其他参数。
num_generations = 50
ga_instance = pygad.GA(num_generations=50,
                       num_parents_mating=10,
                       fitness_func=fitness_func,
                       sol_per_pop=20,
                       num_genes=4,
                       gene_space=[{'low': 0.001, 'high': 0.1},  # 学习率
                                   {'low': 16, 'high': 128},      # 卷积核数量
                                   {'low': 2, 'high': 5},         # 卷积核大小
                                   {'low': 50, 'high': 1000}],   # 训练轮数
                       # init_range_low=[0.0001, 16, 1, 100],
                       # init_range_high=[0.1, 128, 7, 2500],
                       parent_selection_type="rws",
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10,
                       keep_parents=5)                              # 添加精英策略，保留每代中最优秀的2个个体



# 运行遗传算法
gen_count = 0  # 计数器，记录迭代次数
for gen in range(num_generations):
    ga_instance.run()
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    gen_count += 1  # 每迭代一次，计数器加1
    remaining_generations = num_generations - gen_count  # 计算剩余迭代次数
    print("Generation {0} of {1}. Best solution: {2}, fitness: {3}. Remaining generations: {4}".format(gen+1, num_generations, best_solution, best_solution_fitness, remaining_generations))


# 获取最佳解
best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
print("Best solution: {0}, fitness: {1}".format(best_solution, best_solution_fitness))

# 解码最佳解并训练最佳模型
best_learning_rate, best_filters, best_kernel_size, best_epochs = decode(best_solution)

best_model = create_model(best_learning_rate, best_filters, best_kernel_size)
best_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=best_epochs, batch_size=32, verbose=1)

# 在测试集上进行评估
y_pred = scaler_y.inverse_transform(best_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).reshape(-1, 1)).ravel()
y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  # 计算 MAE
print("Optimized Model - R-squared: {:.6f}, Mean Squared Error: {:.6f}, Mean Absolute Error: {:.6f}".format(r2, mse, mae))

# 将预测值和实际值保存到一个新的pandas DataFrame中
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# 将结果DataFrame保存到Excel文件中
results_df.to_excel('神经网络数据.xlsx', index=False)