import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from deap import base, creator, tools, algorithms
import random

# 将数据集读取到 DataFrame 中
df = pd.read_csv(r'C:/Users/10239/Desktop/creepdata.csv')

# 将 DataFrame 分为输入特征 X 和目标值 y
X = df.drop('y', axis=1)
y = df['y']

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用默认参数创建一个 AdaBoostRegressor
ada = AdaBoostRegressor(random_state=42)

# 使用训练数据拟合模型
ada.fit(X_train, y_train)

# 评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    return mse_train, mse_test, mae_train, mae_test, rmse_train, rmse_test, r2_train, r2_test

# 使用测试数据评估模型
mse_train, mse_test, mae_train, mae_test, rmse_train, rmse_test, r2_train, r2_test = evaluate_model(ada, X_train, X_test, y_train, y_test)

print(f'AdaBoostRegressor Train MSE: {mse_train}')
print(f'AdaBoostRegressor Train MAE: {mae_train}')
print(f'AdaBoostRegressor Train RMSE: {rmse_train}')
print(f'AdaBoostRegressor Train R2: {r2_train}')
print(f'AdaBoostRegressor Test MSE: {mse_test}')
print(f'AdaBoostRegressor Test MAE: {mae_test}')
print(f'AdaBoostRegressor Test RMSE: {rmse_test}')
print(f'AdaBoostRegressor Test R2: {r2_test}')

# 定义适应度函数
def eval_individual(individual):
    n_estimators, learning_rate = individual
    model = AdaBoostRegressor(n_estimators=int(n_estimators),
                              learning_rate=float(learning_rate),
                              random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse,

# 定义遗传算法参数
POPULATION_SIZE = 100
NGEN = 20
CXPB = 0.8
MUTPB = 1

# 创建遗传算法中的类型
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 定义遗传算法操作
toolbox.register('n_estimators', random.randint, 10, 100)
toolbox.register('learning_rate', random.uniform, 0.01, 1)

# 注册创建函数
toolbox.register('individual', tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.learning_rate), n=1)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

def custom_mutate(ind, low, up, indpb):
    for i in range(len(ind)):
        if random.random() < indpb:
            if i == 0:
                ind[i] = random.randint(low[i], up[i])
            else:
                ind[i] = random.uniform(low[i], up[i])
    return ind,

# 注册评估和遗传操作
toolbox.register('evaluate', eval_individual)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', custom_mutate, low=[10, 0.01], up=[100, 1], indpb=0.1)
toolbox.register('select', tools.selBest)  # 或使用其他选择算子

# 初始化种群
pop = toolbox.population(n=POPULATION_SIZE)

# 运行遗传算法
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('min', np.min)
stats.register('max', np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

# 输出最佳参数
best_individual = hof[0]
best_n_estimators, best_learning_rate = best_individual
print(f'Best parameters found: n_estimators={best_n_estimators}, learning_rate={best_learning_rate}')

# 使用找到的最佳参数创建和评估新的 AdaBoostRegressor 模型
best_ada = AdaBoostRegressor(n_estimators=int(best_n_estimators),
                             learning_rate=float(best_learning_rate),
                             random_state=42)
best_ada.fit(X_train, y_train)

mse_train_best, mse_test_best, mae_train_best, mae_test_best, rmse_train_best, rmse_test_best, r2_train_best, r2_test_best = evaluate_model(best_ada, X_train, X_test, y_train, y_test)

print(f'Optimized AdaBoostRegressor Train MSE: {mse_train_best}')
print(f'Optimized AdaBoostRegressor Train MAE: {mae_train_best}')
print(f'Optimized AdaBoostRegressor Train RMSE: {rmse_train_best}')
print(f'Optimized AdaBoostRegressor Train R2: {r2_train_best}')
print(f'Optimized AdaBoostRegressor Test MSE: {mse_test_best}')
print(f'Optimized AdaBoostRegressor Test MAE: {mae_test_best}')
print(f'Optimized AdaBoostRegressor Test RMSE: {rmse_test_best}')
print(f'Optimized AdaBoostRegressor Train R2: {r2_test_best}')
