import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
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

# 定义适应度函数
def eval_individual(individual):
    n_estimators, max_samples, max_features = individual
    model = BaggingRegressor(n_estimators=int(n_estimators),
                             max_samples=max_samples,
                             max_features=int(max_features),
                             random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse,

# 定义遗传算法参数
POPULATION_SIZE = 200
NGEN = 10
CXPB = 0.8
MUTPB = 0.8

# 创建遗传算法中的类型
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 定义遗传算法操作
toolbox.register('n_estimators', random.randint, 10, 100)
toolbox.register('max_samples', random.uniform, 0.1, 1)
toolbox.register('max_features', random.randint, 1, X.shape[1])

# 注册创建函数
toolbox.register('individual', tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.max_samples, toolbox.max_features), n=1)
toolbox.register('population', tools.initRepeat,list, toolbox.individual)

# 注册评估和遗传操作
toolbox.register('evaluate', eval_individual)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutGaussian, mu=[50, 0.5, X.shape[1]/2], sigma=[20, 0.2, X.shape[1]/4], indpb=0.1)
toolbox.register('select', tools.selTournament, tournsize=3)

def check_bounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max[i]:
                        child[i] = max[i]
                    elif child[i] < min[i]:
                        child[i] = min[i]
            return offspring
        return wrapper
    return decorator

toolbox.decorate("mate", check_bounds([10, 0.1, 1], [100, 1, X.shape[1]]))
toolbox.decorate("mutate", check_bounds([10, 0.1, 1], [100, 1, X.shape[1]]))

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
best_n_estimators, best_max_samples, best_max_features = best_individual
print(f'Best parameters found: n_estimators={best_n_estimators}, max_samples={best_max_samples}, max_features={best_max_features}')

# 使用找到的最佳参数创建和评估新的 BaggingRegressor 模型
best_bagging = BaggingRegressor(n_estimators=int(best_n_estimators),
                                max_samples=best_max_samples,
                                max_features=int(best_max_features),
                                random_state=42)
best_bagging.fit(X_train, y_train)

mse_train_best, mse_test_best, mae_train_best, mae_test_best, rmse_train_best, rmse_test_best, r2_train_best, r2_test_best = evaluate_model(best_bagging, X_train, X_test, y_train, y_test)

print(f'Optimized BaggingRegressor Train MSE: {mse_train_best}')
print(f'Optimized BaggingRegressor Train MAE: {mae_train_best}')
print(f'Optimized BaggingRegressor Train RMSE: {rmse_train_best}')
print(f'Optimized BaggingRegressor Train R2: {r2_train_best}')
print(f'Optimized BaggingRegressor Test MSE: {mse_test_best}')
print(f'Optimized BaggingRegressor Test MAE: {mae_test_best}')
print(f'Optimized BaggingRegressor Test RMSE: {rmse_test_best}')
print(f'Optimized BaggingRegressor Test R2: {r2_test_best}')
