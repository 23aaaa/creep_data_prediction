import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from deap import base, creator, tools, algorithms
import random
import math

# 读取数据集到DataFrame
df = pd.read_csv(r'C:/Users/10239/Desktop/creepdata.csv')

# 将 DataFrame 分为输入特征 X 和目标值 y
X = df.drop('y', axis=1)
y = df['y']

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# 使用默认参数创建一个XGBoost回归器
xgb = XGBRegressor()

# 使用训练数据拟合模型
xgb.fit(X_train, y_train)

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
mse_train, mse_test, mae_train, mae_test, rmse_train, rmse_test, r2_train, r2_test = evaluate_model(xgb, X_train, X_test, y_train, y_test)

print(f'XGBoost Train MSE: {mse_train}')
print(f'XGBoost Train MAE: {mae_train}')
print(f'XGBoost Train RMSE: {rmse_train}')
print(f'XGBoost Train R2: {r2_train}')
print(f'XGBoost Test MSE: {mse_test}')
print(f'XGBoost Test MAE: {mae_test}')
print(f'XGBoost Test RMSE: {rmse_test}')
print(f'XGBoost Test R2: {r2_test}')


def eval_individual(individual):
    n_estimators, max_depth, learning_rate, min_child_weight, gamma = individual

    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                         min_child_weight=int(min_child_weight), gamma=gamma, objective='reg:squarederror')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if np.isnan(y_pred).any():  # 检查 y_pred 是否包含 NaN 值
        mse = 1e10  # 如果包含 NaN 值，分配较差的适应度值（有限值）

    else:
        mse = mean_squared_error(y_test, y_pred)

    return mse,


# 定义遗传算法参数
POPULATION_SIZE = 80
NGEN = 200
CXPB = 0.7
MUTPB = 0.2

# 模拟退火算法所需的参数
TEMP_START = 1000
TEMP_END = 0.1
ALPHA = 0.98
ITERATIONS_PER_TEMP = 100

# 创建一个名为 params 的字典，其中包含所有需要调整的参数及其默认值
params = {
    'POPULATION_SIZE': 80,
    'NGEN': 200,
    'CXPB': 0.7,
    'MUTPB': 0.2,
    'TEMP_START': 1000,
    'TEMP_END': 0.1,
    'ALPHA': 0.98,
    'ITERATIONS_PER_TEMP': 100
}

# 创建遗传算法中的类型
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# 定义遗传算法操作
toolbox.register('n_estimators', random.randint, 1, 3000)
toolbox.register('max_depth', random.randint, 1, 100)
toolbox.register('learning_rate', random.uniform, 0.001, 0.3)
toolbox.register('min_child_weight', random.randint, 2, 35)
toolbox.register('gamma', random.uniform, 0, 0.9)

# 注册创建函数
toolbox.register('individual', tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.max_depth, toolbox.learning_rate, toolbox.min_child_weight, toolbox.gamma), n=1)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def mutate_float(individual, low, up, indpb):
    for i, (lo, hi) in enumerate(zip(low, up)):
        if np.random.random() < indpb:
            individual[i] = np.random.uniform(lo, hi)
            individual[i] = np.clip(individual[i], lo, hi)  # 限制值在指定范围内
    return individual,


def mutate_int(individual, low, up, indpb):
    for i, (lo, hi) in enumerate(zip(low, up)):
        if random.random() < indpb:
            individual[i] = random.randint(lo, hi)
    return individual,

# 注册评估和遗传操作
# 注册变异操作
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate_individual', mutate_float, low=[1, 1, 0.01, 2, 0], up=[3000, 100, 0.3, 35, 0.9], indpb=0.1)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_individual)


# 固定，使结果可以复现
random.seed(42)

# 初始化种群
pop = toolbox.population(n=POPULATION_SIZE)

# 定义stats，与hof的值
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('min', np.min)
stats.register('max', np.max)
hof = tools.HallOfFame(1)

# 定义模拟退火算法
def simulated_annealing(individual, temp, toolbox, n_iterations=100):
    best_ind = toolbox.clone(individual)
    best_fit = individual.fitness.values

    for _ in range(n_iterations):
        mutant = toolbox.clone(individual)
        mutant, = toolbox.mutate(mutant)
        mutant = creator.Individual(mutant)

        if mutant[2] < 0:  # 检查 learning_rate 是否为负数
            mutant[2] = 0  # 将 learning_rate 设置为 0（或者您认为合适的最小值）

        if mutant[3] < 0:  # 检查 min_split_loss 是否为负数
            mutant[3] = 0  # 将 min_split_loss 设置为 0（或者您认为合适的最小值）

        # Check if learning rate is within the valid range
        learning_rate = mutant[2]
        if learning_rate < 0:
            print("Invalid learning rate after mutation:", learning_rate)
            print("Original individual:", individual)
            print("Mutated individual:", mutant)

        mutant.fitness.values = toolbox.evaluate(mutant)  # 计算变异体的适应度值
        delta_fit = mutant.fitness.values[0] - individual.fitness.values[0]
        if delta_fit < 0 or np.random.random() < np.exp(-delta_fit / temp):
            individual = mutant
            individual.fitness.values = mutant.fitness.values

        if mutant.fitness.values < best_fit:
            best_ind = mutant
            best_fit = mutant.fitness.values

    best_ind.fitness.values = best_fit  # 在返回 best_ind 之前赋予其适应度值
    return best_ind


# 定义遗传算法与模拟退火算法相结合的方法
def combined_ga_sa(population, toolbox, ngen, temp_start, temp_end, alpha, iterations_per_temp, stats=None,
                   halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 计算无效适应度个体的适应度
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # 开始迭代进化过程
    for gen in range(1, ngen + 1):
        # 选择下一代个体
        offspring = toolbox.select(population, len(population))

        # 对个体进行变异
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.2)

        # 确保所有个体都有适应度值
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # 对后代应用模拟退火
        temp = temp_start * (temp_end / temp_start) ** (gen / ngen)
        offspring = [simulated_annealing(ind, temp, toolbox, n_iterations=iterations_per_temp) for ind in offspring]

        # 计算无效适应度后代的适应度
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 用生成的个体更新名人堂
        if halloffame is not None:
            halloffame.update(offspring)

        # 用后代替换旧的种群
        population[:] = offspring

        # 将当前代的统计信息添加到日志中
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# 使用遗传算法和模拟退火算法相结合的方法进行参数调整
pop, log = combined_ga_sa(pop, toolbox, ngen=params['NGEN'], temp_start=params['TEMP_START'], temp_end=params['TEMP_END'], alpha=params['ALPHA'], iterations_per_temp=params['ITERATIONS_PER_TEMP'], stats=stats, halloffame=hof, verbose=True)

# 输出最佳参数
best_individual = hof[0]
best_n_estimators, best_max_depth, best_learning_rate, best_min_child_weight, best_gamma = best_individual
print(f'Best parameters found: n_estimators={best_n_estimators}, max_depth={best_max_depth}, learning_rate={best_learning_rate}, min_child_weight={best_min_child_weight}, gamma={best_gamma}')

# 使用找到的最佳参数创建和评估新的XGBoost模型
best_xgb = XGBRegressor(n_estimators=int(best_n_estimators),
                         max_depth=int(best_max_depth),
                         learning_rate=best_learning_rate,
                         min_child_weight=int(best_min_child_weight),
                         gamma=best_gamma,
                         random_state=42)
best_xgb.fit(X_train, y_train)

mse_train_best, mse_test_best, mae_train_best, mae_test_best, rmse_train_best, rmse_test_best, r2_train_best, r2_test_best = evaluate_model(best_xgb, X_train, X_test, y_train, y_test)


print(f'Best XGBoost model after hyperparameter tuning with genetic algorithm and simulated annealing:')
print(f'Optimized XGBoost Train MSE: {mse_train_best}')
print(f'Optimized XGBoost Train MAE: {mae_train_best}')
print(f'Optimized XGBoost Train RMSE: {rmse_train_best}')
print(f'Optimized XGBoost Train R2: {r2_train_best}')
print(f'Optimized XGBoost Test MSE: {mse_test_best}')
print(f'Optimized XGBoost Test MAE: {mae_test_best}')
print(f'Optimized XGBoost Test RMSE: {rmse_test_best}')
print(f'Optimized XGBoost Test R2: {r2_test_best}')


# 此函数将根据 logbook 数据绘制平均、最小和最大适应度值随代数的变化图。这将帮助你可视化算法在搜索参数空间时的表现。
# 绘制平均适应度值随代数的变化图
def plot_avg(logbook):
    gen = logbook.select("gen")
    avg = logbook.select("avg")

    fig, ax = plt.subplots()
    ax.plot(gen, avg, "b-", label="Average Fitness")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Average Fitness Values over Generations")

    ax.legend()
    plt.show()

# 绘制最小适应度值随代数的变化图
def plot_min(logbook):
    gen = logbook.select("gen")
    min_ = logbook.select("min")

    fig, ax = plt.subplots()
    ax.plot(gen, min_, "r-", label="Minimum Fitness")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Minimum Fitness Values over Generations")

    ax.legend()
    plt.show()

# 调用新的绘图函数
plot_avg(log)
plot_min(log)