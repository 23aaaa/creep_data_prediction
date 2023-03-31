import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from deap import base, creator, tools, algorithms
import random
import math

# 将数据集读取到 DataFrame 中
df = pd.read_csv(r'C:/Users/10239/Desktop/creepdata.csv')

# 将 DataFrame 分为输入特征 X 和目标值 y
X = df.drop('y', axis=1)
y = df['y']

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# 使用默认参数创建一个随机森林回归器
rf = RandomForestRegressor()

# 使用训练数据拟合模型
rf.fit(X_train, y_train)


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
mse_train, mse_test, mae_train, mae_test, rmse_train, rmse_test, r2_train, r2_test = evaluate_model(rf, X_train, X_test,
                                                                                                    y_train, y_test)

print(f'Random Forest Train MSE: {mse_train}')
print(f'Random Forest Train MAE: {mae_train}')
print(f'Random Forest Train RMSE: {rmse_train}')
print(f'Random Forest Train R2: {r2_train}')
print(f'Random Forest Test MSE: {mse_test}')
print(f'Random Forest Test MAE: {mae_test}')
print(f'Random Forest Test RMSE: {rmse_test}')
print(f'Random Forest Test R2: {r2_test}')


# 定义适应度函数
def eval_individual(individual):
    n_estimators, max_depth, min_samples_split, max_features = individual
    model = RandomForestRegressor(n_estimators=int(n_estimators),
                                  max_depth=int(max_depth),
                                  min_samples_split=int(min_samples_split),
                                  max_features=int(max_features),
                                  random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
toolbox.register('n_estimators', random.randint, 10, 1000)
toolbox.register('max_depth', random.randint, 2, 100)
toolbox.register('min_samples_split', random.randint, 2, 10)
toolbox.register('max_features', random.randint, 1, X_train.shape[1])

# 注册创建函数
toolbox.register('individual', tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.max_depth, toolbox.min_samples_split, toolbox.max_features), n=1)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# 注册评估和遗传操作
toolbox.register('evaluate', eval_individual)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutUniformInt, low=[10, 2, 2, 1], up=[1000, 100, 10, X_train.shape[1]], indpb=0.1)
toolbox.register('select', tools.selTournament, tournsize=3)

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
def simulated_annealing(pop, temp):
    best_ind = min(pop, key=lambda ind: ind.fitness.values)
    new_pop = []

    for ind in pop:
        for _ in range(ITERATIONS_PER_TEMP):
            # 随机选择一个个体进行变异
            mutant = toolbox.clone(ind)
            toolbox.mutate(mutant, low=[10, 2, 2, 1], up=[1000, 100, 10, X_train.shape[1]], indpb=0.1)
            mutant.fitness.values = toolbox.evaluate(mutant)

            # 计算接受概率
            delta = ind.fitness.values[0] - mutant.fitness.values[0]
            if delta > 0:
                ind = mutant
            else:
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    ind = mutant
        new_pop.append(ind)

    return new_pop


# 定义遗传算法与模拟退火算法相结合的方法
def combined_ga_sa(pop, toolbox, ngen=100, temp_start=1000, temp_end=0.1, alpha=0.99, iterations_per_temp=10, stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 更新适应度
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # 更新进化记录
    if halloffame is not None:
        halloffame.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

        # 开始进化
        for gen in range(1, ngen + 1):
            # 选择和克隆下一代
            offspring = toolbox.select(pop, len(pop))
            offspring = list(toolbox.clone(ind) for ind in offspring)

            # 应用交叉和突变
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 应用模拟退火算法
            temp = temp_start * (alpha ** gen)
            for i, ind in enumerate(offspring):
                sa_ind = toolbox.clone(ind)
                toolbox.mutate(sa_ind)

                ind_fitness = toolbox.evaluate(ind)
                sa_ind_fitness = toolbox.evaluate(sa_ind)
                if sa_ind_fitness <= ind_fitness:
                    offspring[i] = sa_ind
                    del ind.fitness.values
                else:
                    delta_e = sa_ind_fitness[0] - ind_fitness[0]
                    if random.random() < np.exp(-delta_e / temp):
                        offspring[i] = sa_ind
                        del ind.fitness.values

            # 更新适应度
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 更新统计信息
            if stats is not None:
                record = stats.compile(pop)
                logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # 更新hall of fame
            if halloffame is not None:
                halloffame.update(pop)

            # 将下一代替换为当前代
            pop[:] = offspring

            # 输出统计信息
            if verbose:
                print(logbook.stream)

        return pop, logbook

# 使用遗传算法和模拟退火算法相结合的方法进行参数调整
pop, log = combined_ga_sa(pop, toolbox, ngen=params['NGEN'], temp_start=params['TEMP_START'], temp_end=params['TEMP_END'], alpha=params['ALPHA'], iterations_per_temp=params['ITERATIONS_PER_TEMP'], stats=stats, halloffame=hof, verbose=True)

# 输出最佳参数
best_individual = hof[0]
best_n_estimators, best_max_depth, best_min_samples_split, best_max_features = best_individual
print(f'Best parameters found: n_estimators={best_n_estimators}, max_depth={best_max_depth}, min_samples_split={best_min_samples_split}, max_features={best_max_features}')

# 使用找到的最佳参数创建和评估新的随机森林模型
best_rf = RandomForestRegressor(n_estimators=int(best_n_estimators),
                                max_depth=int(best_max_depth),
                                min_samples_split=int(best_min_samples_split),
                                max_features=int(best_max_features),
                                random_state=42)
best_rf.fit(X_train, y_train)

mse_train_best, mse_test_best, mae_train_best, mae_test_best, rmse_train_best, rmse_test_best, r2_train_best, r2_test_best = evaluate_model(best_rf, X_train, X_test, y_train, y_test)


print(f'Optimized Random Forest Train MSE: {mse_train_best}')
print(f'Optimized Random Forest Train MAE: {mae_train_best}')
print(f'Optimized Random Forest Train RMSE: {rmse_train_best}')
print(f'Optimized Random Forest Train R2: {r2_train_best}')
print(f'Optimized Random Forest Test MSE: {mse_test_best}')
print(f'Optimized Random Forest Test MAE: {mae_test_best}')
print(f'Optimized Random Forest Test RMSE: {rmse_test_best}')
print(f'Optimized Random Forest Test R2: {r2_test_best}')


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