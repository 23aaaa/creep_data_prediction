import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

# 读取数据集到DataFrame
df = pd.read_csv(r'C:/Users/10239/Desktop/creepdata.csv')

# 将 DataFrame 分为输入特征 X 和目标值 y
X = df.drop('y', axis=1)
y = df['y']

# 数据标准化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义 LGBM 回归器评估函数
def evaluate_lgbm(params):
    lgbm = LGBMRegressor(**params)
    scores = cross_val_score(lgbm, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
    return np.mean(np.sqrt(-scores))

# 使用默认参数创建 LGBM 回归器，并在训练集上拟合模型
lgbm_default = LGBMRegressor()
lgbm_default.fit(X_train, y_train)

# 在训练集上评估模型表现
y_pred_train_default = lgbm_default.predict(X_train)
mse_train_default = mean_squared_error(y_train, y_pred_train_default)
mae_train_default = mean_absolute_error(y_train, y_pred_train_default)
r2_train_default = r2_score(y_train, y_pred_train_default)
rmse_train_default = np.sqrt(mse_train_default)
print('Default LGBM Model Evaluation (Train Set):')
print('R2:', r2_train_default)
print('MAE:', mae_train_default)
print('RMSE:', rmse_train_default)

# 在测试集上评估默认参数的模型表现
y_pred_default = lgbm_default.predict(X_test)
mse_default = mean_squared_error(y_test, y_pred_default)
mae_default = mean_absolute_error(y_test, y_pred_default)
r2_default = r2_score(y_test, y_pred_default)
rmse_default = np.sqrt(mse_default)
print('Default LGBM Model Evaluation (Test Set):')
print('R2:', r2_default)
print('MAE:', mae_default)
print('RMSE:', rmse_default)

# 下面是您的 AntColonyOptimizer 类定义和实例化
class AntColonyOptimizer:
    def __init__(self, evaluate_func, params_ranges, heuristic_matrix, ant_count=50, generations=50, alpha=1, beta=1,
                 rho=0.5, q=100, max_no_improvement=3, seed=None):
        self.evaluate_func = evaluate_func
        self.params_ranges = params_ranges
        self.heuristic_matrix = heuristic_matrix
        self.ant_count = ant_count
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.max_no_improvement = max_no_improvement
        self.seed = seed  # 添加此行以将种子值存储在实例中

    def _initialize_pheromone_matrix(self):
        pheromone_matrix = {}
        for param_name, param_range in self.params_ranges.items():
            pheromone_matrix[param_name] = np.ones(len(param_range))
        return pheromone_matrix

    def _select_param_value(self, pheromone_values, heuristic_values):
        rng = np.random.default_rng(self.seed)
        probabilities = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        probabilities /= probabilities.sum()
        return rng.choice(np.arange(len(pheromone_values)), p=probabilities)

    def _create_solution(self, pheromone_matrix):
        solution = {}
        for param_name, param_range in self.params_ranges.items():
            selected_idx = self._select_param_value(pheromone_matrix[param_name], self.heuristic_matrix[param_name])
            solution[param_name] = param_range[selected_idx]
        return solution

    def _update_pheromone(self, pheromone_matrix, solutions, scores):
        for param_name, param_range in self.params_ranges.items():
            for i, value in enumerate(param_range):
                delta_pheromone = sum(score for solution, score in zip(solutions, scores) if solution[param_name] == value)
                pheromone_matrix[param_name][i] = (1 - self.rho) * pheromone_matrix[param_name][i] + self.q * delta_pheromone

    def optimize(self):
        best_solution = None
        best_score = float('inf')

        historical_best_solution = None
        historical_best_score = float('inf')

        pheromone_matrix = self._initialize_pheromone_matrix()
        no_improvement_counter = 0

        for gen in range(self.generations):
            start_time = time.time()
            solutions = [self._create_solution(pheromone_matrix) for _ in range(self.ant_count)]
            scores = [self.evaluate_func(solution) for solution in solutions]

            min_score = min(scores)
            if min_score < best_score:
                best_score = min_score
                best_solution = solutions[scores.index(min_score)]
                no_improvement_counter = 0

                # 更新历史最佳解
                if best_score < historical_best_score:
                    historical_best_score = best_score
                    historical_best_solution = best_solution
            else:
                no_improvement_counter += 1

            time_elapsed = time.time() - start_time
            print(f"第{gen + 1}次搜索: 本次最优解:{best_solution} 耗时:{time_elapsed:.0f}秒 得分:{min_score:.0f}")

            self._update_pheromone(pheromone_matrix, solutions, scores)

            # 检查是否需要重启
            if no_improvement_counter >= self.max_no_improvement:
                print(f"在第{gen + 1}代重启")
                pheromone_matrix = self._initialize_pheromone_matrix()
                no_improvement_counter = 0

        return historical_best_solution, historical_best_score

heuristic_matrix = {
    'n_estimators': np.ones(3000),
    'max_depth': np.ones(100),
    'learning_rate': np.ones(300),
    'min_child_samples': np.ones(34),
    'num_leaves': np.ones(46)
}

# 定义 ACO 优化器
optimizer = AntColonyOptimizer(evaluate_lgbm,
                                {'n_estimators': list(range(1, 3001)),
                                 'max_depth': list(range(1, 101)),
                                 'learning_rate': np.linspace(0.001, 0.3, 300).tolist(),
                                 'min_child_samples': list(range(2, 36)),
                                 'num_leaves': list(range(5, 51))},
                                heuristic_matrix=heuristic_matrix,
                                ant_count=30,
                                generations=60,
                                rho=0.2,
                                alpha=1,
                                beta=15,
                                q=50,
                                max_no_improvement=10,
                                seed=42)


# 运行优化器，得到最优参数组合
best_params, best_score = optimizer.optimize()

# 输出最优参数组合和对应的模型表现
print('Best parameters:', best_params)
print('Best score:', best_score)

# 使用最优参数创建一个新的 LGBM 回归器，并在训练集上拟合模型
lgbm_best = LGBMRegressor(**best_params)
lgbm_best.fit(X_train, y_train)

# 在训练集上评估最优参数的模型表现
y_pred_train_best = lgbm_best.predict(X_train)
mse_train_best = mean_squared_error(y_train, y_pred_train_best)
mae_train_best = mean_absolute_error(y_train, y_pred_train_best)
r2_train_best = r2_score(y_train, y_pred_train_best)
rmse_train_best = np.sqrt(mse_train_best)
print('Best LGBM Model Evaluation (Train Set):')
print('R2:', r2_train_best)
print('MAE:', mae_train_best)
print('RMSE:', rmse_train_best)

# 在测试集上评估最优参数的模型表现
y_pred_best = lgbm_best.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
print('Best LGBM Model Evaluation (Test Set):')
print('R2:', r2_best)
print('MAE:', mae_best)
print('RMSE:', rmse_best)

# 绘制预测结果与真实结果的对比图
plt.scatter(y_test, y_pred_best)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for Best LGBM Model')
plt.show()
