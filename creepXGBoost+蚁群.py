import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

# 定义 XGBoost 回归器评估函数
def evaluate_xgb(params):
    xgb = XGBRegressor(**params)
    scores = cross_val_score(xgb, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    return np.mean(np.sqrt(-scores))

# 使用默认参数创建 XGBoost 回归器，并在训练集上拟合模型
xgb_default = XGBRegressor()
xgb_default.fit(X_train, y_train)

# 在训练集上评估模型表现
y_pred_train_default = xgb_default.predict(X_train)
mse_train_default = mean_squared_error(y_train, y_pred_train_default)
mae_train_default = mean_absolute_error(y_train, y_pred_train_default)
r2_train_default = r2_score(y_train, y_pred_train_default)
rmse_train_default = np.sqrt(mse_train_default)
print('Default XGBoost Model Evaluation (Train Set):')
print('R2:', r2_train_default)
print('MAE:', mae_train_default)
print('RMSE:', rmse_train_default)


# 在测试集上评估默认参数的模型表现
y_pred_default = xgb_default.predict(X_test)
mse_default = mean_squared_error(y_test, y_pred_default)
mae_default = mean_absolute_error(y_test, y_pred_default)
r2_default = r2_score(y_test, y_pred_default)
rmse_default = np.sqrt(mse_default)
print('Default XGBoost Model Evaluation (Test Set):')
print('R2:', r2_default)
print('MAE:', mae_default)
print('RMSE:', rmse_default)

class AntColonyOptimizer:
    def __init__(self, evaluate_func, params_ranges, heuristic_matrix, ant_count=50, generations=50, alpha=1, beta=1, rho=0.5, q=100, max_no_improvement=3):
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

    def _initialize_pheromone_matrix(self):
        pheromone_matrix = {}
        for param_name, param_range in self.params_ranges.items():
            pheromone_matrix[param_name] = np.ones(len(param_range))
        return pheromone_matrix

    def _select_param_value(self, pheromone_values, heuristic_values):
        probabilities = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        probabilities /= probabilities.sum()
        return np.random.choice(np.arange(len(pheromone_values)), p=probabilities)

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
    'min_child_weight': np.ones(34),
    'gamma': np.ones(91)
}

# 定义 ACO 优化器
optimizer = AntColonyOptimizer(evaluate_xgb,
                                {'n_estimators': list(range(1, 3001)),
                                 'max_depth': list(range(1, 101)),
                                 'learning_rate': np.linspace(0.001, 0.3, num=300),
                                 'min_child_weight': list(range(2, 36)),
                                 'gamma': np.linspace(0, 0.9, num=91)},
                               heuristic_matrix,
                                ant_count=35,
                                generations=50,
                                alpha=1,
                                beta=2,
                                rho=0.2,
                                q=80,
                                max_no_improvement=3)


# 运行优化器，得到最优参数组合
best_params, best_score = optimizer.optimize()

# 输出最优参数组合和对应的模型表现
print('Best parameters:', best_params)
print('Best score:', best_score)

# 使用最优参数创建一个新的 XGBoost 回归器，并在训练集上拟合模型
xgb_best = XGBRegressor(**best_params)
eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_best.fit(X_train, y_train, eval_metric='rmse', eval_set=eval_set, verbose=False)

# 在训练集上评估模型表现
y_pred_train = xgb_best.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
print('Train set evaluation:')
print('R2:', r2_train)
print('MAE:', mae_train)
print('RMSE:', rmse_train)

# 在测试集上评估模型表现
y_pred = xgb_best.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print('Test set evaluation:')
print('R2:', r2)
print('MAE:', mae)
print('RMSE:', rmse)


# 绘制迭代曲线
iterations = np.arange(1, xgb_best.n_estimators + 1)

def plot_rmse(iterations, train_rmse, test_rmse):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('RMSE')
    ax.set_title('XGBoost RMSE vs Iterations')
    ax.plot(iterations, train_rmse, label='Train')
    ax.plot(iterations, test_rmse, label='Test')
    ax.legend()
    plt.show()

train_rmse_list = xgb_best.evals_result()['validation_0']['rmse']
test_rmse_list = xgb_best.evals_result()['validation_1']['rmse']

plot_rmse(iterations, train_rmse_list, test_rmse_list)
