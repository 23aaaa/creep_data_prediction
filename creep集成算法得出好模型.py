import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 读取csv文件并划分训练集和测试集
df = pd.read_csv(r'C:/Users/10239/Desktop/creepdata.csv')
X = df.drop(columns=['y'])
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义不同的回归模型
models = [MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=1000, random_state=42),
          SVR(kernel='rbf', C=1.0, epsilon=0.1),
          DecisionTreeRegressor(max_depth=5, random_state=42),
          RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
          AdaBoostRegressor(n_estimators=10, random_state=42),
          GradientBoostingRegressor(n_estimators=10, max_depth=5, random_state=42),
          KNeighborsRegressor(n_neighbors=5),
          BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=5), n_estimators=10, random_state=42),
          AdaBoostRegressor(n_estimators=10, random_state=42),
          LinearRegression()]

# 遍历不同的回归模型，并对模型性能进行评估
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'Model: {type(model).__name__} \t R2 Score: {r2:.3f} \t MAE: {mae:.3f} \t RMSE: {rmse:.3f}')
