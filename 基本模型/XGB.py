import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

params = {
    'n_estimators': 2443,
    'max_depth': 70,
    'learning_rate': 0.2,
    'min_child_weight': 11,
    'gamma': 0.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42
}

# 创建模型对象
model = xgb.XGBRegressor(**params)

# 拟合数据
model.fit(X_train, y_train)


# 训练值评估
train_preds = model.predict(X_train)
r2_train = r2_score(y_train, train_preds)
mae_train = mean_absolute_error(y_train, train_preds)
rmse_train = np.sqrt(mean_squared_error(y_train, train_preds))

# 预测值评估
test_preds = model.predict(X_test)
r2_test = r2_score(y_test, test_preds)
mae_test = mean_absolute_error(y_test, test_preds)
rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))


# 输出评估结果
print("训练集R2：", r2_train)
print("训练集MAE：", mae_train)
print("训练集RMSE：", rmse_train)
print("测试集R2：", r2_test)
print("测试集MAE：", mae_test)
print("测试集RMSE：", rmse_test)
