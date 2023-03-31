import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# streamlit run D:\PycharmProjects\pythonProject1\网页计算器\app.py

# git remote add origin https://github.com/23aaaa/creep-data-prediction.git

# 固定随机种子
random.seed(42)
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

@st.cache_data
def preprocess_data(df):
    X = df.drop('y', axis=1).values
    y = df['y'].values

    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    return X, y, scaler_X, scaler_y

@st.cache(allow_output_mutation=True)
def train_model(X_train, y_train, epochs, batch_size):
    model = Sequential()
    model.add(Conv1D(filters=91, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001712)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=epochs, batch_size=batch_size,
              verbose=1, validation_split=0.1, callbacks=[early_stopping])
    history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=epochs, batch_size=batch_size,
              verbose=1, validation_split=0.1, callbacks=[early_stopping])
    return model, history



st.title("1D CNN用于Creep数据预测")
st.title("CSV格式数据上传")

uploaded_file = st.file_uploader("上传您的Creep数据CSV文件，仅限CSV格式", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    st.write("数据形状：", df.shape)

    X, y, scaler_X, scaler_y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    epochs = 3000
    batch_size = 32

    if st.button("训练模型"):
        model, history = train_model(X_train, y_train, epochs, batch_size)

        # 数据分布子标题
        st.subheader("数据分布")
        sns.pairplot(df)
        st.pyplot()

        y_pred = scaler_y.inverse_transform(
            model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1)).reshape(-1, 1)).ravel()
        y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        # 预测结果子标题
        st.subheader("预测结果")
        fig1, ax1 = plt.subplots()
        ax1.plot(y_test, label='真实值')
        ax1.plot(y_pred, label='预测值')
        ax1.set_title('预测结果')
        ax1.set_ylabel('y')
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("相关性矩阵")
        corr_matrix = df.corr()
        fig2, ax2 = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        st.subheader("散点图矩阵")
        fig3 = plt.figure()
        sns.pairplot(df)
        st.pyplot(fig3)

        st.subheader("预测残差图")
        residuals = y_test - y_pred
        fig4, ax4 = plt.subplots()
        ax4.scatter(y_pred, residuals)
        ax4.set_xlabel('Predicted Values')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Predicted Values vs Residuals')
        ax4.axhline(y=0, color='r', linestyle='--')
        st.pyplot(fig4)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # 模型性能子标题
        st.subheader("模型性能")
        st.write("R-squared: {:.6f}".format(r2))
        st.write("Mean Squared Error: {:.6f}".format(mse))
        st.write("Mean Absolute Error: {:.6f}".format(mae))

        st.subheader("训练过程")
        fig5, ax5 = plt.subplots()
        ax5.plot(history.history['loss'], label='Training Loss')
        ax5.plot(history.history['val_loss'], label='Validation Loss')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.set_title('Training Loss vs Epochs')
        ax5.legend()
        st.pyplot(fig5)
