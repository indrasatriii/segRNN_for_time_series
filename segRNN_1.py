# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:59:27 2025

@author: User1201
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data import load_specific_dataset
from pypots.imputation import SegRNN
from SegRNN import Model as SegRNN
from pypots.utils.metrics import calc_mae
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import time

series =pd.read_csv("D:/xin/segrnn/segRNN_1/df_male_1.csv")
n=0
m=1
def calMAPE(actual, predict):
    sum_up = 0
    size = len(actual)
    for i in range(size):
        diff = actual[i] - predict[i]
        sum_up += abs(diff / actual[i])
    return (sum_up / size) * 100

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        
        a = dataset[i:(i + look_back)]
        #print("a:",a)
        dataX.append(a)
        #print("dataX:",dataX)
        dataY.append(dataset[i + look_back])
        #print("dataY:",dataY)
    return np.array(dataX), np.array(dataY)
series.dropna(inplace = True)
X_temp = series.values
X_temp=X_temp[:,1:]
save=open('segRNN_0_test.txt','w')
for sr in range(n,n+1):#len(X_temp)
    start = time.time()
    save.writelines('第'+str(sr)+str(series.iloc[sr,0])+"類別"+'\n')

    dataset_scaler_temp=X_temp[sr,:]/1000
    dataset_scaler=np.delete(dataset_scaler_temp, np.where(dataset_scaler_temp == 0), axis=0)
    plt.plot(dataset_scaler)
    plt.show()
    train = dataset_scaler[0:-50]
    test = dataset_scaler[-50:]
    look_back = 1
    trainX, trainY= create_dataset(train, look_back)
    testX, testY= create_dataset(test, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1)).astype(np.float32)
    # Data preprocessing. Tedious, but PyPOTS can help.
    #data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
    X = trainX.astype(np.float32)
    
    num_samples = len(X)
    #X = StandardScaler().fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_ori = X  # keep X_ori for validation
    X = mcar(X, 0.1)  # randomly hold out 10% observed values as ground truth
    dataset = {"X": X}  # X for model input
    print(X.shape)  # (7671, 48, 37), 7671 samples, 48 time steps, 37 features

    # initialize the model
    saits = SegRNN(
        n_steps=look_back,
        seg_len=1,
        n_features=1,
        #n_layers=2,
        d_model=2,
        #d_ffn=2,
        #n_heads=4,
        batch_size=1,
        #d_k=12,
        #d_v=12,
        dropout=0.1,
        epochs=1000,
        #saving_path="model/segrnn/4", # set the path for saving tensorboard logging file and model checkpoint
       # model_saving_strategy="best", # only save the model with the best validation performance
    )

    # train the model. Here I consider the train dataset only, and evaluate on it, because ground truth is not visible to the model.
    saits.fit(dataset)
    segrnn_results = saits.predict({"X": testX})
    segrnn_prediction = segrnn_results["imputation"]
    print("segrnn_prediction_shape:",segrnn_prediction.shape)
    # 將 segrnn_prediction 轉換成一維陣列
    segrnn_prediction_1d = segrnn_prediction.squeeze()  # 由 (48, 1, 1) 轉為 (48,)
    
    # 計算 MAPE（平均絕對百分比誤差）
    # 注意：這裡假設 testY 中的數值皆非 0，以免除以 0 的問題
    mape = np.mean(np.abs((testY - segrnn_prediction_1d) / testY)) * 100
    # 計算 RMSE（均方根誤差）
    rmse = np.sqrt(np.mean((testY - segrnn_prediction_1d) ** 2))
    
    print(f"MAPE: {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")
    #評估指標
    rmse_sum = 0
    mape_sum = 0
    rmse_count = 0
    mape_count = 0
    error_mse = mean_squared_error(testY,segrnn_prediction_1d) #MSE & RMSE運算
    rmse_sum += math.sqrt(mean_squared_error(testY,segrnn_prediction_1d))
    error_rmse = math.sqrt(mean_squared_error(testY,segrnn_prediction_1d))
    rmse_count += 1
                
    error_mae = mean_absolute_error(testY,segrnn_prediction_1d) #MAE & MAPE運算 更改predictions變數來獲取不同值
    error_mape = calMAPE(testY,segrnn_prediction_1d)
    print('Test MSE: %.3f' % error_mse)
    print('Test RMSE:%f' % error_rmse)
    print('Test MAE: %.3f' % error_mae)
    print('Test MAPE: %f' % error_mape)
    save.writelines("Test MSE:"+str(error_mse)+'\n'+"Test RMSE:"+str(error_rmse)+'\n'+"Test MAE:"+str(error_mae)+'\n'+"Test MAPE:"+str(error_mape)+'\n')    
    save.writelines('Test:'+str(testY)+'\n'+"predictions:"+str(segrnn_prediction_1d)+'\n')    
    pred = pd.DataFrame(segrnn_prediction_1d)
    pred.to_csv('./predictions/segRNN{}_{}.csv'.format(series.iloc[sr,0],error_mape))
    end = time.time()
    print(f"Training time: {end - start} seconds")
    #預測圖顯示
    plt.plot(testY)
    plt.plot(segrnn_prediction_1d)
    plt.grid()
    ax = pyplot.subplot()
    ax.set_title("seqRNN:{}".format(series.iloc[n,0]),fontproperties="SimSun",fontsize = 20)#="SimHei"
    fig = pyplot.gcf()
    # fig.savefig('./image/LSTM{}.png'.format(series.iloc[3,0]))           
    plt.show() 
save.close()
'''
# 儲存模型至指定路徑
import os
save_path = "model/segrnn_{}_{}.pt".format(series.iloc[sr,0],mape)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
saits.save(save_path)
print(f"模型已儲存至 {save_path}")
'''


'''
#bttf_results = saits.predict(testX)
# impute the originally-missing values and artificially-missing values
imputation = saits.impute(dataset)
# calculate mean absolute error on the ground truth (artificially-missing values)
indicating_mask = np.isnan(X) ^ np.isnan(X_ori)  # indicating mask for imputation error calculation
mae = calc_mae(imputation, np.nan_to_num(X_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
print(f"Testing mean absolute error: {mae:.4f}")
'''