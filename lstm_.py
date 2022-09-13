import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif'

group_1=pd.read_csv(r"E:\坚果云\workworkwork\多变量时序调研9.6.2022\MTSF\数值化_用电负荷_大工业用电.csv",index_col='ds')
''' 
group_1.index=pd.to_datetime(group_1.ds)
group_1=group_1.iloc[:,1:]
# 划分数据集
#train_test_split = int(0.6 * len(label))
'''
def generate_pair(x, y, ts):
    length = len(x)
    start, end = 0, length - ts
    data = []
    label = []
    for i in range(end):
        #print(x.iloc[i: i+ts, :])
        data.append(x[i: i+ts, :])
        #print(y.iloc[i+ts])
        label.append(y[i+ts])
    return np.array(data, dtype=np.float64), np.array(label, dtype=np.float64)

values = group_1.values
# 确保所有数据是浮点数类型
values = values.astype('float32')
# # 对特征标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 分离出特征和标签
data = scaled
label = scaled[:, 0]
'''
values = group_1.values
data = values
label = values[:, 0]
'''
data, label = generate_pair(data, label, ts=7)
print(data.shape)
train_X = data[0: 357]
train_y = label[0: 357]
print(train_X.shape)
print(train_y.shape)
val_X=data[364:484]
val_y=label[364: 484]
print(val_X.shape)
print(val_y.shape)
test_X=data[593-55:]
test_y=label[593-55:]
print(test_X.shape)
print(test_y.shape)
''' 
scaler = MinMaxScaler(feature_range=(0, 1))
train = group_1.loc['2021/1/1':'2021/12/31',:]
train_X,train_y = train.iloc[:,1:],train.iloc[:,0]
train_X= scaler.fit_transform(train_X)
train_X,train_y = generate_pair(train_X, train_y, ts=7)
print(train_X.shape)
print(train_y.shape)
val = group_1.loc['2022/1/1':'2022/4/30',:]
val_X,val_y = val.iloc[:,1:],val.iloc[:,0]
val_X = scaler.fit_transform(val_X)
val_X,val_y = generate_pair(val_X, val_y, ts=7)
print(val_X.shape)
print(val_y.shape)
test = group_1.loc['2022/5/1':,:]
test_X,test_y = test.iloc[:,1:],test.iloc[:,0]
test_X= scaler.fit_transform(test_X)
test_X,test_y = generate_pair(test_X, test_y, ts=7)
print(test_X.shape)
print(test_y.shape)

(357, 7, 6)
(357,)
(113, 7, 6)
(113,)
(109, 7, 6)
(109,)
(593, 7, 7)
(357, 7, 7)
(357,)
(120, 7, 7)
(120,)
(109, 7, 7)
(109,)
''' 



from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
#导入相应的函数库
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np

model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

#  MAPE和SMAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def inverse_transform_col(scaler,y,n_col):
    '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y

# 训练模型

if __name__ == '__main__':
    import time
    T1 = time.time()
    history = model.fit(train_X, train_y, epochs=200, batch_size=64, validation_data=(val_X, val_y), verbose=2, shuffle=True)
    print('train用时',time.time()-T1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()

    # 绘图
    #plt.plot(yhat[train_test_split-50: train_test_split+50])
    #plt.plot(test_y[train_test_split-50: train_test_split+50])
    y_hat0= model.predict(data)

    plt.plot(y_hat0)
    plt.plot(label)
    plt.axvline(x=593-55, c="r", ls="--", lw=2)
    plt.legend(['预测数据', '真实数据', '开始预测'])
    plt.grid()
    plt.legend()
    plt.show()
    # 反归一化
    # 开始预测
    T2=time.time()
    yhat = model.predict(test_X) 
    all=scaler.inverse_transform(values)
    y_test=all[55:,0]
    y=all[:,0]
    y_test=inverse_transform_col(scaler,test_y,n_col=0)
    print('test用时',time.time()-T2)
    #print(y_test)#.shape)
    #y_pred = yhat * np.std(y) + np.mean(y)
    y_pred=inverse_transform_col(scaler,yhat,n_col=0)
    #print(y_pred.shape)
    #print(y_pred)
    plt.figure(figsize=(12,5))
    x = range(0, 55, 15)
    plt.plot(y_pred,'ro-',label='pred')
    plt.plot(y_test,'bo-',label='true')
    plt.xticks(x,pd.date_range('2022-7-1','2022-8-24',freq='15d'),rotation=0)
    plt.title('大工业用电-lstm模型')
    plt.legend()
    plt.show()
    ''' 
    #https://blog.csdn.net/oppo603/article/details/86570450
    print('MSE为：',mean_squared_error(y_test,y_pred))
    print('MSE为(直接计算)：',np.mean((y_test-y_pred)**2))
    print('RMSE为：',np.sqrt(mean_squared_error(y_test,y_pred)))
    print(mean_squared_log_error(y_test,y_pred))
    print(np.mean((np.log(y_test+1)-np.log(y_pred+1))**2))
    print(median_absolute_error(y_test,y_pred))
    print(np.median(np.abs(y_test-y_pred)))
    print(mean_absolute_error(y_test,y_pred))
    print(np.mean(np.abs(y_test-y_pred)))
    print(r2_score(y_test,y_pred))
    print(1-(np.sum((y_test-y_pred)**2))/np.sum((y_test -np.mean(y_test))**2))
    print(mape(y_test, y_pred))
    print(smape(y_test, y_pred))
    '''
    print('mse:', mean_squared_error(y_test,y_pred))
    print('rmse:',np.sqrt(mean_squared_error(y_test,y_pred)))
    print('mae:',mean_absolute_error(y_test,y_pred))
    print('r2_score:',r2_score(y_test,y_pred))
    print('mape:', mape(y_test,y_pred))
    















