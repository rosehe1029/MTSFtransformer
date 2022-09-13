'''
Author: philosophylato
Date: 2022-09-07 09:09:30
LastEditors: philosophylato
LastEditTime: 2022-09-11 21:59:36
Description: your project
version: 1.0
'''
import argparse
from ast import arg
import torch
import math
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
import os
import sys

#from optuna import TrialState
import optuna

#from get_data import get_mape
#from model import TransformerModel

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from itertools import chain
from sklearn.preprocessing import MinMaxScaler
import torch
from scipy.interpolate import make_interp_spline
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import optuna
from matplotlib import pyplot
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import optuna
import torch
from torch import nn
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
#from get_data import nn_seq
#from args import args_parser
#from util import train, test, get_best_parameters
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

PATH = r'E:\坚果云\workworkwork\多变量时序调研9.6.2022\MTSF\Transformer-Timeseries-Forecasting\model\transformer.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inverse_transform_col(scaler,y,n_col):
    '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='epochs')
    parser.add_argument('--seq_len', type=int, default=7, help='seq len')
    parser.add_argument('--input_size', type=int, default=7, help='input dimension')
    parser.add_argument('--d_model', type=int, default=8, help='input dimension')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--lr', type=float, default=0.0008, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--step_size', type=int, default=1, help='step size')
    parser.add_argument('--gamma', type=float, default=0.25, help='gamma')

    args = parser.parse_args()

    return args

def load_data():
    #df = pd.read_csv('E:\坚果云\workworkwork\多变量时序调研9.6.2022\MTSF\Transformer-Timeseries-Forecasting\data\data.csv', encoding='gbk')
    df=pd.read_csv('E:\坚果云\workworkwork\多变量时序调研9.6.2022\MTSF\数值化_用电负荷_大工业用电.csv',encoding='utf-8')
    df.fillna(df.mean(), inplace=True)
    df=df.iloc[:365,1:]
    print(df.head(5))
    return df

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

scaler = MinMaxScaler(feature_range=(0, 1))
def nn_seq(args):
    seq_len, B, num = args.seq_len, args.batch_size, args.output_size
    print('data processing...')
    dataset = load_data()
    # # 对特征标准化
    values = dataset.values
    # 确保所有数据是浮点数类型
    values = values.astype('float32')
    # # 对特征标准化
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    dataset=scaled
    # 分离出特征和标签
    label = scaled[:, 0]
    # split
    #train = dataset[:int(len(dataset) * 0.7)]
    #val = dataset[int(len(dataset) * 0.7):int(len(dataset) * 0.8)]
    #test = dataset[int(len(dataset) * 0.8):len(dataset)]
    print(len(dataset))
    train = dataset[: 365-7-61]
    val = dataset[365-7-61:]
    test = dataset[365-7-61:]
    test_y=label[593-55+8:]
    print('test',test.shape)
    #m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])
    m,n=0,1
    def process(data1, batch_size, step_size, shuffle):
        seq = []
        train_seq, train_label = generate_pair(data1, data1[:, 0], ts=7)
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label)
        ''' 
        load = data[data.columns[1]]
        data = data.values.tolist()
        #load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(0, len(data) - seq_len - num, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(6):
                    x.append(data[j][c])
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)
            seq.append((train_seq, train_label))
        '''
        for i in range(0, len(data1) - seq_len - num, step_size):
            seq.append((train_seq[i], train_label[i]))

        #train_seq, train_label = generate_pair(data, label, ts=7)
        # print(seq[-1])
        #train_seq = torch.FloatTensor(train_seq)
        #train_label = torch.FloatTensor(train_label)
        
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=num, shuffle=False)

    return Dtr, Val, Dte, m, n,test_y


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs(x -y) / x)
    #return (np.abs(x -y) / x)/len(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()
        self.args = args
        # embed_dim = head_dim * num_heads?
        self.input_fc = nn.Linear(args.input_size, args.d_model)
        self.output_fc = nn.Linear(args.input_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=4,
            dim_feedforward=4 * args.input_size,
            batch_first=True,
            dropout=0.1,
            device=device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.d_model,
            nhead=4,
            dropout=0.1,
            dim_feedforward=4 * args.input_size,
            batch_first=True,
            device=device
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.fc = nn.Linear(args.output_size * args.d_model, args.output_size)
        self.fc1 = nn.Linear(args.seq_len * args.d_model, args.d_model)
        self.fc2 = nn.Linear(args.d_model, args.output_size)

    def forward(self, x):
        # print(x.size())  # (256, 24, 7)
        y = x[:, -self.args.output_size:, :]
        # print(y.size())  # (256, 4, 7)
        x = self.input_fc(x)  # (256, 24, 128)
        x = self.pos_emb(x)   # (256, 24, 128)
        x = self.encoder(x)
        # 不经过解码器
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        out = self.fc2(x)
        # y = self.output_fc(y)   # (256, 4, 128)
        # out = self.decoder(y, x)  # (256, 4, 128)
        # out = out.flatten(start_dim=1)  # (256, 4 * 128)
        # out = self.fc(out)  # (256, 4)

        return torch.abs (out)

def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq, label = seq.to(args.device), label.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)

def print_history(train_loss_,val_loss_):
    plt.plot(train_loss_,color='Orange',label="train loss")
    #print("history.history['loss']",len(history.history['loss']))
    plt.plot(val_loss_,color='b',label="validation loss")
    plt.title('train_validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    #plt.savefig(func_path+'/picture/trainvalloss.png')


def train(args, Dtr, Val, path):
    model = TransformerModel(args).to(args.device)
    loss_function = nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('training...')
    epochs = args.epochs
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    final_val_loss = []
    train_loss_,val_loss_=[],[]
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(epochs):
        train_loss = []
        for batch_idx, (seq, target) in enumerate(Dtr, 0):
            seq, target = seq.to(args.device), target.to(args.device)
            target=target.reshape(-1,1)
            optimizer.zero_grad()
            y_pred = model(seq)
            #print(y_pred.shape)
            #print(target.shape)
            loss = loss_function(y_pred, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        final_val_loss.append(val_loss)
        model.train()
        
        train_loss_.append(np.mean(train_loss))
        val_loss_.append(val_loss )

    print_history(train_loss_,val_loss_)


    state = {'model': best_model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state,path)#,"E:\坚果云\workworkwork\多变量时序调研9.6.2022\MTSF\Transformer-Timeseries-Forecasting\model\")#path)
    #torch.save(best_model,path)
    #print(state['optimizer'])
    return np.mean(final_val_loss)


def test(args, Dte, path, m, n,_):
    print('loading model...')
    model = TransformerModel(args).to(args.device)
    model.load_state_dict(torch.load(path)['model'])
    #model=torch.load(path)#['model']
    model.eval()
    pred = []
    y = []
    for batch_idx, (seq, target) in enumerate(Dte, 0):
        seq = seq.to(args.device)
        target = target.to(args.device)
        with torch.no_grad():
            y_pred = model(seq)
            #target = list(chain.from_iterable(target.tolist()))
            #target = chain.from_iterable(target.tolist())
            y.extend(np.array(target.cpu()))
            #y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            #y_pred = chain.from_iterable(y_pred.data.tolist())
            pred.extend(np.array(y_pred.cpu()))

    y, pred = np.array(y), np.array(pred)
    

    #y = (m - n) * y + n
    #pred = (m - n) * pred + n
    print('y',y.shape)
    #print(y-_)
    print('pred',pred.shape)
    y=inverse_transform_col(scaler,y,n_col=0)
    pred=inverse_transform_col(scaler,pred,n_col=0)
    print('mse:', mean_squared_error(y,pred))
    print('rmse:',np.sqrt(mean_squared_error(y,pred)))
    print('mae:',mean_absolute_error(y,pred))
    print('r2_score:',r2_score(y,pred))
    print('mape:', get_mape(y, pred))
    # plot
    # 画图
    #fig = plt.figure()
    #fig.add_subplot()
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.figure(figsize=(12,5))
    plt.plot(y, 'bo-', label='true value')
    plt.plot(pred, 'ro-', label='predict value')
    x = range(0, 61, 15)
    #plt.xticks(x, ('decisiontree', 'knn', 'mlp', 'naive_bayes', 'svm', 'LDA_bagging', 'RandomForest', 'lstm'))
    #plt.plot(np.arange(len(test)), test,'bo-',label='true value')
    #plt.plot(np.arange(len(predictions_)),predictions_,'ro-',label='predict value')
    plt.xticks(x,pd.date_range('2021-11-1','2021-12-31',freq='15d'),rotation=0)
    plt.title('大工业用电-transformer模型')
    plt.legend(loc='best')
    plt.show()
    ''' 
    x = [i for i in range(1, 56)]#151
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    #y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    plt.plot(x_smooth, y_smooth, 'bo-', ms=1, alpha=0.75, label='true')#, marker='*'

    #y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    y_smooth = make_interp_spline(x, pred)(x_smooth)
    plt.plot(x_smooth, y_smooth, 'ro-', ms=1, alpha=0.75, label='pred')# marker='o',
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    '''

def get_best_parameters(args, Dtr, Val):
    def objective(trial):
        model = TransformerModel(args).to(args.device)
        loss_function = nn.MSELoss().to(args.device)
        optimizer = trial.suggest_categorical('optimizer',
                                              [torch.optim.SGD,
                                               torch.optim.RMSprop,
                                               torch.optim.Adam])(
            model.parameters(), lr=trial.suggest_loguniform('lr', 5e-4, 1e-2))
        print('training...')
        epochs = 10
        val_loss = 0
        for epoch in range(epochs):
            train_loss = []
            for batch_idx, (seq, target) in enumerate(Dtr, 0):
                seq, target = seq.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                y_pred = model(seq)
                loss = loss_function(y_pred, target)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            # validation
            val_loss = get_val_loss(args, model, Val)

            print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
            model.train()

        return val_loss

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(func=objective, n_trials=5)
    pruned_trials = study.get_trials(deepcopy=False,
                                     states=tuple([optuna.trial.TrialState.PRUNED]))
    complete_trials = study.get_trials(deepcopy=False,
                                       states=tuple([optuna.trial.TrialState.COMPLETE]))
    best_trial = study.best_trial
    print('val_loss = ', best_trial.value)
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))


def main():
    args = args_parser()
    Dtr, Val, Dte, m, n,_ = nn_seq(args)
    # get_best_parameters(args, Dtr, Val)
    import time
    T1 = time.time()
    train(args, Dtr, Val, PATH)
    print('train用时',time.time()-T1)
    T2=time.time()
    test(args, Dte, PATH, m, n,_)
    print('test用时',time.time()-T2)

if __name__ == '__main__':
    main()
