import argparse
import os
import random
import sys

import joblib
import torch

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from orbitP.script.DataLoader import orbitPDataset
from orbitP.script.util import clean_directory
from orbitP.script.loadData import get_orbitData_Orekit,get_orbitData_Orekit_stlm
from orbitP.script.train import transformer,lstm

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print('torch.cuda.is_available=' + str(torch.cuda.is_available()))
torch.set_default_tensor_type(torch.FloatTensor)

axis = 0
training_length = 720
predicting_length = 720
forecast_window = 1

loadName = 'train_100.pth'
optimName  ='optimizer_100.pth'

dataSGP4Dir = "../../dataset/dataSGP4/"
dataPOEORBDir = "../../dataset/dataPOEORB/"
dataOrekitDir = "../../dataset/dataOrekit/"
saveDir = "../../save/"
loadDir = "../../save/load_model/"

# dataSGP4Dir = "./dataset/dataSGP4/"
# dataPOEORBDir = "./dataset/dataPOEORB/"
# dataOrekitDir = "./dataset/dataOrekit/"
# saveDir = "./save/"
# loadDir = "./save/load_model/"

def main( epoch: int = 1000,
    k: int = 3,
    feature_size: int = 6,
    batch_size: int = 64,
    frequency: int = 100,
    lambda_l2: float = 0.01,
    num_layers: int = 3,
    hidden_dim:int = 1440,
    dropout: float = 0.1,
    training_length:int = 1440,
    forecast_window:int = 1,
    path_to_save_dir = "../../save/",
    path_to_save_model = "save_model/",
    path_to_save_loss = "save_loss/",
    path_to_save_predictions = "save_predictions/",
    path_to_load_model="",
    path_to_load_optimizer="",
    load_epoch=0,
    device = "cpu"
):
    if load_epoch==0:
        clean_directory(path_to_save_dir)
    # orbitData_SGP4 = get_orbitData_SGP4(dataSGP4Dir,dataPOEORBDir)
    orbitData_Orekit = get_orbitData_Orekit_stlm(dataOrekitDir,dataPOEORBDir)
    orbitData = orbitData_Orekit
    # 归一化并保存scaler
    scaler = MinMaxScaler(feature_range=(-1,1))#按列归一化，所以第二维必须表示特征
    scaler.fit(orbitData[:,:,axis])
    orbitData[:,:,axis] = scaler.transform(orbitData[:,:,axis])
    joblib.dump(scaler, path_to_save_model +'Scaler.joblib')
    orbitData_train = orbitData[:-predicting_length]
    orbitData_test = orbitData[-training_length-predicting_length:]
    print(f"Train: {orbitData_train.shape}")
    print(f"Test: {orbitData_test.shape}")
    train_dataset = orbitPDataset(data= orbitData_train, axis= axis, training_length = training_length, forecast_window = forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = orbitPDataset(data= orbitData_test, axis= axis, training_length = training_length, forecast_window = forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # model = transformer(train_dataloader, test_dataloader, epoch, feature_size, k,num_layers,dropout, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)
    model = lstm(train_dataloader, test_dataloader, epoch, feature_size,num_layers,hidden_dim, dropout, path_to_save_model, path_to_save_loss, path_to_save_predictions,path_to_load_model,path_to_load_optimizer,load_epoch, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--feature_size", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lambda_l2", type=float, default=0.000001)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_dir", type=str, default=saveDir)
    parser.add_argument("--path_to_save_model",type=str,default=saveDir+"save_model/")
    parser.add_argument("--path_to_save_loss",type=str,default=saveDir+"save_loss/")
    parser.add_argument("--path_to_save_predictions",type=str,default=saveDir+"save_predictions/")
    parser.add_argument("--path_to_load_model", type=str, default=loadDir+loadName)
    parser.add_argument("--path_to_load_optimizer", type=str, default=loadDir + optimName)
    parser.add_argument("--load_epoch", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        # 使用 CUDA 设备
        device = "cuda"
    else:
        # 使用 CPU 设备
        device = "cpu"


main(
    epoch=args.epoch,
    k = args.k,
    feature_size=args.feature_size,
    batch_size=args.batch_size,
    frequency=args.frequency,
    lambda_l2=args.lambda_l2,
    num_layers=args.num_layers,
    hidden_dim=args.hidden_dim,
    dropout=args.dropout,
    training_length=training_length,
    forecast_window=forecast_window,
    path_to_save_dir=args.path_to_save_dir,
    path_to_save_model=args.path_to_save_model,
    path_to_save_loss=args.path_to_save_loss,
    path_to_save_predictions=args.path_to_save_predictions,
    path_to_load_model=args.path_to_load_model,
    path_to_load_optimizer=args.path_to_load_optimizer,
    load_epoch=args.load_epoch,
    device=device,
)
