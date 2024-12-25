import argparse
import os
import random
import sys
import time

import joblib
import torch
import tqdm

import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from orbitP.model.transformer.DataLoader import orbitPDataset
from orbitP.model.transformer.util import clean_directory
from orbitP.model.transformer.train import transformer
from orbitP.model.transformer.inference import inference
from orbitP.script.loadData import get_orbitData_SGP4,get_orbitData_Orekit

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print('torch.cuda.is_available=' + str(torch.cuda.is_available()))
torch.set_default_tensor_type(torch.FloatTensor)

axis = 0
training_length = 2880
predicting_length = 2880
forecast_window = 1

# dataSGP4Dir = "../../dataset/dataSGP4/"
# dataPOEORBDir = "../../dataset/dataPOEORB/"
# dataOrekitDir = "../../dataset/dataOrekit/"
# saveDir = "../../save/"

dataSGP4Dir = "./dataset/dataSGP4/"
dataPOEORBDir = "./dataset/dataPOEORB/"
dataOrekitDir = "./dataset/dataOrekit/"
saveDir = "./save/"

def main( epoch: int = 1000,
    k: int = 3,
    feature_size: int = 6,
    batch_size: int = 64,
    frequency: int = 100,
    lambda_l2: float = 0.01,
    num_layers: int = 3,
    dropout: float = 0.1,
    training_length = 8640,
    forecast_window = 1,
    path_to_save_dir = "../../save/",
    path_to_save_model = "save_model/",
    path_to_save_loss = "save_loss/",
    path_to_save_predictions = "save_predictions/",
    device = "cpu"
):
    clean_directory(path_to_save_dir)
    # orbitData_SGP4 = get_orbitData_SGP4(dataSGP4Dir,dataPOEORBDir)
    orbitData_Orekit = get_orbitData_Orekit(dataOrekitDir,dataPOEORBDir)
    orbitData = orbitData_Orekit
    # 归一化并保存scaler
    scaler = MinMaxScaler()
    scaler.fit(orbitData[:,:,axis])
    orbitData[:,:,axis] = scaler.transform(orbitData[:,:,axis])
    joblib.dump(scaler, path_to_save_model +'Scaler.joblib')
    orbitData_train = orbitData[:-predicting_length]
    orbitData_test = orbitData[-predicting_length-predicting_length:]
    print(f"Train: {orbitData_train.shape}")
    print(f"Test: {orbitData_test.shape}")
    train_dataset = orbitPDataset(data= orbitData_train, axis= axis, training_length = training_length, forecast_window = forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = orbitPDataset(data= orbitData_test, axis= axis, training_length = training_length, forecast_window = forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = transformer(train_dataloader, test_dataloader, epoch, feature_size, k,num_layers,dropout, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--feature_size", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_l2", type=float, default=0.0000001)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_dir", type=str, default=saveDir)
    parser.add_argument("--path_to_save_model",type=str,default=saveDir+"save_model/")
    parser.add_argument("--path_to_save_loss",type=str,default=saveDir+"save_loss/")
    parser.add_argument("--path_to_save_predictions",type=str,default=saveDir+"save_predictions/")
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
    dropout=args.dropout,
    training_length=training_length,
    forecast_window=forecast_window,
    path_to_save_dir=args.path_to_save_dir,
    path_to_save_model=args.path_to_save_model,
    path_to_save_loss=args.path_to_save_loss,
    path_to_save_predictions=args.path_to_save_predictions,
    device=device,
)
