import argparse
import os
import shutil
import sys
import numpy as np
import joblib
from orbitP.model.transformer.DataLoader import orbitPDataset
from orbitP.model.transformer.loss import WeightedMSELoss
from orbitP.model.transformer.model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import logging
from joblib import load
from tqdm import tqdm

from orbitP.model.transformer.util import Pml
from orbitP.script.loadData import get_orbitData_SGP4,get_orbitData_Orekit
from orbitP.script.plot import plot_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)
torch.manual_seed(0)

axis = 0
training_length = 2880
predicting_length = 2880
forecast_window = 1
modelName = "train_50.pth"

dataSGP4Dir = "../../../dataset/dataSGP4/"
dataPOEORBDir = "../../../dataset/dataPOEORB/"
dataOrekitDir = "../../../dataset/dataOrekit/"
savePmlPath = "../../../save/save_loss/"
saveDir = "../../../save/"
scalerPath ="../../../save/save_model/Scaler.joblib"

# dataSGP4Dir = "./dataset/dataSGP4/"
# dataPOEORBDir = "./dataset/dataPOEORB/"
# dataOrekitDir = "./dataset/dataOrekit/"
# savePmlPath = "./save/save_loss/"
# saveDir = "./save/"
# scalerPath ="./save/save_model/Scaler.joblib"

def getL2(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2) ** 2  # L2 范数的平方
    return l2_reg

def inference(test_dataloader, feature_size, k,num_layers,dropout, path_to_save_model, scalerPath, modelName, device,lambda_l2 = 0.0001):
    if os.path.exists(savePmlPath+'pml_all.txt'):
        os.remove(savePmlPath + 'pml_all.txt')
    device = torch.device(device)
    model = Transformer(feature_size=feature_size,k=k,num_layers=num_layers,dropout=dropout).float().to(device)
    model.load_state_dict(torch.load(path_to_save_model+modelName))
    criterion = WeightedMSELoss()
    scaler = load(scalerPath)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        predList = torch.tensor(np.array([]))
        test_bar = tqdm(test_dataloader, total=len(test_dataloader),position=0 ,leave=True)
        for idx, (idx_pre, idx_suf, orbitData_pre, orbitData_suf, training_length, forecast_window) in enumerate(tqdm(test_bar)):
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            test_bar.set_description(f"inference")
            src = orbitData_pre.permute(1, 0, 2)[:, :, :-2].float().to(device)  # torch.Size([288, 4, 5])
            tmp = torch.cat((orbitData_pre.permute(1, 0, 2)[1:,:,:], orbitData_suf.permute(1, 0, 2)), dim=0).float()
            target = tmp.to(device)

            if len(predList) != 0:
                src[:, :, 0] = predList.squeeze(-1)
            pred = model(src, device)  # torch.Size([1xw, 1, 1])
            predList = pred

            loss = criterion(pred.squeeze(-1), target[:,:,0])
            # l2_reg = getL2(model)
            # loss = loss + lambda_l2*l2_reg
            test_bar.set_postfix({"loss": loss.detach().item()})
            test_loss += loss.detach().item()
            pred_error = orbitData_pre.permute(1, 0, 2)[-predicting_length:, :, :].squeeze(-2).clone()  # (288,7)
            pred_error[:, 0] = predList.squeeze().cpu()
            pred_error = scaler.inverse_transform(pred_error)  # (288,7)
            src_error = scaler.inverse_transform(tmp[-predicting_length:, :, :].squeeze(-2))  # (288,7)

            pml = Pml(src_error[:, 0],pred_error[:, 0])
            with open(savePmlPath+"pml_all.txt", "a") as f:
                f.write(f"Step{idx+1}: {pml}\n")

        test_loss = test_loss/len(test_dataloader)
        print(f"loss_avg: {test_loss}")
        print(f"Pml: {pml}")
        plot_error(saveDir,src_error[:,0],pred_error[:,0],"error_all")


def inference_step(test_dataloader, feature_size, k,num_layers,dropout, path_to_save_model, scalerPath, modelName, device,lambda_l2 = 0.0001):
    if os.path.exists(savePmlPath+'pml_step.txt'):
        os.remove(savePmlPath + 'pml_step.txt')
    device = torch.device(device)
    # model = Transformer(feature_size=feature_size,k=k,num_layers=num_layers,dropout=dropout).float().to(device)
    # model.load_state_dict(torch.load(path_to_save_model+modelName))

    criterion = WeightedMSELoss()
    scaler = load(scalerPath)
    test_loss = 0
    # model.eval()
    with torch.no_grad():
        predList = torch.tensor(np.array([])).float().to(device)
        test_bar = tqdm(test_dataloader, total=len(test_dataloader))
        for idx, (idx_pre, idx_suf, orbitData_pre, orbitData_suf, training_length, forecast_window) in enumerate(tqdm(test_bar)):
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            test_bar.set_description(f"inference")
            src = orbitData_pre.permute(1, 0, 2)[:, :, :-2].float().to(device)  # torch.Size([288, 4, 5])
            tmp = torch.cat((orbitData_pre.permute(1, 0, 2)[1:,:,:], orbitData_suf.permute(1, 0, 2)), dim=0).float()
            target = tmp.to(device)
            #
            # if len(predList) != 0:
            #     src[-len(predList):, 0, 0] = predList
            # pred = model(src, device)  # torch.Size([1xw, 1, 1])
            # predList = torch.cat((predList,pred[-1:,0,0]), dim=0)
            # loss = criterion(pred[-1,:,0], target[-1,:,0])
            # # l2_reg = getL2(model)
            # # loss = loss + lambda_l2*l2_reg
            # test_bar.set_postfix({"loss": loss.detach().item()})
            # test_loss += loss.detach().item()
            # pred_error = orbitData_pre.permute(1, 0, 2)[-predicting_length:, :, :].squeeze(-2).clone()  # (288,7)
            # pred_error[-len(predList):, 0] = predList.cpu()
            # pred_error = scaler.inverse_transform(pred_error)  # (288,7)
            src_error = scaler.inverse_transform(tmp[-predicting_length:, :, :].squeeze(-2))  # (288,7)
            #
            # pml = Pml(src_error[:, 0],pred_error[:, 0])
            # with open(savePmlPath+"pml_step.txt", "a") as f:
            #     f.write(f"Step{idx+1}: {pml}\n")

        test_loss = test_loss/len(test_dataloader)
        print(f"loss_avg: {test_loss}")
        # print(f"Pml: {pml}")
        pred_error = torch.zeros((2880,8))
        plot_error(saveDir,src_error[:,0],pred_error[:,0],"error_step")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--feature_size", type=int, default=6)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--lambda_l2", type=float, default=0.000001)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--path_to_save_dir", type=str, default=saveDir)
    parser.add_argument("--path_to_save_model",type=str,default=saveDir+"save_model/")
    parser.add_argument("--path_to_save_loss",type=str,default=saveDir+"save_loss/")
    parser.add_argument("--path_to_save_predictions",type=str,default=saveDir+"save_predictions/")
    args = parser.parse_args()
    scalerPath = args.path_to_save_model + "Scaler.joblib"
    if torch.cuda.is_available():
        # 使用 CUDA 设备
        device = "cuda"
    else:
        # 使用 CPU 设备
        device = "cpu"
    # 加载scaler
    scaler = joblib.load(scalerPath)
    # orbitData_SGP4 = get_orbitData_SGP4(dataSGP4Dir,dataPOEORBDir)
    orbitData_Orekit = get_orbitData_Orekit(dataOrekitDir,dataPOEORBDir)
    orbitData = orbitData_Orekit
    orbitData = np.array(orbitData)


    orbitData[:, :, axis] = scaler.transform(orbitData[:, :, axis])
    orbitData_train = orbitData[:-predicting_length]
    orbitData_test = orbitData[-predicting_length-predicting_length:]

    test_dataset = orbitPDataset(data= orbitData_test, axis= axis, training_length = training_length, forecast_window = forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # inference(test_dataloader,args.feature_size,args.k,args.num_layers,args.dropout,args.path_to_save_model,scalerPath,modelName,device,lambda_l2=args.lambda_l2)
    inference_step(test_dataloader,args.feature_size,args.k,args.num_layers,args.dropout,args.path_to_save_model,scalerPath,modelName,device,lambda_l2=args.lambda_l2)
