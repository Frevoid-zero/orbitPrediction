import sys

from orbitP.script.loss import WeightedMSELoss
from orbitP.model.transformer import Transformer
from orbitP.model.LSTM import Lstm
import torch
import logging
from tqdm import tqdm
from orbitP.script.util import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def getL2(model):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2) ** 2  # L2 范数的平方
    return l2_reg

def transformer(train_dataloader, test_dataloader, EPOCH, feature_size, k,num_layer,dropout, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device, lambda_l2 = 0.0001):

    device = torch.device(device)
    model = Transformer(feature_size=feature_size,k=k,num_layers=num_layer,dropout=dropout).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
    criterion = WeightedMSELoss(1)
    model.train()
    for epoch in range(EPOCH):
        train_loss = 0
        test_loss = 0
        train_bar = tqdm(train_dataloader,total=len(train_dataloader))
        for idx_pre, idx_suf, orbitData_pre, orbitData_suf, training_length, forecast_window in train_bar:
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            train_bar.set_description(f"train{epoch+1}")
            optimizer.zero_grad()
            src = orbitData_pre.permute(1,0,2)[:,:,:-2].float().to(device) # torch.Size([288, 4, 6])
            target = torch.cat((orbitData_pre.permute(1,0,2)[1:,:,:],orbitData_suf.permute(1,0,2)),dim=0).float().to(device)
            pred= model(src, device) # torch.Size([1xw, 1, 1])
            loss = criterion(pred.squeeze(-1),target[:,:,0])
            # l2_reg = getL2(model)
            # loss = loss + lambda_l2*l2_reg
            loss.backward()
            optimizer.step()

            train_bar.set_postfix({"loss": loss.detach().item()})
            train_loss += loss.detach().item()


        # model.eval()
        # with torch.no_grad():
        #     predList = torch.tensor(np.array([]))
        #     test_bar = tqdm(test_dataloader,total=len(test_dataloader))
        #     for idx_pre, idx_suf, orbitData_pre, orbitData_suf, training_length, forecast_window in test_bar:
        #         # Shape of _input : [batch, input_length, feature]
        #         # Desired input for model: [input_length, batch, feature]
        #         test_bar.set_description(f"test{epoch+1}")
        #         src = orbitData_pre.permute(1,0,2)[:,:,:-2].float().to(device) # torch.Size([288, 4, 6])
        #
        #         target = torch.cat((orbitData_pre.permute(1, 0, 2)[1:,:,:], orbitData_suf.permute(1, 0, 2)), dim=0).float().to(device)
        #
        #         if len(predList)!=0:
        #             src[:,:,0]=predList.squeeze(-1)
        #         pred = model(src, device)  # torch.Size([1xw, 1, 1])
        #         predList = pred
        #
        #         loss = criterion(pred.squeeze(-1),target[:,:,0])
        #         l2_reg = getL2(model)
        #         loss = loss + lambda_l2 * l2_reg
        #         test_bar.set_postfix({"loss": loss.detach().item()})
        #         test_loss += loss.detach().item()
        #
        train_loss /= len(train_dataloader)
        # test_loss /= len(test_dataloader)
        log_loss(epoch,train_loss, path_to_save_loss, train=True)
        # log_loss(epoch,test_loss, path_to_save_loss, train=False)

        if(epoch+1)%5==0:
            if not os.path.exists(path_to_save_model+ f"train_{epoch+1}/"):
                os.makedirs(path_to_save_model+ f"train_{epoch+1}/")
            torch.save(model.state_dict(), path_to_save_model+ f"train_{epoch+1}/" + f"train_{epoch+1}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model+ f"train_{epoch+1}/" + f"optimizer_{epoch+1}.pth")

    if not os.path.exists(path_to_save_model):
        os.makedirs(path_to_save_model)
    torch.save(model.state_dict(), path_to_save_model + f"train_{EPOCH}.pth")
    torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{EPOCH}.pth")
    return model


def lstm(train_dataloader, test_dataloader, EPOCH, feature_size,num_layers,hidden_dim,dropout, path_to_save_model, path_to_save_loss, path_to_save_predictions,path_to_load_model,path_to_load_optimizer, load_epoch, device, lambda_l2 = 0.0001):

    device = torch.device(device)
    model = Lstm(input_dim=feature_size,num_layers=num_layers,hidden_dim=hidden_dim,dropout=dropout,output_dim=1).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    if load_epoch!=0:
        model.load_state_dict(torch.load(path_to_load_model))
        optimizer.load_state_dict(torch.load(path_to_load_optimizer))
        print("load model success!")

    criterion = WeightedMSELoss(1)
    for epoch in range(load_epoch,EPOCH):
        train_loss = 0
        test_loss = 0
        train_bar = tqdm(train_dataloader,total=len(train_dataloader))
        loss = torch.tensor(0.0).float().to(device)
        model.train()
        for idx_pre, idx_suf, orbitData_pre, orbitData_suf, training_length, forecast_window in train_bar:
            # Shape of _input : [batch, input_length, feature]
            train_bar.set_description(f"train{epoch+1}")
            optimizer.zero_grad()
            src = orbitData_pre.float().to(device)
            target = orbitData_suf[:,:,0].float().to(device)

            pred = model(src,device) # torch.Size([1xw, 1])
            loss_now = criterion(pred,target)
            loss += loss_now
            train_bar.set_postfix({"loss": loss_now.detach().item()})

        l2_reg = getL2(model)
        loss = loss/len(train_dataloader) + lambda_l2*l2_reg
        loss.backward()
        optimizer.step()

        # loss.backward()
        # optimizer.step()
        train_loss = loss.detach().item()


        model.eval()
        with torch.no_grad():
            predList = torch.tensor(np.array([])).float().to(device)
            test_bar = tqdm(test_dataloader,total=len(test_dataloader))
            for idx_pre, idx_suf, orbitData_pre, orbitData_suf, training_length, forecast_window in test_bar:
                # Shape of _input : [batch, input_length, feature]
                # Desired input for model: [input_length, batch, feature]
                test_bar.set_description(f"test{epoch+1}")
                src = orbitData_pre.float().to(device)  # torch.Size([4, 288, 6])
                target = orbitData_suf[:, :, 0].float().to(device)

                if len(predList) != 0:
                    src[0, -len(predList):, 0] = predList[:, 0]
                pred = model(src, device)  # torch.Size([1, 1])
                predList = torch.cat((predList, pred), dim=0)
                loss = criterion(pred, target)
                # l2_reg = getL2(model)
                # loss = loss + lambda_l2*l2_reg
                test_bar.set_postfix({"loss": loss.detach().item()})
                test_loss += loss.detach().item()
        #
        # train_loss /= len(train_dataloader)
        test_loss /= len(test_dataloader)
        log_loss(epoch,train_loss, path_to_save_loss, train=True)
        log_loss(epoch,test_loss, path_to_save_loss, train=False)

        if(epoch+1)%5==0:
            if not os.path.exists(path_to_save_model+ f"train_{epoch+1}/"):
                os.makedirs(path_to_save_model+ f"train_{epoch+1}/")
            torch.save(model.state_dict(), path_to_save_model+ f"train_{epoch+1}/" + f"train_{epoch+1}.pth")
            torch.save(optimizer.state_dict(), path_to_save_model+ f"train_{epoch+1}/" + f"optimizer_{epoch+1}.pth")

    if not os.path.exists(path_to_save_model):
        os.makedirs(path_to_save_model)
    torch.save(model.state_dict(), path_to_save_model + f"train_{EPOCH}.pth")
    torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{EPOCH}.pth")
    return model


# def lstm(train_dataloader, test_dataloader, EPOCH, feature_size, num_layers, hidden_dim, dropout, path_to_save_model,
#          path_to_save_loss, path_to_save_predictions, path_to_load_model, path_to_load_optimizer, load_epoch, device,
#          lambda_l2=0.0001):
#     device = torch.device(device)
#     model = Lstm(input_dim=feature_size, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout,
#                  output_dim=1).float().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     if load_epoch != 0:
#         model.load_state_dict(torch.load(path_to_load_model))
#         optimizer.load_state_dict(torch.load(path_to_load_optimizer))
#         print("load model success!")
#
#     criterion = WeightedMSELoss(1)
#
#     for epoch in range(load_epoch, EPOCH):
#         train_loss = 0
#         test_loss = 0
#         train_bar = tqdm(train_dataloader, total=len(train_dataloader))
#         model.train()
#         for idx_pre, idx_suf, orbitData_pre, orbitData_suf, training_length, forecast_window in train_bar:
#             # Shape of _input : [batch, input_length, feature]
#             train_bar.set_description(f"train{epoch + 1}")
#             optimizer.zero_grad()
#             loss = torch.tensor(0.0).float().to(device)
#             total = torch.cat((orbitData_pre, orbitData_suf), dim=1)
#             for i in range(len(orbitData_suf)):
#                 src = total[:, i:i + forecast_window[0], :].float().to(device)
#                 target = total[:, i + forecast_window[0], 0].float().to(device)
#
#                 pred = model(src, device)  # torch.Size([1xw, 1])
#                 loss_now = criterion(pred.squeeze(-1), target)
#                 loss += loss_now
#
#             l2_reg = getL2(model)
#             loss = loss / len(orbitData_suf) + lambda_l2 * l2_reg
#             loss.backward()
#             optimizer.step()
#             train_bar.set_postfix({"loss": loss.detach().item()})
#             train_loss += loss.detach().item()
#
#         # loss.backward()
#         # optimizer.step()
#
#         model.eval()
#         with torch.no_grad():
#             predList = torch.tensor(np.array([])).float().to(device)
#             test_bar = tqdm(test_dataloader, total=len(test_dataloader))
#             for idx_pre, idx_suf, orbitData_pre, orbitData_suf, training_length, forecast_window in test_bar:
#                 # Shape of _input : [batch, input_length, feature]
#                 # Desired input for model: [input_length, batch, feature]
#                 test_bar.set_description(f"test{epoch + 1}")
#                 src = orbitData_pre.float().to(device)  # torch.Size([4, 288, 6])
#                 target = orbitData_suf[:, :, 0].float().to(device)
#
#                 if len(predList) != 0:
#                     src[0, -len(predList):, 0] = predList[:, 0]
#                 pred = model(src, device)  # torch.Size([1, 1])
#                 predList = torch.cat((predList, pred), dim=0)
#                 loss = criterion(pred, target)
#                 # l2_reg = getL2(model)
#                 # loss = loss + lambda_l2*l2_reg
#                 test_bar.set_postfix({"loss": loss.detach().item()})
#                 test_loss += loss.detach().item()
#         #
#         train_loss /= len(train_dataloader)
#         test_loss /= len(test_dataloader)
#         log_loss(epoch, train_loss, path_to_save_loss, train=True)
#         log_loss(epoch, test_loss, path_to_save_loss, train=False)
#
#         if (epoch + 1) % 5 == 0:
#             if not os.path.exists(path_to_save_model + f"train_{epoch + 1}/"):
#                 os.makedirs(path_to_save_model + f"train_{epoch + 1}/")
#             torch.save(model.state_dict(), path_to_save_model + f"train_{epoch + 1}/" + f"train_{epoch + 1}.pth")
#             torch.save(optimizer.state_dict(),
#                        path_to_save_model + f"train_{epoch + 1}/" + f"optimizer_{epoch + 1}.pth")
#
#     if not os.path.exists(path_to_save_model):
#         os.makedirs(path_to_save_model)
#     torch.save(model.state_dict(), path_to_save_model + f"train_{EPOCH}.pth")
#     torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{EPOCH}.pth")
#     return model