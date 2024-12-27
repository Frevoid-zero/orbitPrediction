import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss(path_to_save, train=True):
    plt.rcParams.update({'font.size': 10})
    if train:
        with open(path_to_save+"save_loss/"+"train_loss.txt", 'r') as f:
            loss_list = [float(line.strip().split(": ")[1]) for line in f.readlines()]
    else:
        with open(path_to_save +"save_loss/"+"test_loss.txt", 'r') as f:
            loss_list = [float(line.strip().split(": ")[1]) for line in f.readlines()]
    if train:
        title = "Train"
    else:
        title = "Test"
    plt.plot(loss_list, label = "loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title+"_loss")
    plt.savefig(path_to_save+f"save_predictions/{title}_loss.png")
    plt.close()

def plot_pml(path_to_save):
    plt.rcParams.update({'font.size': 10})
    with open(path_to_save+"save_loss/"+"train_loss.txt", 'r') as f:
        pml_list = [float(line.strip().split(": ")[1]) for line in f.readlines()]
    x = [i*2 for i in range(1,len(pml_list)+1)]
    title = "Pml rate"
    plt.plot(x, pml_list, label = "loss")
    plt.xlabel("t/min")
    plt.ylabel("Pml/%")
    plt.legend()
    plt.title(title)
    plt.savefig(path_to_save+f"save_predictions/pml_rate.png")
    plt.close()


def plot_error(path_to_save, src, tgt, title):

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 16})

    # connect with last elemenet in src
    # tgt = np.append(src[-1], tgt.flatten())
    # prediction = np.append(src[-1], prediction.flatten())

    # plotting
    x = [i*2 for i in range(1, len(src)+1)]
    plt.plot(x, src, '-', color = 'red', label = 'True Error', alpha=0.8)
    plt.plot(x, tgt, '-', color = 'blue', label = 'Prediction Error', alpha=0.8)

    # 添加网格配置
    plt.grid(visible=True, which='major', linestyle='solid')  # 主网格线
    plt.minorticks_on()  # 打开次刻度
    plt.grid(visible=True, which='minor', linestyle='dashed', alpha=0.5)  # 次网格线
    plt.xlabel("t/min")
    plt.ylabel("error/m")
    plt.legend()
    # save
    plt.savefig(path_to_save+f"save_predictions/{title}.png")
    plt.close()

def plot_training(epoch, path_to_save, src, prediction, sensor_number, index_in, index_tar):

    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))

    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(b=True, which='major', linestyle = '-')
    plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)

    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()

def plot_training_3(epoch, path_to_save, src, sampled_src, prediction, sensor_number, index_in, index_tar):

    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))

    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]
    idx_sampled_src = [i for i in range(len(sampled_src))]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(b=True, which='major', linestyle = '-')
    plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    ## REMOVE DROPOUT FOR THIS PLOT TO APPEAR AS EXPECTED !! DROPOUT INTERFERES WITH HOW THE SAMPLED SOURCES ARE PLOTTED
    plt.plot(idx_sampled_src, sampled_src, 'o-.', color='red', label = 'sampled source', linewidth=1, markersize=10)
    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)
    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()