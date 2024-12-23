import os
import shutil

import numpy as np
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from tqdm import tqdm

dataPOEORBDir = "../../dataset/dataPOEORB/"
dataSGP4Dir = "../../dataset/dataSGP4/"

dataTime = []
dataPError = []
dataVError = []

def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(os.path.normpath(file_path))
    return file_paths

if __name__ == "__main__":
    dataPOEORBPaths = get_file_paths(dataPOEORBDir)
    dataSGP4Paths = get_file_paths(dataSGP4Dir)

    fileName = os.path.basename(dataSGP4Paths[0])
    dataPOEORBPath = dataPOEORBDir +fileName
    dataSGP4Path = dataSGP4Dir +fileName
    with open(dataPOEORBPath,"r",encoding="utf-8") as filePOEORB:
        with open(dataSGP4Path,"r",encoding="utf-8") as fileSGP4:
            linesPOEORB = filePOEORB.readlines()
            linesSGP4 = fileSGP4.readlines()
            t_bar = tqdm(range(0,len(linesPOEORB),3),total=len(range(0,len(linesPOEORB),3)))
            for i in t_bar:
                time_utc = datetime.strptime(linesPOEORB[i].strip(),"UTC=%Y-%m-%dT%H:%M:%S.%f")
                t_bar.set_description(time_utc.strftime("%H:%M:%S"))
                p_POEORB = linesPOEORB[i+1].strip().split(" ")
                v_POEORB = linesPOEORB[i+2].strip().split(" ")
                p_SGP4 = linesSGP4[i+1].strip().split(" ")
                v_SGP4 = linesSGP4[i+2].strip().split(" ")
                p_error = [float(x) - float(y) for x, y in zip(p_POEORB, p_SGP4)]
                v_error = [float(x) - float(y) for x, y in zip(v_POEORB, v_SGP4)]
                dataTime.append(time_utc)
                dataPError.append(p_error)
                dataVError.append(v_error)


    dataTime = np.array(dataTime)
    dataPError = np.array(dataPError)
    dataVError = np.array(dataVError)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制轨迹数据
    ax.plot(dataTime, dataPError[:,0], label='SGP4误差', color='blue',alpha=0.8)
    ax.set_xlabel('utc_Time /s')
    ax.set_ylabel('error /m')
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=25)
    plt.show()