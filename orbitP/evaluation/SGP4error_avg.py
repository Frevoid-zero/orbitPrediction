import os
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
    print(dataPOEORBPaths[0])
    with open(dataSGP4Paths[0], "r", encoding="utf-8") as fileSGP4:
        linesSGP4 = fileSGP4.readlines()
        for i in range(0, len(linesSGP4), 3):
            time_utc = datetime.strptime(linesSGP4[i].strip(), "UTC=%Y-%m-%dT%H:%M:%S.%f")
            dataTime.append(time_utc)
            dataPError.append([0, 0, 0])
            dataVError.append([0, 0, 0])
    t_bar = tqdm(dataSGP4Paths,total=len(dataSGP4Paths))
    for dataSGP4Path in t_bar:
        fileName = os.path.basename(dataSGP4Path)
        dataPOEORBPath = dataPOEORBDir +fileName
        t_bar.set_description(f"{fileName.replace('.txt','')}")
        with open(dataPOEORBPath,"r",encoding="utf-8") as filePOEORB:
            with open(dataSGP4Path,"r",encoding="utf-8") as fileSGP4:
                linesPOEORB = filePOEORB.readlines()
                linesSGP4 = fileSGP4.readlines()
                for idx,i in enumerate(range(0,len(linesPOEORB),3)):
                    time_utc = datetime.strptime(linesPOEORB[i].strip(),"UTC=%Y-%m-%dT%H:%M:%S.%f")
                    p_POEORB = linesPOEORB[i+1].strip().split(" ")
                    v_POEORB = linesPOEORB[i+2].strip().split(" ")
                    p_SGP4 = linesSGP4[i+1].strip().split(" ")
                    v_SGP4 = linesSGP4[i+2].strip().split(" ")
                    # p_error = [float(x) - float(y) for x, y in zip(p_POEORB, p_SGP4)]
                    # v_error = [float(x) - float(y) for x, y in zip(v_POEORB, v_SGP4)]
                    p_error = [abs(float(x) - float(y)) for x, y in zip(p_POEORB, p_SGP4)]
                    v_error = [abs(float(x) - float(y)) for x, y in zip(v_POEORB, v_SGP4)]
                    dataPError[idx] =[float(x) + float(y) for x, y in zip(dataPError[idx], p_error)]
                    dataVError[idx] =[float(x) + float(y) for x, y in zip(dataVError[idx], v_error)]
    for i in range(len(dataPError)):
        for j in range(len(dataPError[i])):
            dataPError[i][j] = dataPError[i][j]/len(dataSGP4Paths)
    for i in range(len(dataVError)):
        for j in range(len(dataVError[i])):
            dataVError[i][j] = dataVError[i][j]/len(dataSGP4Paths)


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