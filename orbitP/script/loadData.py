import sys

from torch.utils.data import Dataset
from datetime import datetime
import numpy as np
import os
from tqdm import tqdm
import torch

beginDate = datetime.strptime("2024-01-01","%Y-%m-%d")
endDate = datetime.strptime("2024-02-01","%Y-%m-%d")
dataPOEORBDir = "../../dataset/dataPOEORB/"
dataSGP4Dir = "../../dataset/dataSGP4/"
dataOrekitDir = "../../dataset/dataOrekit/"

def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(os.path.normpath(file_path))
    return file_paths


def get_orbitData_SGP4(dataSGP4Dir,dataPOEORBDir):
    orbitData = []
    filesSGP4Path = get_file_paths(dataSGP4Dir)
    t_bar = tqdm(filesSGP4Path, total=len(filesSGP4Path))
    for fileSGP4Path in t_bar:
        fileName = os.path.basename(fileSGP4Path).replace(".txt", "")
        fileDate = datetime.strptime(fileName, "%Y-%m-%d")
        t_bar.set_description(fileName)
        filePOEORBPath = dataPOEORBDir + os.path.basename(fileSGP4Path)

        with open(fileSGP4Path, "r", encoding="utf-8") as fileSGP4:
            with open(filePOEORBPath, "r", encoding="utf-8") as filePOEORB:
                linesSGP4 = fileSGP4.readlines()
                linesPOEORB = filePOEORB.readlines()
                for i in range(0, len(linesPOEORB), 3):
                    time_utc = datetime.strptime(linesPOEORB[i].strip(), "UTC=%Y-%m-%dT%H:%M:%S.%f")
                    p_POEORB = linesPOEORB[i + 1].strip().split(" ")
                    v_POEORB = linesPOEORB[i + 2].strip().split(" ")
                    p_SGP4 = linesSGP4[i + 1].strip().split(" ")
                    v_SGP4 = linesSGP4[i + 2].strip().split(" ")
                    p_error = [float(x) - float(y) for x, y in zip(p_POEORB, p_SGP4)]
                    v_error = [float(x) - float(y) for x, y in zip(v_POEORB, v_SGP4)]
                    hour = time_utc.hour
                    minute = time_utc.minute
                    second = time_utc.second
                    day = time_utc.day
                    month = time_utc.month
                    seconds = hour*3600 + minute*60 + second
                    # orbitData.append([p_error, v_error, p_SGP4, v_SGP4, [month]*3, [day]*3, [seconds]*3, p_POEORB, v_POEORB])
                    orbitData.append([p_error, [float(x) for x in v_SGP4], [month]*3, [day]*3,[seconds]*3, [float(x) for x in p_SGP4], [float(x) for x in p_POEORB]])

    orbitData = np.array(orbitData)
    return orbitData

def get_orbitData_Orekit(dataOrekitDir,dataPOEORBDir):
    orbitData = []
    filesOrekitPath = get_file_paths(dataOrekitDir)
    t_bar = tqdm(filesOrekitPath, total=len(filesOrekitPath))
    for fileOrekitPath in t_bar:
        fileName = os.path.basename(fileOrekitPath).replace(".txt", "")
        fileDate = datetime.strptime(fileName, "%Y-%m-%d")
        t_bar.set_description(fileName)
        filePOEORBPath = dataPOEORBDir + os.path.basename(fileOrekitPath)

        with open(fileOrekitPath, "r", encoding="utf-8") as fileOrekit:
            with open(filePOEORBPath, "r", encoding="utf-8") as filePOEORB:
                linesOrekit = fileOrekit.readlines()
                linesPOEORB = filePOEORB.readlines()
                j = 0
                for i in range(0, len(linesPOEORB), 3):
                    time_utc = datetime.strptime(linesPOEORB[i].strip(), "UTC=%Y-%m-%dT%H:%M:%S.%f")
                    p_POEORB = linesPOEORB[i + 1].strip().split(" ")
                    v_POEORB = linesPOEORB[i + 2].strip().split(" ")
                    p_Orekit = linesOrekit[j + 1].strip().split(" ")
                    v_Orekit = linesOrekit[j + 2].strip().split(" ")
                    a_Orekit = linesOrekit[j + 3].strip().split(" ")
                    p_error = [float(x) - float(y) for x, y in zip(p_POEORB, p_Orekit)]
                    v_error = [float(x) - float(y) for x, y in zip(v_POEORB, v_Orekit)]
                    hour = time_utc.hour
                    minute = time_utc.minute
                    second = time_utc.second
                    day = time_utc.day
                    month = time_utc.month
                    seconds = hour*3600 + minute*60 + second
                    # orbitData.append([p_error, v_error, p_SGP4, v_SGP4, [month]*3, [day]*3, [seconds]*3, p_POEORB, v_POEORB])
                    orbitData.append([p_error, [float(x) for x in v_Orekit], [float(x) for x in a_Orekit],[month]*3, [day]*3,[seconds]*3, [float(x) for x in p_Orekit], [float(x) for x in p_POEORB]])
                    j+=4

    orbitData = np.array(orbitData)
    return orbitData


if __name__ == "__main__":
    orbitData_SGP4 = get_orbitData_SGP4(dataSGP4Dir,dataPOEORBDir)
    orbitData_Orekit = get_orbitData_Orekit(dataOrekitDir,dataPOEORBDir)
    print(orbitData_SGP4.shape)
    print(orbitData_Orekit.shape)