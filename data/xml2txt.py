import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm import tqdm
import os

def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(os.path.normpath(file_path))
    return file_paths

def cmpLastDate(utc):
    global lastDate
    if lastDate == "":
        lastDate = utc
        return False
    time_format = "UTC=%Y-%m-%dT%H:%M:%S.%f"

    t1 = datetime.strptime(lastDate, time_format)
    t2 = datetime.strptime(utc, time_format)
    if t1 >= t2:
        return True
    lastDate = utc
    return False

dataEOFDir = "../dataset/dataEOF/Finish"
dataPOEORBDir = "../dataset/dataPOEORB"

dateSet = set()
lastDate = ""

if __name__ == "__main__":
    filePaths= get_file_paths(dataEOFDir)
    if os.path.exists('../dataset/dataPOEORB/'):
        shutil.rmtree('../dataset/dataPOEORB/')
    if not os.path.exists(dataPOEORBDir):
        os.makedirs(dataPOEORBDir)

    t_bar = tqdm(filePaths,total=len(filePaths))
    for filePath in t_bar:
        fileNameO = filePath.split('_')[-3][1:9]
        fileNameN = fileNameO[:4]+"-"+fileNameO[4:6]+"-"+fileNameO[6:]
        t_bar.set_description(f"{fileNameN}")

        tree = ET.parse(filePath)
        root = tree.getroot()
        dateDict = {}
        # 解析 XML 中的每个 OSV
        for osv in root.findall('.//OSV'):
            utc_time = osv.find('UTC').text
            # 提取日期部分 (格式: 2023-12-31)
            date_str = utc_time.split('=')[1].split('T')[0]
            # 提取各个数据项
            utc = osv.find('UTC').text
            x = osv.find('X').text
            y = osv.find('Y').text
            z = osv.find('Z').text
            vx = osv.find('VX').text
            vy = osv.find('VY').text
            vz = osv.find('VZ').text

            if(cmpLastDate(utc)):
                continue
            #将数据按日期分组
            if date_str not in dateDict:
                dateDict[date_str] = []

            dateDict[date_str].append({
                'UTC': utc,
                'X': x,
                'Y': y,
                'Z': z,
                'VX': vx,
                'VY': vy,
                'VZ': vz
            })
        time_format = "UTC=%Y-%m-%dT%H:%M:%S.%f"
        # 写入每一天的数据到对应的 txt 文件
        for date, data in dateDict.items():
            writePath = f"../dataset/dataPOEORB/{date}.txt"
            if date in dateSet:
                with open(writePath, 'a') as file:
                    for msg in data:
                        time_utc = datetime.strptime(msg['UTC'], time_format)
                        if time_utc.second!=2:
                            continue
                        file.write(f"{msg['UTC']}\n")
                        file.write(f"{msg['X']} {msg['Y']} {msg['Z']}\n")
                        file.write(f"{msg['VX']} {msg['VY']} {msg['VZ']}\n")
            else:
                dateSet.add(date)
                with open(writePath, 'w') as file:
                    for msg in data:
                        time_utc = datetime.strptime(msg['UTC'], time_format)
                        if time_utc.second!=2:
                            continue
                        file.write(f"{msg['UTC']}\n")
                        file.write(f"{msg['X']} {msg['Y']} {msg['Z']}\n")
                        file.write(f"{msg['VX']} {msg['VY']} {msg['VZ']}\n")