import shutil

import numpy as np
from astropy.coordinates import ICRS, EarthLocation, CartesianDifferential, TEME
from datetime import datetime, timedelta
from sgp4.api import Satrec, WGS84
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from astropy import units as u
import os
from tqdm import tqdm

# TLE数据
tle1 = ""
tle2 = ""
beginDate = datetime.strptime("2024-01-01","%Y-%m-%d")
endDate = datetime.strptime("2024-01-08","%Y-%m-%d")
dataLastTLEPath = "../dataset/dataLastTLE.txt"
dataPOEORBDir = "../dataset/dataPOEORB"
dataSGP4Dir = "../dataset/dataSGP4/"

dataDict = {}

def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(os.path.normpath(file_path))
    return file_paths

# 将TLE的时间部分转换为标准的datetime对象
def epoch2datetime(year, day):
    """
    将 TLE 中的 epoch year 和 epoch day 转换为标准 datetime 格式。
    :param year: TLE 中的年份，例如 24 表示 2024 年
    :param day: TLE 中的天数，格式为小数，表示该年的第几天
    :return: 对应的 datetime 对象 (UTC)
    """
    if year < 57:
        year = 2000 + year
    else:
        year = 1900 + year
    epoch_date = datetime(year, 1, 1) + timedelta(days=day - 1)
    return epoch_date

# 计算卫星在 TLE 时刻的位置和速度
def getTLEMsg(line1, line2):
    satellite = Satrec.twoline2rv(line1, line2, WGS84)
    # 提取TLE的信息
    year = satellite.epochyr
    day = satellite.epochdays
    time = epoch2datetime(year, day)

    # 将历元转为儒略日
    jd, fr = satellite.jdsatepoch, satellite.jdsatepochF

    # 使用SGP4模型计算位置和速度
    e, p, v = satellite.sgp4(jd, fr)
    if e != 0:
        raise ValueError(f"SGP4 Propagation error, code: {e}")

    return time, p, v

def getPredictMsg(line1, line2, time_utc):
    # Step 1: 初始化 SGP4 卫星对象
    satellite = Satrec.twoline2rv(line1, line2,WGS84)

    # Step 2: 时间转换到 Julian Date
    time = Time(time_utc, scale="utc")
    jd1, jd2 = time.jd1, time.jd2

    # Step 3: 使用 SGP4 获取 TEME 坐标
    e, p_teme, v_teme = satellite.sgp4(jd1, jd2)
    if e != 0:
        raise ValueError(f"SGP4 Propagation error, code: {e}")

    # Step 4: 将位置和速度表示为 CartesianRepresentation 和 CartesianDifferential
    r_teme_cartesian = CartesianRepresentation(p_teme * u.km)
    v_teme_cartesian = CartesianDifferential(v_teme * u.km / u.s)

    # Step 5: 创建 TEME 坐标
    teme = TEME(r_teme_cartesian.with_differentials(v_teme_cartesian), obstime=time)

    # Step 6: 从 TEME 转换到 ITRS
    itrs = teme.transform_to(ITRS(obstime=time))

    # Step 7: 提取 ITRS 中的位置和速度
    p_itrs = itrs.cartesian.xyz.to(u.m).value  # 转为米
    v_itrs = itrs.velocity.d_xyz.to(u.m / u.s).value  # 转为米/秒

    return p_itrs, v_itrs

# 使用示例
if __name__ == "__main__":
    if os.path.exists(dataSGP4Dir):
        shutil.rmtree(dataSGP4Dir)
    if not os.path.exists(dataSGP4Dir):
        os.mkdir(dataSGP4Dir)
    with open(dataLastTLEPath, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            TLE = lines[i:i + 2]
            tle1 = TLE[0].strip()
            tle2 = TLE[1].strip()
            time_TLE, p_TLE, v_TLE = getTLEMsg(tle1, tle2)
            dataDict[time_TLE.date().strftime("%Y-%m-%d")]=[tle1,tle2]

    filesPath = get_file_paths(dataPOEORBDir)
    for filePath in filesPath:

        fileName = os.path.basename(filePath).replace(".txt","")
        fileDate = datetime.strptime(fileName,"%Y-%m-%d")
        if fileDate < beginDate or fileDate > endDate:
            continue
        writePath = dataSGP4Dir + os.path.basename(filePath)
        with open(writePath, "w", encoding="utf-8") as writeFile:
            with open(filePath,"r",encoding="utf-8") as file:
                lines = file.readlines()

                t_bar = tqdm(range(0,len(lines),3),desc=fileName,total=len(range(0,len(lines),3)))
                for i in t_bar:
                    time_utc = lines[i].strip().replace("UTC=","")
                    time_utc_pre = datetime.strptime(time_utc,"%Y-%m-%dT%H:%M:%S.%f") - timedelta(days=1)#使用前一天最晚的TLE预测
                    nowTLE = dataDict[time_utc_pre.date().strftime("%Y-%m-%d")]
                    p_itrs, v_itrs = getPredictMsg(nowTLE[0], nowTLE[1], time_utc)
                    writeFile.write("UTC="+time_utc+"\n")
                    writeFile.write(f"{p_itrs[0]} {p_itrs[1]} {p_itrs[2]}"+"\n")
                    writeFile.write(f"{v_itrs[0]} {v_itrs[1]} {v_itrs[2]}"+"\n")
                print(f"dataSGP4 数据已保存到 {writePath}")
