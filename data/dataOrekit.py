import shutil
import sys

import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.data import DataProvidersManager, DirectoryCrawler
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.time import AbsoluteDate, DateTimeComponents, TimeScalesFactory, Month
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions
from datetime import datetime, timedelta
from sgp4.api import Satrec, WGS84
import os

from tqdm import tqdm


# 初始化 Orekit 环境
orekit.initVM()
setup_orekit_curdir("../orekit-data-master")  # 替换为你的 orekit 数据路径

# TLE数据
tle1 = ""
tle2 = ""
beginDate = datetime.strptime("2024-01-10","%Y-%m-%d")
endDate = datetime.strptime("2024-01-20","%Y-%m-%d")
dataLastTLEPath = "../dataset/dataLastTLE.txt"
dataPOEORBDir = "../dataset/dataPOEORB"
dataOrekitDir = "../dataset/dataOrekit/"

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

def getMonth(month):
    # 将月份数字转换为Month枚举
    months = {
        1: Month.JANUARY,
        2: Month.FEBRUARY,
        3: Month.MARCH,
        4: Month.APRIL,
        5: Month.MAY,
        6: Month.JUNE,
        7: Month.JULY,
        8: Month.AUGUST,
        9: Month.SEPTEMBER,
        10: Month.OCTOBER,
        11: Month.NOVEMBER,
        12: Month.DECEMBER
    }
    return months[month]

def getPredictMsg(line1, line2, time_utc):
    tle = TLE(line1, line2)
    utc_time = datetime.strptime(time_utc, "%Y-%m-%dT%H:%M:%S.%f")
    # 创建 TLEPropagator
    propagator = TLEPropagator.selectExtrapolator(tle)

    # 设置目标时间（2024年1月1日0时0分2秒）
    utc = TimeScalesFactory.getUTC()
    # 创建 DateTimeComponents 对象，表示2024年1月1日0时0分2秒的时间
    # target_date = DateTimeComponents(2024, Month.JANUARY, 1, 0, 0, 2.0)
    # 创建 DateTimeComponents 对象，使用 Month 枚举
    month_enum = getMonth(utc_time.month)
    target_date = DateTimeComponents(utc_time.year, month_enum, utc_time.day, utc_time.hour, utc_time.minute,
                                     utc_time.second + utc_time.microsecond / 1e6)

    target_time = AbsoluteDate(target_date, utc)

    # 传播到目标时间并获取轨道状态
    orbit_at_target_time = propagator.propagate(target_time)

    # 获取ITRF坐标系
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)  # 使用IERSConventions枚举值

    # 获取ITRF坐标系中的PV坐标
    pv_coordinates_itrf = orbit_at_target_time.getPVCoordinates(itrf)

    # 获取位置、速度和加速度
    p_itrf = pv_coordinates_itrf.getPosition()
    v_itrf = pv_coordinates_itrf.getVelocity()
    a_itrf = pv_coordinates_itrf.getAcceleration()

    return p_itrf, v_itrf, a_itrf


if __name__ == "__main__":
    if os.path.exists(dataOrekitDir):
        shutil.rmtree(dataOrekitDir)
    if not os.path.exists(dataOrekitDir):
        os.mkdir(dataOrekitDir)
    with open(dataLastTLEPath, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            TLE_arr = lines[i:i + 2]
            tle1 = TLE_arr[0].strip()
            tle2 = TLE_arr[1].strip()
            time_TLE, p_TLE, v_TLE = getTLEMsg(tle1, tle2)
            dataDict[time_TLE.date().strftime("%Y-%m-%d")]=[tle1,tle2]

    filesPath = get_file_paths(dataPOEORBDir)
    for filePath in filesPath:

        fileName = os.path.basename(filePath).replace(".txt", "")
        fileDate = datetime.strptime(fileName, "%Y-%m-%d")
        if fileDate < beginDate or fileDate > endDate:
            continue
        writePath = dataOrekitDir + os.path.basename(filePath)
        with open(writePath, "w", encoding="utf-8") as writeFile:
            with open(filePath, "r", encoding="utf-8") as file:
                lines = file.readlines()
                t_bar = tqdm(range(0, len(lines), 3), desc=fileName, total=len(range(0, len(lines), 3)))
                for i in t_bar:
                    time_utc = lines[i].strip().replace("UTC=", "")
                    dateTime = datetime.strptime(time_utc, "%Y-%m-%dT%H:%M:%S.%f")
                    time_utc_pre = datetime.strptime(time_utc, "%Y-%m-%dT%H:%M:%S.%f") - timedelta(
                        days=1)  # 使用前一天最晚的TLE预测
                    nowTLE = dataDict[time_utc_pre.date().strftime("%Y-%m-%d")]
                    p_itrf,v_itrf,a_itrf = getPredictMsg(nowTLE[0], nowTLE[1], time_utc)
                    writeFile.write("UTC=" + time_utc + "\n")
                    writeFile.write(f"{p_itrf.getX()} {p_itrf.getY()} {p_itrf.getZ()}" + "\n")
                    writeFile.write(f"{v_itrf.getX()} {v_itrf.getY()} {v_itrf.getZ()}" + "\n")
                    writeFile.write(f"{a_itrf.getX()} {a_itrf.getY()} {a_itrf.getZ()}" + "\n")
                print(f"dataOrekit 数据已保存到 {writePath}")