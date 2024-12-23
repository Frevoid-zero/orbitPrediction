import numpy as np
from astropy.coordinates import ICRS, EarthLocation, CartesianDifferential, TEME
from datetime import datetime, timedelta
from sgp4.api import Satrec, WGS84
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from astropy import units as u


dataTLEPath = "../dataset/dataTLE.txt"
dataLastTLEPath = "../dataset/dataLastTLE.txt"
lastTime = None
lastTLE = []
lastTLEs = []
# 将TLE的历元转换为标准的datetime对象
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

# 使用示例
if __name__ == "__main__":

    with open(dataTLEPath, 'r',encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0,len(lines),2):
            TLE = lines[i:i+2]
            tle1 =TLE[0].strip()
            tle2 =TLE[1].strip()
            time_TLE, p_TLE, v_TLE = getTLEMsg(tle1, tle2)
            if lastTime == None:
                lastTime = time_TLE
                lastTLE = [tle1,tle2]
            elif time_TLE.date()==lastTime.date():
                lastTime = time_TLE
                lastTLE = [tle1, tle2]
            else:
                lastTLEs.extend(lastTLE)
                lastTime = time_TLE
                lastTLE = [tle1, tle2]

        lastTLEs.extend(lastTLE)
        tle_data = "\n".join(lastTLEs)
        with open(dataLastTLEPath, "w", encoding="utf-8") as file:
            file.write(tle_data)
            print(f"LastTLE 数据已保存到 {dataLastTLEPath}")
        # for i in range(0,len(lastTLEs),2):
        #     TLE = lastTLEs[i:i + 2]
        #     tle1 = TLE[0].strip()
        #     tle2 = TLE[1].strip()
        #     time_TLE, p_TLE, v_TLE = getTLEMsg(tle1, tle2)
        #     # 获取 TLE epoch 的位置和速度
        #     print(f"TLE Epoch 时间 (UTC): {time_TLE}")
        #     print(f"TLE Epoch 位置 (m): X={p_TLE[0]*1000}, Y={p_TLE[1]*1000}, Z={p_TLE[2]*1000}")
        #     print(f"TLE Epoch 速度 (m/s): VX={v_TLE[0]*1000}, VY={v_TLE[1]*1000}, VZ={v_TLE[2]*1000}")
        #     print()
