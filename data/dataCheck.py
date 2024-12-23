import numpy as np
from astropy.coordinates import ICRS, EarthLocation, CartesianDifferential, TEME
from datetime import datetime, timedelta
from sgp4.api import Satrec, WGS84
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from astropy import units as u

# TLE数据
tle1 = "1 41335U 16011A   24001.48389511  .00000177  00000-0  91083-4 0  9991"
tle2 = "2 41335  98.6302  70.6795 0000859 109.8694 250.2579 14.26740438410058"

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
    p_ecef = itrs.cartesian.xyz.to(u.m).value  # 转为米
    v_ecef = itrs.velocity.d_xyz.to(u.m / u.s).value  # 转为米/秒

    return p_ecef, v_ecef

# 使用示例
if __name__ == "__main__":

    # 获取 TLE epoch 的位置和速度
    time_TLE, p_TLE, v_TLE = getTLEMsg(tle1, tle2)
    print(f"TLE Epoch 时间 (UTC): {time_TLE}")
    print(f"TLE Epoch 位置 (m): X={p_TLE[0]*1000}, Y={p_TLE[1]*1000}, Z={p_TLE[2]*1000}")
    print(f"TLE Epoch 速度 (m/s): VX={v_TLE[0]*1000}, VY={v_TLE[1]*1000}, VZ={v_TLE[2]*1000}")

    # 输入指定时间 (UTC)
    # time_utc = "2024-01-01T11:36:52"
    time_utc = "2024-01-01T14:29:12"

    # 获取指定时间的ICRF和ECEF坐标
    p_ecef, v_ecef = getPredictMsg(tle1, tle2, time_utc)
    print(f"预测的时间 (UTC): {time_utc.replace('T',' ')}")
    print(f"预测的位置 (m): X={p_ecef[0]}, Y={p_ecef[1]}, Z={p_ecef[2]}")
    print(f"预测的速度 (m/s): VX={v_ecef[0]}, VY={v_ecef[1]}, VZ={v_ecef[2]}")
