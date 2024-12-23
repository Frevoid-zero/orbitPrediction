from sgp4.api import Satrec, WGS84, WGS72
from datetime import datetime, timedelta


def tle_epoch_to_datetime(epoch_year, epoch_day):
    """
    将 TLE 中的 epoch year 和 epoch day 转换为标准 datetime 格式。

    :param epoch_year: TLE 中的年份，例如 24 表示 2024 年
    :param epoch_day: TLE 中的天数，格式为小数，表示该年的第几天
    :return: 对应的 datetime 对象 (UTC)
    """
    # 确定年份归属
    if epoch_year < 57:  # 根据 SGP4 规则，57 之前属于 2000 年之后
        year = 2000 + epoch_year
    else:
        year = 1900 + epoch_year

    # 转换天数为日期
    epoch_date = datetime(year, 1, 1) + timedelta(days=epoch_day - 1)
    return epoch_date


def calculate_tle_epoch_position_velocity(tle_line1, tle_line2):
    """
    根据 TLE 计算其 epoch 时间的卫星位置和速度。

    :param tle_line1: TLE 第一行
    :param tle_line2: TLE 第二行
    :return: (epoch_time, position, velocity)
             - epoch_time: TLE 的时间 (datetime)
             - position: 卫星位置 (km)，格式为 (x, y, z)
             - velocity: 卫星速度 (km/s)，格式为 (vx, vy, vz)
    """
    # 解析 TLE
    satellite = Satrec.twoline2rv(tle_line1, tle_line2, WGS84)

    # 提取 TLE 时间
    epoch_year = satellite.epochyr
    epoch_day = satellite.epochdays
    epoch_time = tle_epoch_to_datetime(epoch_year, epoch_day)

    # 将 epoch 转为儒略日
    jd, fr = satellite.jdsatepoch, satellite.jdsatepochF

    # 计算位置和速度
    e, r, v = satellite.sgp4(jd, fr)
    if e != 0:
        raise ValueError(f"SGP4 Propagation error, code: {e}")

    # 返回结果
    return epoch_time, r, v


# 示例使用
if __name__ == "__main__":
    # 示例 TLE 数据
    tle_line1 = "1 41335U 16011A   24001.48389511  .00000177  00000-0  91083-4 0  9991"
    tle_line2 = "2 41335  98.6302  70.6795 0000859 109.8694 250.2579 14.26740438410058"

    try:
        epoch_time, position, velocity = calculate_tle_epoch_position_velocity(tle_line1, tle_line2)
        print(f"TLE 时间 (UTC): {epoch_time}")
        print(f"位置 (km): X={position[0]}, Y={position[1]}, Z={position[2]}")
        print(f"速度 (km/s): VX={velocity[0]}, VY={velocity[1]}, VZ={velocity[2]}")
    except ValueError as e:
        print(f"计算失败: {e}")
