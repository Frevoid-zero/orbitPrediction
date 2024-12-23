import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.data import DataProvidersManager, DirectoryCrawler
from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
from org.orekit.time import AbsoluteDate, DateTimeComponents, TimeScalesFactory, Month
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions

# 初始化 Orekit 环境
orekit.initVM()
setup_orekit_curdir("../../orekit-data-master")  # 替换为你的 orekit 数据路径

# 定义 TLE 数据
line1 = "1 41335U 16011A   23365.92285751  .00000165  00000-0  86205-4 0  9990"
line2 = "2 41335  98.6302  70.1263 0000848 109.5187 250.6085 14.26740110409977"

if __name__ == "__main__":
    # 从 TLE 数据创建 TLE 对象
    tle = TLE(line1, line2)

    # 创建 TLEPropagator
    propagator = TLEPropagator.selectExtrapolator(tle)

    # 设置目标时间（2024年1月1日0时0分2秒）
    utc = TimeScalesFactory.getUTC()
    # 创建 DateTimeComponents 对象，表示2024年1月1日0时0分2秒的时间
    target_date = DateTimeComponents(2024, Month.JANUARY, 1, 0, 0, 2.0)
    # 创建 AbsoluteDate 对象
    target_time = AbsoluteDate(target_date, utc)

    # 传播到目标时间并获取轨道状态
    orbit_at_target_time = propagator.propagate(target_time)

    # 获取 ICRF 坐标系中的 PV 坐标
    pv_coordinates_icrf = orbit_at_target_time.getPVCoordinates(FramesFactory.getICRF())

    # 获取 ITRS 坐标系
    itrs = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # 从 ICRF 到 ITRS 的转换
    transform_to_itrf = FramesFactory.getICRF().getTransformTo(itrs, target_time)

    # 转换 PV 坐标到 ITRS 坐标系
    pv_coordinates_itrf = transform_to_itrf.transformPVCoordinates(pv_coordinates_icrf)

    # 获取位置、速度和加速度
    position_itrf = pv_coordinates_itrf.getPosition()
    velocity_itrf = pv_coordinates_itrf.getVelocity()
    acceleration_itrf = pv_coordinates_itrf.getAcceleration()

    # 输出位置、速度和加速度
    print(f"Position in ITRS: {position_itrf}")
    print(f"Velocity in ITRS: {velocity_itrf}")
    print(f"Acceleration in ITRS: {acceleration_itrf}")