# HWT906P Python SDK（内置副本）

本目录是 WitMotion HWT906P 官方 Python SDK 的副本，用于让 sWAT 控制软件直接读取
陀螺仪数据，源码未做任何修改。

- 来源：HWT906P 工具工程（本机已验证可正常连接设备、读取角度与四元数）
- 协议：WitStandardProtocol（0x55 帧头，11 字节定长帧）
- 串口：默认 115200 bps
- 依赖：pyserial、numpy

对外接口：

```python
from hwt906p import HWT906P, list_ports, IMUData, gravity_angles, is_static

imu = HWT906P()
imu.open("COM3", 115200)
print(imu.angle_x, imu.angle_y, imu.angle_z)   # Roll, Pitch, Yaw
print(imu.quaternion)                          # (w, x, y, z)
imu.zero_calibrate()                           # 设备侧零位（本功能未使用）
imu.close()
```

说明：本项目里的 theta 计算在 `services/rotation_math.py` 中独立实现（软件侧锁定
初始姿态），不依赖 SDK 内部的 `rotation_angle`，便于单元测试与后续替换硬件。
