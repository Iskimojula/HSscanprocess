# -*- coding: utf-8 -*-
"""四元数姿态运算（纯标准库实现，便于单元测试与复用）。

数学约定
    * 四元数一律写作 [w, x, y, z]，表示"机体坐标系 -> 世界坐标系"的旋转，
      与 HWT906P 输出一致；
    * 欧拉角顺序为 ZYX（先 yaw 绕 Z，再 pitch 绕 Y，最后 roll 绕 X），
      对应传感器 (angle_z=Yaw, angle_y=Pitch, angle_x=Roll)；
    * 相对转动四元数 q_delta = q_cur ⊗ q_ref⁻¹；
    * 转动角 theta = 2·atan2(‖v‖, |w|) ∈ [0, 180]，转轴 L = v/‖v‖。

为什么这样算
    1. 直接用欧拉角相减在 ±180° 处会跳变（179° - (-179°) = 358° 的错误结果），
       四元数差分天然连续；
    2. 用 atan2(‖v‖, |w|) 而不是 acos(w)，在 theta ≈ 0 时数值更稳定；
    3. 取 |w| 消除四元数双重覆盖（q 与 -q 表示同一姿态）。
"""

import math
from typing import NamedTuple, Sequence, Tuple

EPS = 1e-9


class RotationResult(NamedTuple):
    """一次相对转动的结果。"""

    theta_deg: float                                # 转动角 theta，单位：度，范围 [0, 180]
    axis: Tuple[float, float, float]                # 转轴 L 单位向量；无转动时为 (0, 0, 0)
    quat_delta: Tuple[float, float, float, float]   # 归一化后的相对四元数 [w, x, y, z]


def normalize(q: Sequence[float]) -> list:
    """把四元数归一化；零四元数返回单位四元数。"""
    n = math.sqrt(sum(float(x) * float(x) for x in q))
    if n < EPS:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(x) / n for x in q]


def multiply(q1: Sequence[float], q2: Sequence[float]) -> list:
    """四元数乘法 q1 ⊗ q2，输入顺序为 [w, x, y, z]。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def conjugate(q: Sequence[float]) -> list:
    """单位四元数的共轭（即其逆）：[w, -x, -y, -z]。"""
    return [q[0], -q[1], -q[2], -q[3]]


def axis_angle_to_quaternion(axis: Sequence[float], angle_deg: float) -> list:
    """由转轴与转角构造四元数（测试与自检用）。"""
    n = math.sqrt(sum(float(a) * float(a) for a in axis))
    if n < EPS:
        return [1.0, 0.0, 0.0, 0.0]
    ax, ay, az = (float(a) / n for a in axis)
    half = math.radians(angle_deg) * 0.5
    s = math.sin(half)
    return [math.cos(half), ax * s, ay * s, az * s]


def euler_to_quaternion(yaw_deg: float, pitch_deg: float, roll_deg: float) -> list:
    """欧拉角 -> 四元数（ZYX 顺序，先 yaw 后 pitch 再 roll）。"""
    cy, sy = math.cos(math.radians(yaw_deg) * 0.5), math.sin(math.radians(yaw_deg) * 0.5)
    cp, sp = math.cos(math.radians(pitch_deg) * 0.5), math.sin(math.radians(pitch_deg) * 0.5)
    cr, sr = math.cos(math.radians(roll_deg) * 0.5), math.sin(math.radians(roll_deg) * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def quaternion_to_euler(q: Sequence[float]) -> Tuple[float, float, float]:
    """四元数 -> 欧拉角，返回 (yaw, pitch, roll)，单位：度。"""
    w, x, y, z = normalize(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def relative_rotation(q_ref: Sequence[float], q_cur: Sequence[float]) -> RotationResult:
    """计算 q_ref -> q_cur 的相对转动角 theta 与转轴 L。

    参数
        q_ref: 初始姿态四元数（按"记录"按钮时锁定的姿态）
        q_cur: 当前姿态四元数
    返回
        RotationResult(theta_deg, axis, quat_delta)
    """
    qd = normalize(multiply(normalize(q_cur), conjugate(normalize(q_ref))))
    w, x, y, z = qd
    vnorm = math.sqrt(x * x + y * y + z * z)
    theta = math.degrees(2.0 * math.atan2(vnorm, abs(w)))
    if vnorm < EPS:
        return RotationResult(0.0, (0.0, 0.0, 0.0), (w, x, y, z))
    return RotationResult(theta, (x / vnorm, y / vnorm, z / vnorm), (w, x, y, z))
