# -*- coding: utf-8 -*-
"""
HWT906P WitStandardProtocol parser.
Based on the official WitMotion WitStandardProtocol (JY901 compatible).
"""

import struct
import math
import time
from threading import Thread, Lock
from collections import namedtuple

import serial
import numpy as np

# ── Data packet types ──────────────────────────────────────────────
PKT_CHIP_TIME = 0x50
PKT_ACCEL = 0x51
PKT_GYRO = 0x52
PKT_ANGLE = 0x53
PKT_MAG = 0x54
PKT_PORT = 0x55      # barometric pressure / height (HWT906P specific)
PKT_LONLAT = 0x57
PKT_GPS = 0x58
PKT_QUATERNION = 0x59
PKT_FIND = 0x5F      # register read response

PACK_SIZE = 11
HEADER = 0x55

# ── Calibration constants ──────────────────────────────────────────
GYRO_RANGE = 2000.0     # deg/s
ACC_RANGE = 16.0         # g
ANGLE_RANGE = 180.0      # degrees

# ── Register addresses ─────────────────────────────────────────────
REG_SAVE = 0x00
REG_CALSW = 0x01
REG_RSW = 0x02      # data output content
REG_RRATE = 0x03    # return rate
REG_BAUD = 0x04
REG_AXIS_DIR = 0x23
REG_ALGORITHM = 0x24
REG_UNLOCK = 0x69

# ── Data container ─────────────────────────────────────────────────
IMUData = namedtuple("IMUData", [
    "timestamp",
    "accel_x", "accel_y", "accel_z",       # g
    "gyro_x", "gyro_y", "gyro_z",           # deg/s
    "angle_x", "angle_y", "angle_z",        # degrees
    "mag_x", "mag_y", "mag_z",              # magnetic field raw
    "quat_w", "quat_x", "quat_y", "quat_z", # quaternion [w,x,y,z]
    "temperature",                          # degrees C (from LonLat packet)
    "pressure", "height",                   # barometric
    "lon", "lat", "yaw_gps", "speed",       # GPS
    "chip_time",                            # "YYYY-MM-DD HH:MM:SS.mmm"
])


class WitProtocolParser:
    """Parses WitStandardProtocol byte streams into structured IMU data."""

    def __init__(self):
        self._buffer = bytearray()
        self._lock = Lock()

        # Latest parsed values
        self._accel = [0.0, 0.0, 0.0]
        self._gyro = [0.0, 0.0, 0.0]
        self._angle = [0.0, 0.0, 0.0]
        self._mag = [0.0, 0.0, 0.0]
        self._quat = [1.0, 0.0, 0.0, 0.0]
        self._chip_time = ""
        self._temperature = 0.0
        self._pressure = 0.0
        self._height = 0.0
        self._lon = 0.0
        self._lat = 0.0
        self._yaw_gps = 0.0
        self._speed = 0.0

        # Register read-back storage
        self._find_values = []

        # Data callback
        self._callbacks = []

    def register_callback(self, callback):
        """Register a callback fn(data: IMUData) invoked on each complete data set."""
        self._callbacks.append(callback)

    def feed(self, raw_bytes: bytes):
        """Feed raw serial bytes into the parser."""
        with self._lock:
            self._buffer.extend(raw_bytes)
            self._parse_buffer()

    def _parse_buffer(self):
        """Scan buffer for complete 11-byte packets."""
        while len(self._buffer) >= PACK_SIZE:
            if self._buffer[0] != HEADER:
                self._buffer.pop(0)
                continue

            pkt = self._buffer[:PACK_SIZE]
            # Validate checksum
            if (sum(pkt[:10]) & 0xFF) != pkt[10]:
                self._buffer.pop(0)
                continue

            pkt_type = pkt[1]
            self._dispatch(pkt_type, pkt[2:10])
            self._buffer = self._buffer[PACK_SIZE:]

            # Fire callback when a complete angle packet arrives (common sync point)
            if pkt_type == PKT_ANGLE:
                self._fire_callback()

    def _dispatch(self, pkt_type: int, data: bytes):
        """Route parsed packet to the correct handler."""
        if pkt_type == PKT_CHIP_TIME:
            self._parse_chip_time(data)
        elif pkt_type == PKT_ACCEL:
            self._parse_accel(data)
        elif pkt_type == PKT_GYRO:
            self._parse_gyro(data)
        elif pkt_type == PKT_ANGLE:
            self._parse_angle(data)
        elif pkt_type == PKT_MAG:
            self._parse_mag(data)
        elif pkt_type == PKT_PORT:  # 0x56
            self._parse_pressure(data)
        elif pkt_type == PKT_LONLAT:
            self._parse_lonlat(data)
        elif pkt_type == PKT_GPS:
            self._parse_gps(data)
        elif pkt_type == PKT_QUATERNION:
            self._parse_quaternion(data)
        elif pkt_type == PKT_FIND:
            self._parse_find(data)

    # ── Packet parsers ─────────────────────────────────────────────

    @staticmethod
    def _parse_s16(lo: int, hi: int) -> int:
        val = (hi << 8) | lo
        return val - 65536 if val >= 32768 else val

    @staticmethod
    def _parse_u16(lo: int, hi: int) -> int:
        return (hi << 8) | lo

    def _parse_accel(self, data: bytes):
        self._accel[0] = self._parse_s16(data[0], data[1]) / 32768.0 * ACC_RANGE
        self._accel[1] = self._parse_s16(data[2], data[3]) / 32768.0 * ACC_RANGE
        self._accel[2] = self._parse_s16(data[4], data[5]) / 32768.0 * ACC_RANGE

    def _parse_gyro(self, data: bytes):
        self._gyro[0] = self._parse_s16(data[0], data[1]) / 32768.0 * GYRO_RANGE
        self._gyro[1] = self._parse_s16(data[2], data[3]) / 32768.0 * GYRO_RANGE
        self._gyro[2] = self._parse_s16(data[4], data[5]) / 32768.0 * GYRO_RANGE

    def _parse_angle(self, data: bytes):
        self._angle[0] = self._parse_s16(data[0], data[1]) / 32768.0 * ANGLE_RANGE
        self._angle[1] = self._parse_s16(data[2], data[3]) / 32768.0 * ANGLE_RANGE
        self._angle[2] = self._parse_s16(data[4], data[5]) / 32768.0 * ANGLE_RANGE

    def _parse_mag(self, data: bytes):
        self._mag[0] = self._parse_s16(data[0], data[1])
        self._mag[1] = self._parse_s16(data[2], data[3])
        self._mag[2] = self._parse_s16(data[4], data[5])

    def _parse_chip_time(self, data: bytes):
        y = data[0] + 2000
        m = data[1]
        d = data[2]
        hh = data[3]
        mm = data[4]
        ss = data[5]
        ms = self._parse_u16(data[6], data[7])
        self._chip_time = f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

    def _parse_quaternion(self, data: bytes):
        q0 = self._parse_s16(data[0], data[1]) / 32768.0
        q1 = self._parse_s16(data[2], data[3]) / 32768.0
        q2 = self._parse_s16(data[4], data[5]) / 32768.0
        q3 = self._parse_s16(data[6], data[7]) / 32768.0
        # Store as [w, x, y, z] convention
        self._quat = [q0, q1, q2, q3]

    def _parse_lonlat(self, data: bytes):
        lon_deg = self._parse_s16(data[0], data[1])
        lon_min = self._parse_u16(data[2], data[3]) / 100.0
        lat_deg = self._parse_s16(data[4], data[5])
        lat_min = self._parse_u16(data[6], data[7]) / 100.0
        self._lon = lon_deg + lon_min / 60.0
        self._lat = lat_deg + lat_min / 60.0

    def _parse_gps(self, data: bytes):
        height = self._parse_s16(data[0], data[1]) / 10.0
        yaw_deg = self._parse_s16(data[2], data[3]) / 100.0
        speed = self._parse_u16(data[4], data[5])  # knots * 100?
        self._height = height
        self._yaw_gps = yaw_deg
        self._speed = speed
        self._temperature = self._parse_s16(data[6], data[7]) / 100.0

    def _parse_pressure(self, data: bytes):
        p = self._parse_u16(data[0], data[1]) + (self._parse_u16(data[2], data[3]) << 16)
        h = self._parse_u16(data[4], data[5]) + (self._parse_u16(data[6], data[7]) << 16)
        self._pressure = p / 100.0
        self._height = h / 100.0

    def _parse_find(self, data: bytes):
        self._find_values = [
            self._parse_u16(data[0], data[1]),
            self._parse_u16(data[2], data[3]),
            self._parse_u16(data[4], data[5]),
            self._parse_u16(data[6], data[7]),
        ]

    def _fire_callback(self):
        d = IMUData(
            timestamp=time.time(),
            accel_x=self._accel[0], accel_y=self._accel[1], accel_z=self._accel[2],
            gyro_x=self._gyro[0], gyro_y=self._gyro[1], gyro_z=self._gyro[2],
            angle_x=self._angle[0], angle_y=self._angle[1], angle_z=self._angle[2],
            mag_x=self._mag[0], mag_y=self._mag[1], mag_z=self._mag[2],
            quat_w=self._quat[0], quat_x=self._quat[1], quat_y=self._quat[2], quat_z=self._quat[3],
            temperature=self._temperature,
            pressure=self._pressure, height=self._height,
            lon=self._lon, lat=self._lat, yaw_gps=self._yaw_gps, speed=self._speed,
            chip_time=self._chip_time,
        )
        for cb in self._callbacks:
            try:
                cb(d)
            except Exception as e:
                print(f"[Parser] callback error: {e}")

    # ── Property accessors ─────────────────────────────────────────

    @property
    def accel(self):
        return tuple(self._accel)

    @property
    def gyro(self):
        return tuple(self._gyro)

    @property
    def angle(self):
        return tuple(self._angle)

    @property
    def quat(self):
        return tuple(self._quat)

    @property
    def mag(self):
        return tuple(self._mag)

    @property
    def temperature(self):
        return self._temperature

    @property
    def find_values(self):
        return list(self._find_values)


def gravity_angles(acc_x: float, acc_y: float, acc_z: float):
    """Compute roll and pitch from gravity vector (static accelerometer)."""
    pitch = math.atan2(acc_x, math.sqrt(acc_y ** 2 + acc_z ** 2))
    roll = math.atan2(acc_y, math.sqrt(acc_x ** 2 + acc_z ** 2))
    return math.degrees(roll), math.degrees(pitch)


def is_static(accel_history: np.ndarray, threshold: float = 0.02) -> bool:
    """Check if accelerometer readings are stable (static)."""
    if accel_history.shape[0] < 10:
        return False
    recent = accel_history[-10:]
    return float(np.std(recent, axis=0).max()) < threshold


def quat_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_inverse(q):
    """Inverse of a unit quaternion [w, x, y, z]."""
    return [q[0], -q[1], -q[2], -q[3]]


def quat_to_euler(q):
    """Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in degrees."""
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def euler_to_quat(yaw_deg, pitch_deg, roll_deg):
    """Convert Euler angles (yaw, pitch, roll in degrees) to quaternion [w, x, y, z].
    Rotation order: ZYX (yaw then pitch then roll)."""
    import math
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def quat_rotation_angle(q):
    """Extract scalar rotation angle from quaternion [w, x, y, z] in degrees.
    theta = 2 * acos(|w|), the total rotation magnitude."""
    import math
    w = abs(q[0])
    w = max(-1.0, min(1.0, w))
    return math.degrees(2.0 * math.acos(w))



# ── Register command builders ──────────────────────────────────────

def build_read_reg(reg_addr: int) -> bytes:
    """Build command to read a register: FF AA 27 addr_lo addr_hi"""
    return bytes([0xFF, 0xAA, 0x27, reg_addr & 0xFF, reg_addr >> 8])


def build_write_reg(reg_addr: int, value: int) -> bytes:
    """Build command to write a register: FF AA reg_addr value_lo value_hi"""
    return bytes([0xFF, 0xAA, reg_addr, value & 0xFF, value >> 8])


def build_unlock() -> bytes:
    """Unlock command (0x69, 0xB588)."""
    return build_write_reg(0x69, 0xB588)


def build_save() -> bytes:
    """Save configuration command."""
    return build_write_reg(0x00, 0x00)


def build_accel_calibration() -> bytes:
    """Acceleration calibration command."""
    return build_write_reg(0x01, 0x01)


def build_begin_field_calibration() -> bytes:
    """Begin magnetic field calibration command."""
    return build_write_reg(0x01, 0x07)
