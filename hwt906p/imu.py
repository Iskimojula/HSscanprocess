# -*- coding: utf-8 -*-
"""
HWT906P IMU sensor interface.
Provides hardware initialization, status reading, angle access, and
relative-origin angle tracking via quaternion differencing.
"""

import time
import threading
import math
from typing import Callable, Optional

import serial
import serial.tools.list_ports
import numpy as np

from .protocol import (
    WitProtocolParser,
    IMUData,
    gravity_angles,
    is_static,
    quat_multiply,
    quat_inverse,
    quat_to_euler,
    quat_rotation_angle,
    euler_to_quat,
    build_read_reg,
    build_write_reg,
    build_unlock,
    build_save,
    REG_RSW,
    REG_RRATE,
    REG_BAUD,
    REG_AXIS_DIR,
    REG_ALGORITHM,
)


class HWT906P:
    """HWT906P 6-axis/9-axis IMU sensor interface via serial (WitStandardProtocol).

    Coordinate system (WitMotion convention):
        X-axis = Roll  (rotation around X, forward/backward tilt)
        Y-axis = Pitch (rotation around Y, left/right tilt)
        Z-axis = Yaw   (rotation around Z, horizontal rotation)
        Range: +/- 180 degrees

    Usage:
        imu = HWT906P()
        imu.open("COM3", baud=115200)
        print(imu.status)
        print(imu.angle_x, imu.angle_y, imu.angle_z)
        imu.zero_calibrate()
        print(imu.relative_angle_x, imu.relative_angle_y, imu.relative_angle_z)
        imu.close()
    """

    def __init__(self):
        self._parser = WitProtocolParser()
        self._serial: Optional[serial.Serial] = None
        self._port = ""
        self._baud = 115200
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Latest data snapshot (thread-safe)
        self._latest: Optional[IMUData] = None

        # Callback list
        self._user_callbacks: list = []

        # Acceleration history for static detection (ring buffer)
        self._accel_history = np.zeros((50, 3))

        # Relative-origin tracking
        self._q_ref = [1.0, 0.0, 0.0, 0.0]  # reference quaternion (identity)
        self._zeroed = False

        # Data rate estimation
        self._data_count = 0
        self._data_t0 = time.time()
        self._data_rate_hz = 0.0

        # Register parser callback
        self._parser.register_callback(self._on_data)

    # ── Connection Management ──────────────────────────────────────

    def open(self, port: str = "COM3", baud: int = 115200) -> bool:
        """Open serial connection and start data acquisition thread."""
        self._port = port
        self._baud = baud

        try:
            self._serial = serial.Serial(port, baud, timeout=0.5)
        except serial.SerialException as e:
            raise ConnectionError(f"Cannot open {port} at {baud} bps: {e}")

        self._running = True
        self._data_t0 = time.time()
        self._data_count = 0

        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="HWT906P-Reader")
        self._thread.start()

        # Give sensor time to start streaming
        time.sleep(0.5)

        # Read config
        self._print_config()
        return True

    def close(self):
        """Stop data thread and close serial port."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    # ── Register / Config ──────────────────────────────────────────

    def read_register(self, reg_addr: int, count: int = 4, timeout: float = 1.0) -> list:
        """Read one or more register values from the sensor."""
        if not self.is_connected:
            return []
        cmd = build_read_reg(reg_addr)
        self._serial.write(cmd)
        # Wait for find response
        t0 = time.time()
        while len(self._parser.find_values) < count:
            if time.time() - t0 > timeout:
                break
            time.sleep(0.05)
        return self._parser.find_values[:count]

    def write_register(self, reg_addr: int, value: int):
        """Write a value to a register (unlock first)."""
        if not self.is_connected:
            return
        self._serial.write(build_unlock())
        time.sleep(0.1)
        self._serial.write(build_write_reg(reg_addr, value))

    def save_config(self):
        """Save current configuration to flash."""
        if self.is_connected:
            self._serial.write(build_unlock())
            time.sleep(0.1)
            self._serial.write(build_save())

    def _print_config(self):
        """Read and print current device configuration."""
        config_desc = {
            REG_RSW: ("Data output content", "0x02"),
            REG_RRATE: ("Return rate", "0x03"),
            REG_BAUD: ("Baud rate code", "0x04"),
            REG_AXIS_DIR: ("Install direction", "0x23"),
            REG_ALGORITHM: ("Algorithm (9/6 axis)", "0x24"),
        }
        for reg, (name, hex_addr) in config_desc.items():
            vals = self.read_register(reg, count=1, timeout=0.5)
            if vals:
                print(f"  [{hex_addr}] {name}: {vals[0]}")
            else:
                print(f"  [{hex_addr}] {name}: (no response)")

    # ── Data Thread ────────────────────────────────────────────────

    def _read_loop(self):
        """Background thread: read serial bytes and feed to parser."""
        while self._running:
            try:
                if self._serial and self._serial.is_open:
                    n = self._serial.in_waiting
                    if n > 0:
                        raw = self._serial.read(n)
                        self._parser.feed(raw)
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.1)
            except (serial.SerialException, OSError):
                self._running = False
                break

    def _on_data(self, data: IMUData):
        """Internal callback from parser when new data set arrives."""
        self._latest = data

        # Update accel history ring buffer
        self._accel_history = np.roll(self._accel_history, -1, axis=0)
        self._accel_history[-1] = [data.accel_x, data.accel_y, data.accel_z]

        # Data rate estimation
        self._data_count += 1
        now = time.time()
        if now - self._data_t0 >= 1.0:
            self._data_rate_hz = self._data_count / (now - self._data_t0)
            self._data_count = 0
            self._data_t0 = now

        # Fire user callbacks
        for cb in self._user_callbacks:
            try:
                cb(data)
            except Exception:
                pass

    # ── Status ─────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        """Return current device status."""
        d = self._latest
        return {
            "connected": self.is_connected,
            "port": self._port,
            "baud": self._baud,
            "data_rate_hz": round(self._data_rate_hz, 1),
            "temperature_c": round(d.temperature, 2) if d else None,
            "chip_time": d.chip_time if d else None,
            "zeroed": self._zeroed,
        }

    def get_all_data(self) -> Optional[IMUData]:
        """Return the latest full IMUData snapshot."""
        return self._latest

    # ── Convenience Property Accessors ─────────────────────────────

    @property
    def angle_x(self) -> float:
        return self._latest.angle_x if self._latest else 0.0

    @property
    def angle_y(self) -> float:
        return self._latest.angle_y if self._latest else 0.0

    @property
    def angle_z(self) -> float:
        return self._latest.angle_z if self._latest else 0.0

    @property
    def gyro_x(self) -> float:
        return self._latest.gyro_x if self._latest else 0.0

    @property
    def gyro_y(self) -> float:
        return self._latest.gyro_y if self._latest else 0.0

    @property
    def gyro_z(self) -> float:
        return self._latest.gyro_z if self._latest else 0.0

    @property
    def accel_x(self) -> float:
        return self._latest.accel_x if self._latest else 0.0

    @property
    def accel_y(self) -> float:
        return self._latest.accel_y if self._latest else 0.0

    @property
    def accel_z(self) -> float:
        return self._latest.accel_z if self._latest else 0.0

    @property
    def quaternion(self) -> tuple:
        if self._latest:
            return (self._latest.quat_w, self._latest.quat_x,
                    self._latest.quat_y, self._latest.quat_z)
        return (1.0, 0.0, 0.0, 0.0)

    @property
    def temperature(self) -> float:
        return self._latest.temperature if self._latest else 0.0

    @property
    def is_static(self) -> bool:
        return is_static(self._accel_history)

    @property
    def gravity_roll_pitch(self) -> tuple:
        """Compute roll, pitch from gravity (valid when static)."""
        if self._latest is None:
            return (0.0, 0.0)
        return gravity_angles(self._latest.accel_x, self._latest.accel_y, self._latest.accel_z)

    # ── Callbacks ──────────────────────────────────────────────────

    def on_data_update(self, callback: Callable[[IMUData], None]):
        """Register a callback invoked on every complete data set."""
        self._user_callbacks.append(callback)

    # ── Zero Calibration (Relative Origin) ─────────────────────────

    def zero_calibrate(self):
        """Set current orientation as origin (0,0,0).

        Falls back to Euler-to-Quat if sensor does not output quaternion data.
        """
        q = list(self.quaternion)
        if q == [1.0, 0.0, 0.0, 0.0] and self._latest is not None:
            d = self._latest
            q = euler_to_quat(d.angle_z, d.angle_y, d.angle_x)
            print(f"[Zero] Quat unavailable, Euler fallback: yaw={d.angle_z:.1f} pitch={d.angle_y:.1f} roll={d.angle_x:.1f}")
        self._q_ref = q
        self._zeroed = True
        print(f"[Zero] Origin set at quaternion: ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")

    @property
    def relative_angle_x(self) -> float:
        return self._relative_angles[0]

    @property
    def relative_angle_y(self) -> float:
        return self._relative_angles[1]

    @property
    def relative_angle_z(self) -> float:
        return self._relative_angles[2]

    @property
    def _relative_angles(self) -> tuple:
        """Compute Euler angles relative to the zero-reference quaternion.

        Uses quaternion differencing: q_rel = q_cur * inverse(q_ref)
        This naturally handles +/- 180 deg wrap-around and gimbal lock.
        """
        if not self._zeroed or self._latest is None:
            return (0.0, 0.0, 0.0)
        q_cur = list(self.quaternion)
        if q_cur == [1.0, 0.0, 0.0, 0.0] and self._latest is not None:
            d = self._latest
            q_cur = euler_to_quat(d.angle_z, d.angle_y, d.angle_x)
        q_inv_ref = quat_inverse(self._q_ref)
        q_rel = quat_multiply(q_cur, q_inv_ref)
        # Normalize
        norm = math.sqrt(sum(x * x for x in q_rel))
        if norm > 1e-9:
            q_rel = [x / norm for x in q_rel]
        return quat_to_euler(q_rel)

    @property
    def rotation_angle(self):
        """Scalar rotation angle theta from zero-ref to current pose (degrees).
        Uses quaternion differencing: q_diff = q_cur * inverse(q_ref),
        then theta = quat_rotation_angle(q_diff)."""
        if not self._zeroed or self._latest is None:
            return 0.0
        q_cur = list(self.quaternion)
        if q_cur == [1.0, 0.0, 0.0, 0.0] and self._latest is not None:
            d = self._latest
            q_cur = euler_to_quat(d.angle_z, d.angle_y, d.angle_x)
        q_inv_ref = quat_inverse(self._q_ref)
        q_diff = quat_multiply(q_cur, q_inv_ref)
        norm = math.sqrt(sum(x * x for x in q_diff))
        if norm > 1e-9:
            q_diff = [x / norm for x in q_diff]
        return quat_rotation_angle(q_diff)


# ── Helper: list available serial ports ────────────────────────────

def list_ports() -> list:
    """Return list of available COM port device names."""
    return [p.device for p in serial.tools.list_ports.comports()]
