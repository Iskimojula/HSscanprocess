# -*- coding: utf-8 -*-
"""陀螺仪（HWT906P）采集服务。

设计要点：
    * 主线程零阻塞：打开/关闭串口、端口枚举等耗时操作全部在子线程执行，
      start() / stop() 立即返回；
    * 线程安全：所有共享状态由 threading.Lock 保护，UI 只通过
      GyroService.get_snapshot() 读取不可变快照；
    * UI 通知用有界队列：queue.Queue(maxsize=1) 只保留"有新数据"这一个信号，
      UI 用 after 轮询消费，永远不会积压也不会跨线程操作 tkinter；
    * 记录/清零语义：记录 = 锁定当前姿态为初始姿态 (Yaw1, Pitch1, Roll1)；
      清零 = 丢弃基准，theta 回到未测量状态（显示 -- ，而不是 0）。
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from services import rotation_math as rm

# 超过该秒数没有新数据即视为"数据超时"
STALE_AFTER_S = 1.0
DEFAULT_BAUD = 115200


@dataclass(frozen=True)
class GyroSample:
    """一帧陀螺仪数据（角度单位：度）。"""

    timestamp: float
    yaw: float
    pitch: float
    roll: float
    quat: Tuple[float, float, float, float]
    has_quat: bool
    temperature: float = 0.0

    def as_quaternion(self) -> list:
        """姿态四元数：优先用传感器输出的四元数，没有则用欧拉角换算。"""
        if self.has_quat:
            return rm.normalize(self.quat)
        return rm.euler_to_quaternion(self.yaw, self.pitch, self.roll)


@dataclass(frozen=True)
class AngleTriple:
    """三方向角度。"""

    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class GyroSnapshot:
    """供 UI 一次性读取的不可变状态快照。"""

    connected: bool
    connecting: bool
    port: str
    status_text: str
    error: str
    current: Optional[AngleTriple]
    reference: Optional[AngleTriple]
    theta_deg: Optional[float]
    axis: Optional[Tuple[float, float, float]]
    rate_hz: float
    temperature: Optional[float]
    fresh: bool


class SensorAdapter:
    """传感器适配层协议：便于用假设备替换真设备做测试。"""

    def open(self, port: str, baud: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def read(self) -> Optional[GyroSample]:
        raise NotImplementedError

    def set_callback(self, fn: Callable[[GyroSample], None]) -> None:
        raise NotImplementedError


def _sdk_data_to_sample(data) -> Optional[GyroSample]:
    """把 HWT906P SDK 的 IMUData 映射成 GyroSample。"""
    if data is None:
        return None
    quat = (
        float(getattr(data, "quat_w", 1.0)),
        float(getattr(data, "quat_x", 0.0)),
        float(getattr(data, "quat_y", 0.0)),
        float(getattr(data, "quat_z", 0.0)),
    )
    has_quat = any(abs(v) > 1e-6 for v in quat[1:]) or abs(abs(quat[0]) - 1.0) > 1e-6
    return GyroSample(
        timestamp=float(getattr(data, "timestamp", time.time())),
        yaw=float(getattr(data, "angle_z", 0.0)),
        pitch=float(getattr(data, "angle_y", 0.0)),
        roll=float(getattr(data, "angle_x", 0.0)),
        quat=quat,
        has_quat=has_quat,
        temperature=float(getattr(data, "temperature", 0.0)),
    )


def _default_sdk_factory():
    """懒加载官方 SDK：没装 SDK / pyserial 也不影响程序启动。"""
    from hwt906p import HWT906P

    return HWT906P()


class HWT906PAdapter(SensorAdapter):
    """HWT906P 官方 SDK 适配层（不修改 SDK 源码）。"""

    def __init__(self, device_factory: Optional[Callable[[], object]] = None):
        self._factory = device_factory or _default_sdk_factory
        self._device = None
        self._callback = None

    def set_callback(self, fn: Callable[[GyroSample], None]) -> None:
        self._callback = fn
        if self._device is not None:
            self._device.on_data_update(self._wrap(fn))

    def open(self, port: str, baud: int) -> None:
        try:
            self._device = self._factory()
            self._device.open(port, baud)
        except ImportError as exc:
            raise ConnectionError("未找到 hwt906p SDK（%s）" % exc) from exc
        except ConnectionError:
            raise
        except Exception as exc:  # 串口被占用、设备无响应等
            raise ConnectionError(str(exc)) from exc
        if self._callback is not None:
            self._device.on_data_update(self._wrap(self._callback))

    def close(self) -> None:
        device, self._device = self._device, None
        if device is not None:
            try:
                device.close()
            except Exception:
                pass

    def read(self) -> Optional[GyroSample]:
        if self._device is None:
            return None
        return _sdk_data_to_sample(self._device.get_all_data())

    @staticmethod
    def _wrap(fn: Callable[[GyroSample], None]):
        def handler(data):
            sample = _sdk_data_to_sample(data)
            if sample is not None:
                fn(sample)

        return handler


class GyroService:
    """陀螺仪服务：设备生命周期 + 后台数据流 + 记录/清零状态机。"""

    def __init__(self, adapter: Optional[SensorAdapter] = None):
        self._adapter = adapter if adapter is not None else HWT906PAdapter()
        self._lock = threading.RLock()
        self._events = queue.Queue(maxsize=1)

        self._port = ""
        self._baud = DEFAULT_BAUD
        self._connected = False
        self._connecting = False
        self._error = ""

        self._current: Optional[AngleTriple] = None
        self._last_sample: Optional[GyroSample] = None
        self._last_rx_time = 0.0
        self._temperature: Optional[float] = None

        self._reference: Optional[AngleTriple] = None
        self._q_ref: Optional[list] = None
        self._theta: Optional[float] = None
        self._axis: Optional[Tuple[float, float, float]] = None

        self._rate_hz = 0.0
        self._rate_count = 0
        self._rate_t0 = 0.0
        self._stop_requested = False

    # ── 生命周期 ───────────────────────────────────────────────────

    def start(self, port: str, baud: int = DEFAULT_BAUD) -> None:
        """启动采集。立即返回，串口打开在子线程完成。"""
        with self._lock:
            self._stop_requested = False
            self._port = port
            self._baud = baud
            self._connecting = True
            self._connected = False
            self._error = ""
            self._reference = None
            self._q_ref = None
            self._theta = None
            self._axis = None
        self._put_event()
        thread = threading.Thread(
            target=self._connect_worker,
            args=(port, baud),
            daemon=True,
            name="GyroService-Connect",
        )
        thread.start()

    def stop(self) -> None:
        """停止采集并释放串口。立即返回，关闭动作在子线程完成。"""
        with self._lock:
            self._stop_requested = True
            self._connected = False
            self._connecting = False
        self._put_event()
        threading.Thread(target=self._close_worker, daemon=True, name="GyroService-Close").start()

    def _connect_worker(self, port: str, baud: int) -> None:
        try:
            self._adapter.set_callback(self._on_sample)
            self._adapter.open(port, baud)
        except Exception as exc:
            with self._lock:
                self._connecting = False
                self._connected = False
                self._error = "连接 %s 失败：%s" % (port, exc)
            self._put_event()
            return
        with self._lock:
            self._connecting = False
            if self._stop_requested:
                self._connected = False
                self._error = ""
            else:
                self._connected = True
                self._error = ""
                self._last_rx_time = time.time()
        self._put_event()
        if self._stop_requested:
            try:
                self._adapter.close()
            except Exception:
                pass
            return
        first = self._adapter.read()
        if first is not None:
            self._on_sample(first)

    def _close_worker(self) -> None:
        try:
            self._adapter.close()
        except Exception:
            pass
        self._put_event()

    # ── 数据流 ─────────────────────────────────────────────────────

    def _on_sample(self, sample: GyroSample) -> None:
        """传感器读线程回调：只做 O(1) 计算与入队，绝不触碰 UI。"""
        now = time.time()
        with self._lock:
            self._last_sample = sample
            self._current = AngleTriple(sample.yaw, sample.pitch, sample.roll)
            self._last_rx_time = now
            if sample.temperature:
                self._temperature = sample.temperature
            if self._q_ref is not None:
                result = rm.relative_rotation(self._q_ref, sample.as_quaternion())
                self._theta = result.theta_deg
                self._axis = result.axis
            if self._rate_t0 == 0.0:
                self._rate_t0 = now
            self._rate_count += 1
            elapsed = now - self._rate_t0
            if elapsed >= 1.0:
                self._rate_hz = self._rate_count / elapsed
                self._rate_count = 0
                self._rate_t0 = now
        self._put_event()

    def _put_event(self) -> None:
        """向 UI 发一个"有新数据"的信号；只保留最新一个，永不积压。"""
        try:
            self._events.put_nowait(True)
        except queue.Full:
            try:
                self._events.get_nowait()
            except queue.Empty:
                pass
            try:
                self._events.put_nowait(True)
            except queue.Full:
                pass

    def drain_ui_events(self) -> bool:
        """UI 轮询：返回自上次调用以来是否有新状态。"""
        try:
            self._events.get_nowait()
            return True
        except queue.Empty:
            return False

    # ── 记录 / 清零 ────────────────────────────────────────────────

    def record_reference(self) -> bool:
        """锁定当前姿态为初始姿态 (Yaw1, Pitch1, Roll1)。"""
        with self._lock:
            if self._current is None or self._last_sample is None:
                self._error = "暂无陀螺仪数据，无法记录"
                ok = False
            else:
                self._reference = self._current
                self._q_ref = self._last_sample.as_quaternion()
                self._theta = 0.0
                self._axis = (0.0, 0.0, 0.0)
                self._error = ""
                ok = True
        self._put_event()
        return ok

    def clear_reference(self) -> None:
        """清除基准：theta 回到未测量状态。"""
        with self._lock:
            self._reference = None
            self._q_ref = None
            self._theta = None
            self._axis = None
            self._error = ""
        self._put_event()

    # ── 状态查询 ───────────────────────────────────────────────────

    def get_snapshot(self) -> GyroSnapshot:
        with self._lock:
            fresh = (
                self._current is not None
                and (time.time() - self._last_rx_time) < STALE_AFTER_S
            )
            if self._connecting:
                status = "连接中…"
            elif not self._connected:
                status = "未连接"
            elif self._current is None:
                status = "已连接，等待数据"
            elif not fresh:
                status = "数据超时"
            else:
                status = "已连接"
            return GyroSnapshot(
                connected=self._connected,
                connecting=self._connecting,
                port=self._port,
                status_text=status,
                error=self._error,
                current=self._current,
                reference=self._reference,
                theta_deg=self._theta,
                axis=self._axis,
                rate_hz=self._rate_hz,
                temperature=self._temperature,
                fresh=fresh,
            )

    @staticmethod
    def available_ports() -> list:
        """枚举可用串口；任何异常都兜底为空列表，避免拖垮 UI。"""
        try:
            from serial.tools import list_ports

            return [p.device for p in list_ports.comports()]
        except Exception:
            return []
