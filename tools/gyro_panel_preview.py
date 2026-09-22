# -*- coding: utf-8 -*-
"""无硬件预览：用模拟陀螺仪数据展示"陀螺仪 HWT906P"面板的布局与交互。

用途：没有接 HWT906P 时也能检查界面布局、记录/清零按钮与 theta 刷新是否正常。

    python tools/gyro_panel_preview.py
"""

import math
import os
import sys
import threading
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import rotation_math as rm  # noqa: E402
from services.gyro_service import GyroSample, GyroService, SensorAdapter  # noqa: E402
from views.gyro_panel import GyroPanel  # noqa: E402


class SimulatedAdapter(SensorAdapter):
    """模拟绕 Z 轴匀速转动、同时有小幅俯仰/滚转的陀螺仪。"""

    def __init__(self, period_s=10.0, rate_hz=50.0):
        self.period_s = period_s
        self.interval = 1.0 / rate_hz
        self._callback = None
        self._running = False
        self._thread = None

    def set_callback(self, fn):
        self._callback = fn

    def open(self, port, baud):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="Sim-I MU")
        self._thread.start()

    def close(self):
        self._running = False

    def read(self):
        return None

    def _run(self):
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            yaw = (360.0 * t / self.period_s + 180.0) % 360.0 - 180.0
            pitch = 6.0 * math.sin(2 * math.pi * t / 5.0)
            roll = 4.0 * math.cos(2 * math.pi * t / 7.0)
            quat = rm.euler_to_quaternion(yaw, pitch, roll)
            sample = GyroSample(
                timestamp=time.time(),
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                quat=tuple(quat),
                has_quat=True,
                temperature=25.6,
            )
            if self._callback is not None:
                self._callback(sample)
            time.sleep(self.interval)


def main():
    root = tk.Tk()
    root.title("陀螺仪面板预览（模拟数据）")
    service = GyroService(adapter=SimulatedAdapter())
    panel = GyroPanel(root, service)
    panel.pack(fill="x", padx=12, pady=12)
    tk.Label(
        root,
        text="① 点「记录」锁定初始角度 ② 观察转动角 theta 变化 ③ 点「清零」回到未测量状态",
        justify="left",
        anchor="w",
    ).pack(fill="x", padx=12, pady=(0, 12))
    service.start("SIM")
    try:
        root.mainloop()
    finally:
        service.stop()


if __name__ == "__main__":
    main()
