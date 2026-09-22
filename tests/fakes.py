# -*- coding: utf-8 -*-
"""测试替身：不接硬件也能跑通服务层与视图层。"""

import time

from services.gyro_service import GyroSample


class FakeAdapter:
    """可编程的假传感器。"""

    def __init__(self, fail_open=False, open_delay=0.0):
        self.fail_open = fail_open
        self.opened = False
        self.closed = False
        self.callback = None
        self.sample = None
        self.open_delay = open_delay
        self.open_error = None

    def open(self, port, baud):
        time.sleep(self.open_delay)
        if self.fail_open:
            raise ConnectionError("port busy")
        if self.open_error:
            raise self.open_error
        self.opened = True

    def close(self):
        self.closed = True

    def set_callback(self, fn):
        self.callback = fn

    def read(self):
        return self.sample

    def feed(self, yaw, pitch, roll, quat=None):
        """模拟传感器推送一帧数据。"""
        self.sample = GyroSample(
            timestamp=time.time(),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            quat=quat if quat is not None else (1.0, 0.0, 0.0, 0.0),
            has_quat=quat is not None,
        )
        if self.callback:
            self.callback(self.sample)
        return self.sample
