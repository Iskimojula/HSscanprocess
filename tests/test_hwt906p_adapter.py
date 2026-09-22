# -*- coding: utf-8 -*-
"""HWT906P 适配层测试：用假 SDK 对象替换真设备，无需硬件。"""

from services.gyro_service import GyroSample, HWT906PAdapter, _default_sdk_factory


class FakeSdkData:
    def __init__(self, **kw):
        self.timestamp = kw.get("timestamp", 1.0)
        self.angle_x = kw.get("roll", 1.0)
        self.angle_y = kw.get("pitch", 2.0)
        self.angle_z = kw.get("yaw", 3.0)
        self.quat_w, self.quat_x, self.quat_y, self.quat_z = kw.get(
            "quat", (1.0, 0.0, 0.0, 0.0)
        )
        self.temperature = 25.0


class FakeSdkDevice:
    def __init__(self):
        self.opened = None
        self.cbs = []
        self.closed = False

    def open(self, port, baud):
        self.opened = (port, baud)
        return True

    def close(self):
        self.closed = True

    def on_data_update(self, cb):
        self.cbs.append(cb)

    def get_all_data(self):
        return FakeSdkData()


def test_adapter_maps_sdk_fields_to_sample():
    dev = FakeSdkDevice()
    ad = HWT906PAdapter(device_factory=lambda: dev)
    received = []
    ad.set_callback(received.append)
    ad.open("COM9", 115200)
    assert dev.opened == ("COM9", 115200)
    assert len(dev.cbs) == 1
    dev.cbs[0](FakeSdkData(roll=-1.5, pitch=0.5, yaw=179.0))
    assert received and isinstance(received[0], GyroSample)
    s = received[0]
    assert (s.roll, s.pitch, s.yaw) == (-1.5, 0.5, 179.0)
    assert s.has_quat is False          # (1,0,0,0) 视为没有四元数，用欧拉角兜底
    assert ad.read() is not None
    ad.close()
    assert dev.closed


def test_adapter_marks_real_quaternion_as_available():
    dev = FakeSdkDevice()
    ad = HWT906PAdapter(device_factory=lambda: dev)
    received = []
    ad.set_callback(received.append)
    ad.open("COM9", 115200)
    dev.cbs[0](FakeSdkData(quat=(0.9239, 0.0, 0.0, 0.3827)))
    assert received[0].has_quat is True
    ad.close()


def test_adapter_reports_missing_sdk():
    def boom():
        raise ImportError("no hwt906p")

    ad = HWT906PAdapter(device_factory=boom)
    try:
        ad.open("COM1", 115200)
    except ConnectionError as exc:
        assert "hwt906p" in str(exc)
    else:
        raise AssertionError("应当抛出可读错误")


def test_vendored_sdk_is_importable():
    dev = _default_sdk_factory()
    assert hasattr(dev, "open") and hasattr(dev, "get_all_data")
