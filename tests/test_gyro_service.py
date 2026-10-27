# -*- coding: utf-8 -*-
import time

import services.rotation_math as rm
import services.gyro_service as gs
from services.gyro_service import GyroService
from tests.fakes import FakeAdapter


def wait_for(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_start_is_non_blocking_and_connects():
    ad = FakeAdapter(open_delay=0.3, auto_feed=(10.0, -5.0, 3.0))
    svc = GyroService(adapter=ad)
    t0 = time.time()
    svc.start("COM_TEST", 115200)
    elapsed = time.time() - t0
    assert elapsed < 0.05, "start() 阻塞了 %.3fs" % elapsed
    assert wait_for(lambda: svc.get_snapshot().connected)
    svc.stop()


def test_auto_baud_uses_detected_value():
    ad = FakeAdapter(auto_feed=(0.0, 0.0, 0.0))
    svc = GyroService(adapter=ad, baud_detector=lambda port: 921600)
    svc.start("COM_TEST")                     # 不指定波特率 -> 走自动识别
    assert wait_for(lambda: svc.get_snapshot().connected)
    assert ad.baud == 921600
    assert svc.get_snapshot().baud == 921600
    svc.stop()


def test_auto_baud_failure_lists_candidates():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad, baud_detector=lambda port: None)
    svc.start("COM_TEST")
    assert wait_for(lambda: svc.get_snapshot().error)
    snap = svc.get_snapshot()
    assert ad.opened is False
    for candidate in gs.BAUD_CANDIDATES:
        assert str(candidate) in snap.error
    svc.stop()


def test_open_but_no_frames_reports_error(monkeypatch):
    monkeypatch.setattr(gs, "PROBE_TIMEOUT_S", 0.2)
    ad = FakeAdapter()                        # 打开成功但一帧都不上报
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    assert wait_for(lambda: svc.get_snapshot().error)
    snap = svc.get_snapshot()
    assert snap.connected is False and "没有收到数据帧" in snap.error
    assert ad.closed is True
    svc.stop()


def test_theta_none_before_record():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    ad.feed(10, -5, 3)
    assert wait_for(lambda: svc.get_snapshot().current is not None)
    snap = svc.get_snapshot()
    assert snap.reference is None
    assert snap.theta_deg is None
    svc.stop()


def test_record_then_rotate_gives_theta():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    ad.feed(10, -5, 3)
    assert wait_for(lambda: svc.get_snapshot().current is not None)
    assert svc.record_reference() is True
    ref = svc.get_snapshot().reference
    assert (round(ref.yaw), round(ref.pitch), round(ref.roll)) == (10, -5, 3)
    q = rm.euler_to_quaternion(10, -5, 3)
    q_turned = rm.multiply(rm.axis_angle_to_quaternion((0, 0, 1), 30), q)
    ad.feed(40, -5, 3, quat=q_turned)
    assert wait_for(lambda: abs((svc.get_snapshot().theta_deg or 0.0) - 30.0) < 0.5)
    svc.stop()


def test_record_uses_euler_fallback_when_no_quaternion():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    ad.feed(0, 0, 0)
    assert wait_for(lambda: svc.record_reference() is True)
    ad.feed(0, 0, 45)          # 无四元数数据包，用欧拉角兜底
    assert wait_for(lambda: abs((svc.get_snapshot().theta_deg or 0.0) - 45.0) < 0.5)
    svc.stop()


def test_record_without_data_returns_false():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    # 一帧数据都还没到，直接点"记录"必须失败并给出原因
    assert svc.record_reference() is False
    assert "无法记录" in svc.get_snapshot().error
    svc.stop()


def test_clear_reference_resets_theta():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    ad.feed(0, 0, 0)
    assert wait_for(lambda: svc.record_reference() is True)
    svc.clear_reference()
    snap = svc.get_snapshot()
    assert snap.reference is None and snap.theta_deg is None
    svc.stop()


def test_open_failure_is_reported_not_raised():
    ad = FakeAdapter(fail_open=True)
    svc = GyroService(adapter=ad)
    svc.start("COM_BAD", 115200)
    assert wait_for(lambda: svc.get_snapshot().error)
    snap = svc.get_snapshot()
    assert snap.connected is False and "COM_BAD" in snap.error
    svc.stop()


def test_stop_closes_adapter_in_background():
    ad = FakeAdapter(auto_feed=(0.0, 0.0, 0.0))
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    assert wait_for(lambda: svc.get_snapshot().connected)
    t0 = time.time()
    svc.stop()
    assert time.time() - t0 < 0.05
    assert wait_for(lambda: ad.closed)


def test_drain_ui_events_reports_new_data():
    ad = FakeAdapter(auto_feed=(0.0, 0.0, 0.0))
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    assert wait_for(lambda: svc.get_snapshot().connected)
    svc.drain_ui_events()
    ad.feed(1, 2, 3)
    assert wait_for(lambda: svc.drain_ui_events() is True)
    assert svc.drain_ui_events() is False
    svc.stop()
