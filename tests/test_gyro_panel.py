# -*- coding: utf-8 -*-
"""陀螺仪面板测试：用假服务验证布局文案与交互，不需要硬件。"""

import time

import pytest

tk = pytest.importorskip("tkinter")

from services.gyro_service import GyroService
from tests.fakes import FakeAdapter
from views.gyro_panel import GyroPanel


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("无图形环境")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def wait_for(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_format_triple_is_fixed_width():
    assert GyroPanel.format_triple(10.234, -5.4, 3.0) == "   10.23 /    -5.40 /     3.00"


def test_format_triple_fits_three_digit_negative_values():
    #-179.99 这类值过去会被 6 字符定宽截断（显示成 -4.7x），必须完整显示
    text = GyroPanel.format_triple(-179.99, 179.99, -100.5)
    assert "-179.99" in text and "179.99" in text and "-100.50" in text


def test_format_triple_placeholder_for_missing_values():
    assert GyroPanel.format_triple(None, None, None) == "-- / -- / --"


def test_panel_shows_dashes_before_data(root):
    svc = GyroService(adapter=FakeAdapter())
    panel = GyroPanel(root, svc)
    panel.refresh()
    assert panel.var_theta.get() == "--"
    assert panel.var_reference.get().count("--") == 3
    assert panel.var_current.get().count("--") == 3
    svc.stop()


def test_panel_updates_after_record(root):
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    ad.feed(10.0, -5.0, 3.0)
    assert wait_for(lambda: svc.get_snapshot().current is not None)
    panel = GyroPanel(root, svc)
    assert panel.on_record() is True
    panel.refresh()
    assert "10.00" in panel.var_reference.get()
    assert panel.var_theta.get() != "--"
    assert "已记录" in panel.var_status.get()
    panel.on_clear()
    panel.refresh()
    assert panel.var_theta.get() == "--"
    assert "已清零" in panel.var_status.get()
    svc.stop()


def test_record_without_data_shows_reason(root):
    svc = GyroService(adapter=FakeAdapter())
    panel = GyroPanel(root, svc)
    assert panel.on_record() is False
    assert "无法记录" in panel.var_message.get()
    svc.stop()


def test_record_and_clear_are_fast(root):
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST", 115200)
    ad.feed(1.0, 2.0, 3.0)
    assert wait_for(lambda: svc.get_snapshot().current is not None)
    panel = GyroPanel(root, svc)
    t0 = time.time()
    panel.on_record()
    panel.on_clear()
    assert time.time() - t0 < 0.05
    svc.stop()


def test_connect_toggle_does_not_block(root):
    ad = FakeAdapter(open_delay=0.3, auto_feed=(0.0, 0.0, 0.0))
    svc = GyroService(adapter=ad)
    panel = GyroPanel(root, svc)
    panel.var_port.set("COM_TEST")
    panel.var_baud.set("115200")
    t0 = time.time()
    panel.on_connect_toggle()
    assert time.time() - t0 < 0.05
    assert panel.btn_connect.cget("state") == "disabled"
    assert wait_for(lambda: svc.get_snapshot().connected)
    panel.refresh()
    assert panel.btn_connect.cget("state") == "normal"
    assert "已连接" in panel.var_status.get()
    svc.stop()
