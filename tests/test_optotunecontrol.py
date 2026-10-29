# -*- coding: utf-8 -*-
"""振镜驱动层：用户坐标 (SETX, SETY) 与两路通道之间的实机标定关系。

标定规则（2026-10-29 实机确认）：
    channel_0 = -SETY，channel_1 = SETX
即坐标轴互换并取反，等价于旋转 90°。所以界面上输入 SETX=2.2 / SETY=-0.7
打出的振镜状态，与旧版输入 SETX=0.7 / SETY=2.2 完全一样。
"""

import pytest

import optotunecontrol
from optotunecontrol import deg_to_xy


class _FakeStaticInput:
    def __init__(self):
        self.value = 0.0

    def SetXY(self, value):
        self.value = float(value)

    def GetXY(self):
        return (self.value,)


class _FakeChannel:
    def __init__(self):
        self.StaticInput = _FakeStaticInput()


class _FakeMirror:
    def __init__(self):
        self.Channel_0 = _FakeChannel()
        self.Channel_1 = _FakeChannel()


class _FakeMRE2:
    def __init__(self):
        self.Mirror = _FakeMirror()


def make_mirror(available=True):
    """绕过 optoMDC 连接，直接造一个已连接的振镜对象。"""
    mir = optotunecontrol.optotune.__new__(optotunecontrol.optotune)
    mir.available = available
    mir.mre2 = _FakeMRE2() if available else None
    return mir


def test_setxy_drives_channels_with_calibrated_axes():
    """SETX=2.2 / SETY=-0.7 必须打出旧版 SETX=0.7 / SETY=2.2 对应的通道值。"""
    mir = make_mirror()
    mir.setxy(2.2, -0.7)
    mirror = mir.mre2.Mirror
    assert mirror.Channel_0.StaticInput.value == pytest.approx(deg_to_xy(0.7))
    assert mirror.Channel_1.StaticInput.value == pytest.approx(deg_to_xy(2.2))


def test_getxy_returns_user_coordinates_matching_setxy():
    """读回值换算回用户坐标后应与 setxy 的输入一致（界面再取反显示）。"""
    mir = make_mirror()
    mir.setxy(2.2, -0.7)
    assert mir.getxy() == pytest.approx((2.2, -0.7), abs=0.05)

    mir.setxy(0.7, 2.2)
    assert mir.getxy() == pytest.approx((0.7, 2.2), abs=0.05)


def test_setzero_centers_both_channels():
    mir = make_mirror()
    mir.setxy(2.2, -0.7)
    mir.setzero()
    mirror = mir.mre2.Mirror
    assert mirror.Channel_0.StaticInput.value == pytest.approx(0.0)
    assert mirror.Channel_1.StaticInput.value == pytest.approx(0.0)


def test_unavailable_mirror_is_noop():
    """没装 optoMDC / 没插振镜时所有接口都不能抛异常。"""
    mir = make_mirror(available=False)
    mir.setxy(2.2, -0.7)
    mir.setzero()
    assert mir.getxy() == pytest.approx((0.0, 0.0))
