# -*- coding: utf-8 -*-
"""主界面组装测试：用 stub 顶掉 cv2 / optoMDC / 摄像头 / 振镜，验证陀螺仪面板已挂载。"""

import sys
import time
import types

import pytest

tk = pytest.importorskip("tkinter")


def _install_stubs(monkeypatch):
    """安装最小可用替身，避免依赖真实相机与振镜硬件。"""
    cv2 = types.ModuleType("cv2")
    cv2.CAP_ANY = 0
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4

    class _Capture:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args):
            pass

        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    cv2.VideoCapture = _Capture
    cv2.imwrite = lambda *args, **kwargs: True
    # 真实 cv2.waitKey(500) 会阻塞 500ms，这里用等量睡眠模拟，避免测试线程空转
    cv2.waitKey = lambda *args, **kwargs: time.sleep(0.05)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "optoMDC", types.ModuleType("optoMDC"))


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


def test_app_builds_gyro_panel(root, monkeypatch):
    _install_stubs(monkeypatch)
    import optotunecontrol

    class FakeMirror:
        def setxy(self, x, y):
            pass

        def setzero(self):
            pass

        def getxy(self):
            return 0.0, 0.0

    monkeypatch.setattr(optotunecontrol, "optotune", FakeMirror)

    from sWATGUI import App

    app = App(root, "sWAT system")
    try:
        assert app.gyro_panel is not None
        assert app.gyro_panel.winfo_exists()
        assert app.gyro_panel.winfo_manager() == "grid"
        assert app.gyro_panel.grid_info()["row"] == 6
        # 面板初始状态：未连接 + 占位符
        app.gyro_panel.refresh()
        assert app.gyro_panel.var_theta.get() == "--"
        assert app.gyro_panel.var_status.get().endswith("未连接")
    finally:
        app.gyro_service.stop()
