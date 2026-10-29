# -*- coding: utf-8 -*-
"""主界面组装测试：用 stub 顶掉 cv2 / optoMDC / 摄像头 / 振镜，验证陀螺仪面板已挂载。"""

import sys
import time
import types

import os

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


class _FakeMirror:
    """振镜替身：不需要 optoMDC SDK。"""

    def setxy(self, x, y):
        pass

    def setzero(self):
        pass

    def getxy(self):
        return 0.0, 0.0


@pytest.fixture
def root():
    """Tk 根窗口：同一进程里反复创建/销毁偶尔会失败，重试几次再跳过。"""
    r = None
    for _ in range(3):
        try:
            r = tk.Tk()
            break
        except tk.TclError:
            time.sleep(0.2)
    if r is None:
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

    monkeypatch.setattr(optotunecontrol, "optotune", _FakeMirror)

    from sWATGUI import App

    app = App(root, "sWAT system")
    try:
        assert app.gyro_panel is not None
        assert app.gyro_panel.winfo_exists()
        assert app.gyro_panel.winfo_manager() == "grid"
        # 陀螺仪面板横跨“光学参数/模式操作/振镜”三列，位于它们下方（第 1 行）
        assert app.gyro_panel.grid_info()["row"] == 1
        # 面板初始状态：未连接 + 占位符
        app.gyro_panel.refresh()
        assert app.gyro_panel.var_theta.get() == "--"
        assert app.gyro_panel.var_status.get().endswith("未连接")
    finally:
        app.gyro_service.stop()


class _RampMirror(_FakeMirror):
    """getxy() 固定返回两路不同读数，便于验证界面的取值映射。"""

    def getxy(self):
        return 1.0, 2.0


def test_getx_gety_mapping(root, monkeypatch):
    """GETX 显示 getxy() 的第二路读数，GETY 显示第一路读数取反。

    振镜读数和界面上的“坐标x/坐标y”不是同一套量，这个映射是仪器约定，
    曾被误当成 bug“修正”成 x↔x / y↔y，这里用用例锁住。
    """
    _install_stubs(monkeypatch)
    import optotunecontrol

    monkeypatch.setattr(optotunecontrol, "optotune", _RampMirror)

    from sWATGUI import App

    app = App(root, "sWAT system")
    try:
        app.update_mirror()
        assert app.getx_value.get() == "2.0°"
        assert app.gety_value.get() == "-1.0°"
    finally:
        app.gyro_service.stop()


def test_saveframe_filename_uses_coordinate_names(root, monkeypatch, tmp_path):
    """保存图像的文件名要跟界面一致：坐标x / 坐标y，不再叫 angle / theta。"""
    _install_stubs(monkeypatch)
    import optotunecontrol

    monkeypatch.setattr(optotunecontrol, "optotune", _FakeMirror)

    written = {}
    monkeypatch.chdir(tmp_path)

    from sWATGUI import App
    import sWATGUI

    # 注意：sWATGUI 里 import 的是本模块首次导入时那个 stub，patched 的 sys.modules["cv2"]
    # 可能已经被前一个用例的 monkeypatch 还原，所以直接补在 sWATGUI.cv2 上。
    monkeypatch.setattr(sWATGUI.cv2, "imwrite",
                        lambda path, image: written.setdefault("path", path))

    app = App(root, "sWAT system")
    try:
        # theta=90（坐标y）、angle=3（坐标x）
        app.para.setconfigparameters(2.0, 1.64, 90.0, 3.0, [1.0, 0.5, 10.0])
        app.currentframe = object()  # 占位帧，不需要真实图像
        app.saveframe()
        name = os.path.basename(written["path"])
    finally:
        app.gyro_service.stop()

    assert "_x3.0" in name
    assert "_y90.0" in name
    assert "angle" not in name
    assert "theta" not in name


def test_saveframe_without_frame_is_noop(root, monkeypatch, tmp_path):
    """没有相机（currentframe 为 None）时点“保存图像”只提示，不能抛 AttributeError。"""
    _install_stubs(monkeypatch)
    import optotunecontrol

    monkeypatch.setattr(optotunecontrol, "optotune", _FakeMirror)

    written = {}
    monkeypatch.chdir(tmp_path)

    from sWATGUI import App
    import sWATGUI

    monkeypatch.setattr(sWATGUI.cv2, "imwrite",
                        lambda path, image: written.setdefault("path", path))

    app = App(root, "sWAT system")
    try:
        app.saveframe()
    finally:
        app.gyro_service.stop()

    assert written == {}
    assert not (tmp_path / "res").exists()
