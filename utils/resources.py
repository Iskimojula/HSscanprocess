# -*- coding: utf-8 -*-
"""资源路径解析：同时兼容开发环境与 PyInstaller 打包环境。

开发时资源与代码在同一目录树中；用 PyInstaller 打包后资源会被解包到
sys._MEIPASS 指向的临时目录。所有资源读取都必须经过 resource_path()，
避免"开发时正常、打包成 exe 后找不到文件闪退"。
"""

import os
import sys


def app_base_dir() -> str:
    """返回资源根目录：打包后为解包目录 sys._MEIPASS，否则为仓库根目录。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return meipass
    # 本文件位于 <repo>/utils/resources.py，上溯两级即仓库根目录
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(relative_path: str) -> str:
    """把相对资源路径解析成绝对路径。"""
    return os.path.normpath(os.path.join(app_base_dir(), relative_path))
