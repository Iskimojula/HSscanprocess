# -*- coding: utf-8 -*-
import os
import sys

import utils.resources as res


def test_resource_path_uses_repo_root_when_not_frozen():
    p = res.resource_path("no frame.png")
    assert os.path.isabs(p)
    assert p.endswith("no frame.png")
    assert os.path.exists(p)


def test_resource_path_prefers_meipass(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    p = res.resource_path("no frame.png")
    assert p == os.path.join(str(tmp_path), "no frame.png")


def test_app_base_dir_is_repo_root_in_development():
    assert os.path.isfile(os.path.join(res.app_base_dir(), "sWATGUI.py"))
