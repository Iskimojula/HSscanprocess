# -*- coding: utf-8 -*-
"""configpara 的输出字段命名：存档文本要和界面一致（坐标X / 坐标Y）。"""

import configpara


def test_makestrforsave_uses_coordinate_names():
    para = configpara.configparameters()
    para.setconfigparameters(2.0, 1.64, 90.0, 3.0, [1.0, 0.5, 10.0])

    line = para.makestrforsave()

    assert "坐标X：3.0" in line
    assert "坐标Y：90.0" in line
    # 旧版把 X 和 Y 都写成 angle，这里必须分开
    assert "偏转X" not in line
    assert "偏转Y" not in line
