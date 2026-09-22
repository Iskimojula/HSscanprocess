# -*- coding: utf-8 -*-
"""pytest 根配置：使 tests/ 能 import 到仓库根目录下的 services/views/utils。"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
