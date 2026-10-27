# 陀螺仪转动角度（theta）测量功能 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 sWAT 控制软件的「实验数据」面板中，实时显示 HWT906P 陀螺仪的初始角度（Yaw1, Pitch1, Roll1）、当前角度（Yaw2, Pitch2, Roll2），并用四元数方法计算、显示平面绕轴 L 转动的角度 theta，配套「记录 / 清零」两个按钮。

**Architecture:** 新增三个包：`services/`（纯计算 + 设备 I/O，含后台线程与线程安全队列）、`views/`（tkinter 组件，只做布局与刷新）、`utils/`（资源路径等基础设施）。`sWATGUI.py` 只作为组装根节点，把 `views/gyro_panel.GyroPanel` 放进既有 `fr_configpara` 的空白（截图蓝框）区域。设备读取全部在子线程，UI 通过 `queue.Queue` + `root.after` 拉取快照，主线程不做任何超过 50ms 的工作。

**Tech Stack:** Python 3.9+、tkinter、pyserial、numpy、pytest；HWT906P 官方 Python SDK（WitStandardProtocol，串口 115200）。

**Spec:** 用户需求原文（2026-09-22）：
1. 蓝框处显示陀螺仪三方向角度 `(Yaw1, Pitch1, Roll1)`；
2. 平面绕轴 L 转动 theta 后角度为 `(Yaw2, Pitch2, Roll2)`；
3. 用**四元数**方法根据这两组数据算出 theta 并显示；
4. 增加「记录」（锁定当前角度为初始角度）与「清零」（清除）按钮；
5. 在外推法测试程序所在 git 仓库新建分支开发，之后合并/上传 GitHub 交开发者审核。

## Global Constraints

- 任何耗时超过 50ms 的操作（串口打开/关闭、端口枚举、寄存器读取、文件 I/O）必须放到子线程（`threading` / `QThread`），并通过线程安全队列或信号回调更新 UI。
- UI 与核心业务分层：`views/` 只负责布局与组件，`services/` 负责实际计算与设备访问；禁止把上百行设备/网络逻辑塞进按钮的 `on_click`。
- 资源路径统一走动态解析函数（兼容 `sys._MEIPASS` / PyInstaller），禁止裸相对路径读资源。
- 不修改既有测量算法（`imageproc.py` / `optometry.py` / `capprocess.py`）的行为；不改动 `visual` 分支既有 UI 的既有控件语义。
- 保持既有平铺模块的导入方式（`import configpara`、`python main.py` 从仓库根目录启动）。
- 所有新增代码使用 UTF-8；注释与界面文案使用简体中文，标识符使用英文。

## Review Focus

1. 陀螺仪未连接 / 串口被占用 / 无 quaternion 数据包时，面板必须给出可见的文字状态，且主界面不卡死、不抛异常。
2. 「记录」在还没有任何一帧数据时被按下：必须给出可见提示（数据不足），且不能把 theta 显示成 0 让人误判为已测量。
3. theta 在 ±180° 附近跨越（例如从 179° 转到 -179°）必须连续（约 2°），不能跳变到 358°。
4. 数值刷新时不得让整行文字宽度变化导致面板抖动（数字右对齐、定长格式）。
5. 关闭主窗口时串口必须释放，不能残留线程/句柄导致下一次连接失败。

---

## File Structure

| 文件 | 责任 |
| --- | --- |
| `utils/__init__.py` | 包标记 |
| `utils/resources.py` | `resource_path()`：开发/PyInstaller 双环境资源定位 |
| `services/__init__.py` | 包标记 |
| `services/rotation_math.py` | 纯数学：四元数乘法/共轭/归一化、欧拉↔四元数、相对转动角 theta 与转轴 L（无 IO、无第三方依赖） |
| `services/gyro_service.py` | 陀螺仪服务：`GyroSample`/`GyroSnapshot` 数据结构、`SensorAdapter` 协议、`HWT906PAdapter`（懒加载 SDK）、`GyroService`（后台线程 + 线程安全队列 + 记录/清零） |
| `views/__init__.py` | 包标记 |
| `views/gyro_panel.py` | `GyroPanel`：实验数据面板内的陀螺仪子面板（标签、状态、端口、记录/清零按钮、50ms 轮询刷新） |
| `hwt906p/` | 官方 SDK（从 HWT906P 工具工程拷入，独立第三方代码，不修改） |
| `tests/` | pytest 用例：数学、服务、视图、资源路径 |
| `sWATGUI.py` | 仅改动：资源路径、组装 `GyroPanel`、窗口尺寸/关闭事件（Modify） |
| `.gitignore` | 忽略 `__pycache__/`、`*.pyc`、`res/` 运行产物（Create） |

---

### Task 1: 资源路径工具（打包兼容）

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/resources.py`
- Test: `tests/test_resources.py`

**Interfaces:**
- Produces: `utils.resources.resource_path(relative_path: str) -> str`（返回绝对路径；冻结环境用 `sys._MEIPASS`，否则用仓库根目录）
- Produces: `utils.resources.app_base_dir() -> str`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_resources.py
import os, sys
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_resources.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'utils'`

- [ ] **Step 3: 最小实现**

```python
# utils/resources.py
"""资源路径解析：同时兼容开发环境与 PyInstaller 打包环境。"""
import os
import sys

def app_base_dir() -> str:
    """返回资源根目录：打包后为解包目录 sys._MEIPASS，否则为仓库根目录。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return meipass
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resource_path(relative_path: str) -> str:
    """把相对资源路径解析成绝对路径。"""
    return os.path.normpath(os.path.join(app_base_dir(), relative_path))
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_resources.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add utils/ tests/test_resources.py
git commit -m "feat(utils): 新增兼容 PyInstaller 的资源路径解析"
```

---

### Task 2: 四元数转动角计算（核心数学）

**Files:**
- Create: `services/__init__.py`
- Create: `services/rotation_math.py`
- Test: `tests/test_rotation_math.py`

**Interfaces:**
- Produces: `normalize(q) -> list[float]`
- Produces: `multiply(q1, q2) -> list[float]`
- Produces: `conjugate(q) -> list[float]`（单位四元数的逆）
- Produces: `euler_to_quaternion(yaw_deg, pitch_deg, roll_deg) -> list[float]`（ZYX 顺序，`[w,x,y,z]`）
- Produces: `quaternion_to_euler(q) -> tuple[float,float,float]`（返回 `(yaw, pitch, roll)`，单位：度）
- Produces: `axis_angle_to_quaternion(axis, angle_deg) -> list[float]`
- Produces: `relative_rotation(q_ref, q_cur) -> RotationResult(theta_deg, axis, quat_delta)`
- Consumes: 无（纯标准库 `math`）

**数学约定（写进 docstring，开发者审核重点）：** `q_cur`、`q_ref` 都是「机体→世界」的姿态四元数；相对姿态 `q_delta = q_cur ⊗ q_ref⁻¹`；转动角 `theta = 2·atan2(‖v‖, |w|)`∈[0°,180°]；转轴 `L = v/‖v‖`（`v = (x,y,z)`），当 `‖v‖ < 1e-9` 视为无转动；用 `atan2` 而非 `acos`，在 theta≈0 时数值更稳；取 `|w|` 消除四元数双重覆盖（q 与 −q 表示同一姿态）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rotation_math.py
import math
import services.rotation_math as rm

def test_identity_gives_zero():
    r = rm.relative_rotation([1,0,0,0], [1,0,0,0])
    assert r.theta_deg == 0.0
    assert r.axis == (0.0, 0.0, 0.0)

def test_ninety_degrees_about_z():
    q2 = rm.axis_angle_to_quaternion((0,0,1), 90.0)
    r = rm.relative_rotation([1,0,0,0], q2)
    assert abs(r.theta_deg - 90.0) < 1e-6
    assert abs(r.axis[2] - 1.0) < 1e-6

def test_one_eighty_about_x():
    q2 = rm.axis_angle_to_quaternion((1,0,0), 180.0)
    r = rm.relative_rotation([1,0,0,0], q2)
    assert abs(r.theta_deg - 180.0) < 1e-6

def test_arbitrary_axis_60_degrees():
    axis = (1/math.sqrt(3),)*3
    q2 = rm.axis_angle_to_quaternion(axis, 60.0)
    r = rm.relative_rotation([1,0,0,0], q2)
    assert abs(r.theta_deg - 60.0) < 1e-6
    for a, b in zip(r.axis, axis):
        assert abs(a - b) < 1e-6

def test_double_cover_same_theta():
    q2 = rm.axis_angle_to_quaternion((0,0,1), 120.0)
    neg = [-x for x in q2]
    r1 = rm.relative_rotation([1,0,0,0], q2)
    r2 = rm.relative_rotation([1,0,0,0], neg)
    assert abs(r1.theta_deg - r2.theta_deg) < 1e-6

def test_wraparound_is_continuous():
    q179 = rm.axis_angle_to_quaternion((0,0,1), 179.0)
    qn179 = rm.axis_angle_to_quaternion((0,0,1), -179.0)
    r = rm.relative_rotation(q179, qn179)
    assert abs(r.theta_deg - 2.0) < 1e-6

def test_euler_quat_roundtrip():
    for yaw, pitch, roll in [(0,0,0), (30,10,-20), (-120, 45, 90)]:
        q = rm.euler_to_quaternion(yaw, pitch, roll)
        ry, rp, rr = rm.quaternion_to_euler(q)
        assert abs(ry - yaw) < 1e-6
        assert abs(rp - pitch) < 1e-6
        assert abs(rr - roll) < 1e-6

def test_unnormalized_quaternions_are_normalized():
    r = rm.relative_rotation([2,0,0,0], rm.axis_angle_to_quaternion((0,1,0), 45.0))
    assert abs(r.theta_deg - 45.0) < 1e-6
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_rotation_math.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'services'`

- [ ] **Step 3: 最小实现**

```python
# services/rotation_math.py
"""四元数姿态运算（纯标准库实现）。

约定：
  * 四元数一律写作 [w, x, y, z]，表示「机体坐标系 -> 世界坐标系」的旋转；
  * 欧拉角顺序为 ZYX（先 yaw 绕 Z，再 pitch 绕 Y，最后 roll 绕 X），与 HWT906P
    输出 (angle_x=Roll, angle_y=Pitch, angle_z=Yaw) 对应；
  * 相对转动：q_delta = q_cur ⊗ q_ref⁻¹，theta = 2·atan2(‖v‖, |w|) ∈ [0, 180]。
"""
import math
from typing import NamedTuple, Sequence, Tuple

EPS = 1e-9

class RotationResult(NamedTuple):
    theta_deg: float            # 相对转动的总角度，单位：度，范围 [0, 180]
    axis: Tuple[float, float, float]   # 转轴 L 的单位向量；无转动时为 (0,0,0)
    quat_delta: Tuple[float, float, float, float]  # 归一化后的相对四元数 [w,x,y,z]

def normalize(q: Sequence[float]) -> list:
    n = math.sqrt(sum(float(x) * float(x) for x in q))
    if n < EPS:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(x) / n for x in q]

def multiply(q1: Sequence[float], q2: Sequence[float]) -> list:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]

def conjugate(q: Sequence[float]) -> list:
    return [q[0], -q[1], -q[2], -q[3]]

def axis_angle_to_quaternion(axis: Sequence[float], angle_deg: float) -> list:
    n = math.sqrt(sum(float(a) * float(a) for a in axis))
    if n < EPS:
        return [1.0, 0.0, 0.0, 0.0]
    ax, ay, az = (float(a) / n for a in axis)
    half = math.radians(angle_deg) * 0.5
    s = math.sin(half)
    return [math.cos(half), ax * s, ay * s, az * s]

def euler_to_quaternion(yaw_deg: float, pitch_deg: float, roll_deg: float) -> list:
    cy, sy = math.cos(math.radians(yaw_deg) * 0.5), math.sin(math.radians(yaw_deg) * 0.5)
    cp, sp = math.cos(math.radians(pitch_deg) * 0.5), math.sin(math.radians(pitch_deg) * 0.5)
    cr, sr = math.cos(math.radians(roll_deg) * 0.5), math.sin(math.radians(roll_deg) * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]

def quaternion_to_euler(q: Sequence[float]) -> Tuple[float, float, float]:
    w, x, y, z = normalize(q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

def relative_rotation(q_ref: Sequence[float], q_cur: Sequence[float]) -> RotationResult:
    qd = normalize(multiply(normalize(q_cur), conjugate(normalize(q_ref))))
    w, x, y, z = qd
    vnorm = math.sqrt(x * x + y * y + z * z)
    theta = math.degrees(2.0 * math.atan2(vnorm, abs(w)))
    if vnorm < EPS:
        return RotationResult(0.0, (0.0, 0.0, 0.0), tuple(qd))
    return RotationResult(theta, (x / vnorm, y / vnorm, z / vnorm), tuple(qd))
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_rotation_math.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: Commit**

```bash
git add services/__init__.py services/rotation_math.py tests/test_rotation_math.py
git commit -m "feat(services): 四元数相对转动角 theta 计算"
```

---

### Task 3: 陀螺仪服务（后台线程 + 线程安全队列）

**Files:**
- Create: `services/gyro_service.py`
- Create: `tests/fakes.py`
- Test: `tests/test_gyro_service.py`

**Interfaces:**
- Produces: `GyroSample(timestamp, yaw, pitch, roll, quat, has_quat)`
- Produces: `AngleTriple(yaw, pitch, roll)`
- Produces: `GyroSnapshot(connected, connecting, port, status_text, error, current, reference, theta_deg, axis, rate_hz, temperature, fresh)`
- Produces: `SensorAdapter` 协议类（`open/close/read/set_callback`）
- Produces: `GyroService(adapter=None)`
  - `start(port, baud=115200) -> None`（非阻塞，立即返回）
  - `stop() -> None`（非阻塞，后台线程关闭串口）
  - `record_reference() -> bool`（O(1)，不阻塞）
  - `clear_reference() -> None`
  - `get_snapshot() -> GyroSnapshot`（线程安全）
  - `drain_ui_events() -> bool`（是否有新数据待刷新）
  - `available_ports() -> list[str]`（静态方法，内部异常兜底）
- Consumes: `services.rotation_math`

- [ ] **Step 1: 写失败测试（先写 `tests/fakes.py`）**

```python
# tests/fakes.py
import time
from services.gyro_service import GyroSample

class FakeAdapter:
    """可编程的假传感器，用于不接硬件也能跑服务层。"""
    def __init__(self, fail_open=False):
        self.fail_open = fail_open
        self.opened = False
        self.closed = False
        self.callback = None
        self.sample = None
        self.open_delay = 0.0

    def open(self, port, baud):
        time.sleep(self.open_delay)
        if self.fail_open:
            raise ConnectionError("port busy")
        self.opened = True

    def close(self):
        self.closed = True

    def set_callback(self, fn):
        self.callback = fn

    def read(self):
        return self.sample

    def feed(self, yaw, pitch, roll, quat=None):
        self.sample = GyroSample(time.time(), yaw, pitch, roll,
                                 quat or (1.0, 0.0, 0.0, 0.0), quat is not None)
        if self.callback:
            self.callback(self.sample)
```

```python
# tests/test_gyro_service.py
import time
import services.rotation_math as rm
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
    ad = FakeAdapter(open_delay=0.3)
    svc = GyroService(adapter=ad)
    t0 = time.time()
    svc.start("COM_TEST")
    elapsed = time.time() - t0
    assert elapsed < 0.05, f"start() 阻塞了 {elapsed:.3f}s"
    assert wait_for(lambda: svc.get_snapshot().connected)
    svc.stop()

def test_theta_none_before_record():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST")
    ad.feed(10, -5, 3)
    assert wait_for(lambda: svc.get_snapshot().current is not None)
    snap = svc.get_snapshot()
    assert snap.reference is None
    assert snap.theta_deg is None
    svc.stop()

def test_record_then_rotate_gives_theta():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST")
    ad.feed(10, -5, 3)
    assert wait_for(lambda: svc.get_snapshot().current is not None)
    assert svc.record_reference() is True
    ref = svc.get_snapshot().reference
    assert (round(ref.yaw), round(ref.pitch), round(ref.roll)) == (10, -5, 3)
    q = rm.euler_to_quaternion(10, -5, 3)
    q_turned = rm.multiply(rm.axis_angle_to_quaternion((0, 0, 1), 30), q)
    ad.feed(40, -5, 3, quat=q_turned)
    assert wait_for(lambda: abs((svc.get_snapshot().theta_deg or 0) - 30.0) < 0.5)
    svc.stop()

def test_record_without_data_returns_false():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST")
    assert wait_for(lambda: svc.get_snapshot().connected)
    assert svc.record_reference() is False
    assert svc.get_snapshot().error
    svc.stop()

def test_clear_reference_resets_theta():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST")
    ad.feed(0, 0, 0)
    assert wait_for(lambda: svc.record_reference() is True)
    svc.clear_reference()
    snap = svc.get_snapshot()
    assert snap.reference is None and snap.theta_deg is None
    svc.stop()

def test_open_failure_is_reported_not_raised():
    ad = FakeAdapter(fail_open=True)
    svc = GyroService(adapter=ad)
    svc.start("COM_BAD")
    assert wait_for(lambda: svc.get_snapshot().error)
    snap = svc.get_snapshot()
    assert snap.connected is False and "COM_BAD" in snap.error
    svc.stop()

def test_stop_closes_adapter_in_background():
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST")
    assert wait_for(lambda: svc.get_snapshot().connected)
    t0 = time.time()
    svc.stop()
    assert time.time() - t0 < 0.05
    assert wait_for(lambda: ad.closed)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_gyro_service.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'services.gyro_service'`

- [ ] **Step 3: 实现服务层**

要点（完整代码在实现步骤落地）：

```python
# services/gyro_service.py（结构）
@dataclass(frozen=True)
class GyroSample:
    timestamp: float; yaw: float; pitch: float; roll: float
    quat: tuple; has_quat: bool
    def as_quaternion(self):  # 有 quat 用 quat，否则用欧拉角兜底
        return self.quat if self.has_quat else rm.euler_to_quaternion(self.yaw, self.pitch, self.roll)

class SensorAdapter:  # open/close/read/set_callback
class HWT906PAdapter(SensorAdapter):  # Task 4
class GyroService:
    def start(self, port, baud=115200):   # 只起线程，立即返回
    def _connect_worker(self, port, baud): pass
    def _on_sample(self, sample):          # 运行在 SDK 读线程：只做 O(1) 计算 + 入队
    def record_reference(self) -> bool: pass
    def clear_reference(self) -> None: pass
    def get_snapshot(self) -> GyroSnapshot: pass
    def drain_ui_events(self) -> bool: pass
    @staticmethod
    def available_ports() -> list: pass
```

细节约束：
1. `_events` 为 `queue.Queue(maxsize=1)`，入队用 `put_nowait`，`Full` 时 `get_nowait` 丢弃旧值（永远只保留最新，防积压）。
2. `theta_deg` 在 `reference is None` 时为 `None`（UI 显示 `--`），不用 0 冒充。
3. `fresh` = 最近一帧距今 < 1.0s；超时显示「数据超时」。
4. `has_quat=False` 时用 `rotation_math.euler_to_quaternion(yaw, pitch, roll)` 兜底。
5. `record_reference()` 无数据时写入 `error="暂无陀螺仪数据，无法记录"` 并返回 `False`。
6. `stop()` 起后台线程调用 `adapter.close()`，主线程立即返回。
7. `available_ports()` 用 `serial.tools.list_ports`，`try/except` 兜底返回 `[]`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_gyro_service.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: Commit**

```bash
git add services/gyro_service.py tests/fakes.py tests/test_gyro_service.py
git commit -m "feat(services): 陀螺仪后台采集服务与记录/清零状态机"
```

---

### Task 4: 接入 HWT906P 官方 SDK

**Files:**
- Create: `hwt906p/__init__.py`、`hwt906p/imu.py`、`hwt906p/protocol.py`、`hwt906p/NOTICE.md`（从 HWT906P 工具工程拷入，不改内容）
- Test: `tests/test_hwt906p_adapter.py`

**Interfaces:**
- Consumes: `services.gyro_service.HWT906PAdapter`
- Produces: 可在无硬件环境下验证的适配层测试（用假 SDK 对象替换真 SDK）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hwt906p_adapter.py
from services.gyro_service import HWT906PAdapter, GyroSample

class FakeSdkData:
    def __init__(self, **kw):
        self.timestamp = kw.get("timestamp", 1.0)
        self.angle_x = kw.get("roll", 1.0)
        self.angle_y = kw.get("pitch", 2.0)
        self.angle_z = kw.get("yaw", 3.0)
        self.quat_w, self.quat_x, self.quat_y, self.quat_z = kw.get("quat", (1.0, 0.0, 0.0, 0.0))
        self.temperature = 25.0

class FakeSdkDevice:
    def __init__(self):
        self.opened = None
        self.cbs = []
        self.closed = False
    def open(self, port, baud): self.opened = (port, baud); return True
    def close(self): self.closed = True
    def on_data_update(self, cb): self.cbs.append(cb)
    def get_all_data(self): return FakeSdkData()

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
    ad.close()
    assert dev.closed

def test_adapter_reports_missing_sdk():
    def boom():
        raise ImportError("no hwt906p")
    ad = HWT906PAdapter(device_factory=boom)
    try:
        ad.open("COM1", 115200)
    except ConnectionError as e:
        assert "hwt906p" in str(e)
    else:
        raise AssertionError("应当抛出可读错误")
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_hwt906p_adapter.py -v`
Expected: FAIL `ImportError: cannot import name 'HWT906PAdapter'`

- [ ] **Step 3: 拷贝 SDK 并实现适配层**

```powershell
# 从本机已验证可用的 HWT906P 工具工程拷入（只拷包，不拷测试工具）
Copy-Item "D:\codex任务文件夹\HWT906P\hwt906p\*" "hwt906p\" -Recurse -Force
```

```python
# services/gyro_service.py 追加
def _default_sdk_factory():
    from hwt906p import HWT906P          # 懒加载：没装 SDK 也不影响程序启动
    return HWT906P()

class HWT906PAdapter(SensorAdapter):
    def __init__(self, device_factory=None):
        self._factory = device_factory or _default_sdk_factory
        self._device = None
        self._callback = None

    def open(self, port, baud):
        try:
            self._device = self._factory()
            self._device.open(port, baud)
        except ImportError as e:
            raise ConnectionError(f"未找到 hwt906p SDK（{e}）") from e
        except Exception as e:
            raise ConnectionError(str(e)) from e
        if self._callback:
            self._device.on_data_update(self._wrap(self._callback))

    def set_callback(self, fn):
        self._callback = fn
        if self._device is not None:
            self._device.on_data_update(self._wrap(fn))

    def _wrap(self, fn):
        def handler(data):
            sample = _sdk_data_to_sample(data)
            if sample is not None:
                fn(sample)
        return handler

    def close(self):
        if self._device is not None:
            try:
                self._device.close()
            finally:
                self._device = None

    def read(self):
        d = self._device.get_all_data() if self._device else None
        return _sdk_data_to_sample(d)
```

SDK 字段映射：`IMUData.angle_z → yaw`、`angle_y → pitch`、`angle_x → roll`、`(quat_w, quat_x, quat_y, quat_z) → quat`；`has_quat` 在四元数不是 `(1,0,0,0)` 时为 `True`。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_hwt906p_adapter.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: Commit**

```bash
git add hwt906p/ services/gyro_service.py tests/test_hwt906p_adapter.py
git commit -m "feat(hwt906p): 内置 HWT906P Python SDK 与适配层"
```

---

### Task 5: 陀螺仪面板视图（截图蓝框区域）

**Files:**
- Create: `views/__init__.py`、`views/gyro_panel.py`
- Test: `tests/test_gyro_panel.py`

**Interfaces:**
- Consumes: `GyroService`（`start/stop/record_reference/clear_reference/get_snapshot/available_ports`）
- Produces: `GyroPanel(master, service, title="陀螺仪 HWT906P", poll_ms=50)`
  - `format_triple(yaw, pitch, roll) -> str`（静态方法，定长格式）
  - `refresh()`（从 service 拉快照写入 StringVar，测试可直接调用）
  - `on_record() -> bool`、`on_clear() -> None`、`on_connect_toggle() -> None`

界面规约（来自 UI/UX Pro Max 检索结果，逐条落地）：

| 规则（检索来源） | 落地方式 |
| --- | --- |
| `loading-states` / `loading-buttons` | 点「连接」后按钮禁用并显示「连接中…」，直到线程回报 |
| `submit-feedback` / `success-feedback` | 记录/清零后状态行短暂显示「已记录初始角度」/「已清零」 |
| `error-feedback` / `error-placement` | 连接失败在面板状态行显示 `连接 COMx 失败：原因`，红色文字 + 文字说明（不只靠颜色） |
| `color-not-only` | 状态用「● 已连接 / 未连接」文字 + 颜色双通道 |
| `content-jumping` | 数值用 `{:.2f}` 定长格式 + 等宽字体，刷新不抖动 |
| `visual-hierarchy` | theta 单独一行、更大更粗，比角度行醒目 |
| `input-labels` | 端口下拉框前置可见标签「端口」 |
| `no-blocking-animation` / `main-thread-budget` | 端口枚举在子线程做，结果经 `after` 回主线程填 Combobox |

- [ ] **Step 1: 写失败测试**

```python
# tests/test_gyro_panel.py
import time
import tkinter as tk
import pytest
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
    r.destroy()

def test_format_triple_is_fixed_width():
    assert GyroPanel.format_triple(10.234, -5.4, 3.0) == " 10.23 /  -5.40 /   3.00"

def test_panel_shows_dashes_before_data(root):
    svc = GyroService(adapter=FakeAdapter())
    panel = GyroPanel(root, svc)
    panel.refresh()
    assert panel.var_theta.get() == "--"
    assert panel.var_reference.get().count("--") == 3
    svc.stop()

def test_panel_updates_after_record(root):
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST")
    ad.feed(10.0, -5.0, 3.0)
    t0 = time.time()
    while svc.get_snapshot().current is None and time.time() - t0 < 3:
        time.sleep(0.01)
    panel = GyroPanel(root, svc)
    assert panel.on_record() is True
    panel.refresh()
    assert "10.00" in panel.var_reference.get()
    assert panel.var_theta.get() != "--"
    assert "已记录" in panel.var_status.get()
    panel.on_clear()
    panel.refresh()
    assert panel.var_theta.get() == "--"
    svc.stop()

def test_record_and_clear_are_fast(root):
    ad = FakeAdapter()
    svc = GyroService(adapter=ad)
    svc.start("COM_TEST")
    ad.feed(1.0, 2.0, 3.0)
    t0 = time.time()
    while svc.get_snapshot().current is None and time.time() - t0 < 3:
        time.sleep(0.01)
    panel = GyroPanel(root, svc)
    t0 = time.time()
    panel.on_record()
    panel.on_clear()
    assert time.time() - t0 < 0.05
    svc.stop()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_gyro_panel.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'views'`

- [ ] **Step 3: 实现面板**

```python
# views/gyro_panel.py（结构）
class GyroPanel(tk.LabelFrame):
    """实验数据面板中「陀螺仪」子面板：只做布局与刷新，不含设备逻辑。"""
    def __init__(self, master, service, title="陀螺仪 HWT906P", poll_ms=50):
        super().__init__(master, text=title, relief="solid", bd=1)
        self.service = service
        self.poll_ms = poll_ms
        self._flash_until = 0.0      # 临时成功提示的过期时间
        self._build()
        self.refresh()
        self.after(poll_ms, self._poll)
    # 行 0：端口 [Combobox] [连接/断开]        状态：● 未连接
    # 行 1：初始 Y1/P1/R1:  --.-- / --.-- / --.--    [记录]
    # 行 2：当前 Y2/P2/R2:  --.-- / --.-- / --.--    [清零]
    # 行 3：转动角 theta = --.-- °   转轴 L = (--, --, --)
```

行为细节：
1. `_poll()`：`service.drain_ui_events()` 为真或每 20 个周期兜底调用 `refresh()`，再 `after(poll_ms, self._poll)`。
2. `on_record()` 调 `service.record_reference()`；`False` 时状态显示「暂无陀螺仪数据，无法记录」。
3. `on_clear()` 调 `service.clear_reference()`，状态「已清零」。
4. `on_connect_toggle()`：按钮置「连接中…」+ `state="disabled"`，`service.start(port)` 立即返回，状态由轮询更新；已连接时按钮变「断开」。
5. 端口枚举：`threading.Thread(target=..., daemon=True)` 取 `GyroService.available_ports()`，用 `self.after(0, self._apply_ports, ports)` 回主线程填 Combobox。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_gyro_panel.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: Commit**

```bash
git add views/ tests/test_gyro_panel.py
git commit -m "feat(views): 实验数据面板新增陀螺仪角度/theta 子面板"
```

---

### Task 6: 组装进 sWATGUI 并修资源路径

**Files:**
- Modify: `sWATGUI.py`（导入、`__init__`、`update_frame` 的图片路径、面板挂载、窗口尺寸、关闭事件）
- Test: `tests/test_app_integration.py`

**Interfaces:**
- Consumes: `views.gyro_panel.GyroPanel`、`services.gyro_service.GyroService`、`utils.resources.resource_path`
- Produces: `App` 新增属性 `self.gyro_service`、`self.gyro_panel`，新增方法 `on_close()`

- [ ] **Step 1: 写失败测试（用 stub 顶掉 cv2 / optoMDC / 摄像头）**

```python
# tests/test_app_integration.py
import sys, types, tkinter as tk
import pytest

def _install_stubs(monkeypatch):
    for name in ("cv2",):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    cv2 = sys.modules["cv2"]
    cv2.CAP_ANY = 0
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    class _Cap:
        def __init__(self, *a, **k): pass
        def set(self, *a): pass
        def isOpened(self): return True
        def read(self): return False, None
    cv2.VideoCapture = _Cap
    cv2.imwrite = lambda *a, **k: True

def test_app_builds_gyro_panel(monkeypatch):
    _install_stubs(monkeypatch)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("无图形环境")
    root.withdraw()
    from sWATGUI import App
    app = App(root, "sWAT system")
    assert app.gyro_panel is not None
    assert app.gyro_panel.winfo_exists()
    app.gyro_service.stop()
    root.destroy()
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_app_integration.py -v`
Expected: FAIL `AttributeError: 'App' object has no attribute 'gyro_panel'`

- [ ] **Step 3: 修改 `sWATGUI.py`**

```python
# 顶部：新增导入
from services.gyro_service import GyroService
from views.gyro_panel import GyroPanel
from utils.resources import resource_path

# __init__：窗口尺寸与关闭事件
self.root.geometry("520x920")
self.root.resizable(False, False)
self.root.protocol("WM_DELETE_WINDOW", self.on_close)

# __init__：创建服务与面板（放在 self.mirrorGUI() 之后）
self.gyro_service = GyroService()
self.gyro_panel = GyroPanel(self.fr_configpara, self.gyro_service)
self.gyro_panel.grid(row=6, column=0, columnspan=9, sticky="we", padx=6, pady=(6, 8))

# update_frame：修掉裸相对路径（打包后会崩）
img = Image.open(resource_path("no frame.png")).convert("L")

# 新增方法：关闭窗口时后台释放串口，主线程不被 2s 的 close() 卡住
def on_close(self):
    self.gyro_service.stop()
    self.root.destroy()
```

- [ ] **Step 4: 运行确认通过 + 全量回归**

Run: `python -m pytest tests -v`
Expected: PASS（全部用例）

- [ ] **Step 5: Commit**

```bash
git add sWATGUI.py tests/test_app_integration.py
git commit -m "feat(ui): 实验数据面板接入陀螺仪 theta 显示与记录/清零按钮"
```

---

### Task 7: 文档、忽略规则与交付

**Files:**
- Create: `.gitignore`
- Create: `docs/陀螺仪角度功能说明.md`

- [ ] **Step 1: 写使用与算法说明**（依赖安装、端口选择、记录/清零语义、theta 公式、无硬件时的自检命令）
- [ ] **Step 2: 写 `.gitignore`**（`__pycache__/`、`*.pyc`、`res/`、`.pytest_cache/`）
- [ ] **Step 3: 全量测试 + 语法编译检查**

Run: `python -m pytest tests -v` 与 `python -m compileall services views utils sWATGUI.py`

- [ ] **Step 4: Commit 并推送分支**

```bash
git add .gitignore docs/
git commit -m "docs: 陀螺仪转动角功能说明与仓库忽略规则"
git push -u origin feature/gyro-rotation-angle
```

---

## Self-Review

**Spec coverage:**

| 需求 | 覆盖任务 |
| --- | --- |
| 蓝框处显示 (Yaw1,Pitch1,Roll1) | Task 5（行 1 + 记录锁定） |
| 显示 (Yaw2,Pitch2,Roll2) | Task 5（行 2 实时） |
| 四元数法计算 theta 并显示 | Task 2（算法）+ Task 3（接入）+ Task 5（显示） |
| 记录 / 清零按钮 | Task 3（语义）+ Task 5（按钮） |
| 分支开发 → 合并/上传 GitHub | 分支 `feature/gyro-rotation-angle`，Task 7 推送；合并由开发者审核后执行 |
| 耗时任务子线程 + 队列更新 UI | Task 3（服务线程）+ Task 5（`after` 轮询） |
| views/services 分层 | Task 2/3/4 服务层、Task 5 视图层、Task 6 组装 |
| 资源路径兼容打包 | Task 1 + Task 6 |

**Placeholder scan:** 无 TBD/TODO；每个任务都带可执行代码与命令。

**Type consistency:** `GyroSample.quat` 全链路为 `(w,x,y,z)`；`theta_deg` 在无基准时为 `None`，UI 统一用 `--`；`SensorAdapter` 的 `open/close/read/set_callback` 在 Task 3/4 一致。

**Review Focus → 测试归属:** 前述 5 条风险分别由 `test_open_failure_is_reported_not_raised`、`test_record_without_data_returns_false`、`test_wraparound_is_continuous`、`test_format_triple_is_fixed_width`、`test_stop_closes_adapter_in_background` 覆盖。
