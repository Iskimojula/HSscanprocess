# -*- coding: utf-8 -*-
"""实验数据面板中的"陀螺仪 HWT906P"子面板。

职责边界：
    * 只做布局与刷新，所有设备访问、theta 计算都在 services/gyro_service.py 中；
    * 按钮回调里不做任何耗时操作（start/stop/record/clear 都是立即返回的）；
    * 刷新走 after 轮询 + 服务端线程安全快照，绝不跨线程操作 tkinter。
"""

import threading
import time
import tkinter as tk
from tkinter import ttk

FONT_VALUE = ("Consolas", 9)
FONT_THETA = ("Arial", 12, "bold")
FONT_SMALL = ("Arial", 8)

COLOR_OK = "#137333"
COLOR_WARN = "#B06000"
COLOR_OFF = "#5F6368"
COLOR_ERROR = "#B3261E"
COLOR_HINT = "#1A4B8C"

PLACEHOLDER_TRIPLE = "-- / -- / --"


class GyroPanel(tk.LabelFrame):
    """陀螺仪角度 / 转动角 theta 面板。"""

    def __init__(self, master, service, title="陀螺仪 HWT906P", poll_ms=50):
        super().__init__(master, text=title, relief="solid", bd=1)
        self.service = service
        self.poll_ms = poll_ms

        self._flash_text = ""
        self._flash_until = 0.0
        self._ticks = 0

        self._build()
        self.refresh()
        self.after(poll_ms, self._poll)
        self._load_ports_async()

    # ── 布局 ───────────────────────────────────────────────────────

    def _build(self):
        self.var_port = tk.StringVar(value="COM3")
        self.var_status = tk.StringVar(value="● 未连接")
        self.var_rate = tk.StringVar(value="")
        self.var_reference = tk.StringVar(value=PLACEHOLDER_TRIPLE)
        self.var_current = tk.StringVar(value=PLACEHOLDER_TRIPLE)
        self.var_theta = tk.StringVar(value="--")
        self.var_axis = tk.StringVar(value="(--, --, --)")
        self.var_message = tk.StringVar(value="")

        # 第 0 行：端口选择 + 连接按钮 + 状态
        tk.Label(self, text="端口").grid(row=0, column=0, sticky="w", padx=(4, 2))
        self.cmb_port = ttk.Combobox(
            self, textvariable=self.var_port, width=7, values=[], state="normal"
        )
        self.cmb_port.grid(row=0, column=1, sticky="w")
        self.btn_connect = tk.Button(
            self, text="连接", width=7, command=self.on_connect_toggle
        )
        self.btn_connect.grid(row=0, column=2, sticky="w", padx=4)
        #固定宽度：状态/速率/数值长度变化时不改变面板宽度（避免窗口抖动或被裁切）
        self.lbl_status = tk.Label(self, textvariable=self.var_status, anchor="w", width=24)
        self.lbl_status.grid(row=0, column=3, columnspan=2, sticky="w")
        tk.Label(self, textvariable=self.var_rate, anchor="e", fg=COLOR_OFF, width=8).grid(
            row=0, column=5, sticky="e"
        )

        # 第 1 行：初始角度（按下"记录"时锁定）
        tk.Label(self, text="初始 Yaw1/Pitch1/Roll1").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=(4, 2)
        )
        tk.Label(
            self, textvariable=self.var_reference, font=FONT_VALUE, anchor="w", width=24
        ).grid(
            row=1, column=2, columnspan=2, sticky="w"
        )
        self.btn_record = tk.Button(self, text="记录", width=7, command=self.on_record)
        self.btn_record.grid(row=1, column=4, sticky="e", padx=4)

        # 第 2 行：当前角度
        tk.Label(self, text="当前 Yaw2/Pitch2/Roll2").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=(4, 2)
        )
        tk.Label(
            self, textvariable=self.var_current, font=FONT_VALUE, anchor="w", width=24
        ).grid(
            row=2, column=2, columnspan=2, sticky="w"
        )
        self.btn_clear = tk.Button(self, text="清零", width=7, command=self.on_clear)
        self.btn_clear.grid(row=2, column=4, sticky="e", padx=4)

        # 第 3 行：转动角 theta 与转轴 L
        tk.Label(self, text="转动角 theta =").grid(row=3, column=0, sticky="w", padx=(4, 2))
        tk.Label(self, textvariable=self.var_theta, font=FONT_THETA, anchor="w").grid(
            row=3, column=1, sticky="w"
        )
        tk.Label(self, text="°", anchor="w").grid(row=3, column=2, sticky="w")
        tk.Label(self, text="转轴 L =", anchor="w").grid(row=3, column=3, sticky="w")
        tk.Label(self, textvariable=self.var_axis, font=FONT_VALUE, anchor="w", width=25).grid(
            row=3, column=4, columnspan=2, sticky="w"
        )

        # 第 4 行：错误/提示信息（就近显示，不只靠颜色）
        tk.Label(
            self,
            textvariable=self.var_message,
            font=FONT_SMALL,
            fg=COLOR_ERROR,
            anchor="w",
            justify="left",
            width=58,
            wraplength=430,
        ).grid(row=4, column=0, columnspan=6, sticky="w", padx=4)

    # ── 刷新 ───────────────────────────────────────────────────────

    @staticmethod
    def format_triple(yaw, pitch, roll) -> str:
        """定长格式化三方向角度，避免刷新时文字宽度抖动。"""
        if yaw is None or pitch is None or roll is None:
            return PLACEHOLDER_TRIPLE
        return "{:6.2f} / {:6.2f} / {:6.2f}".format(yaw, pitch, roll)

    def refresh(self):
        """从服务端拉取快照并刷新界面（不阻塞）。"""
        snap = self.service.get_snapshot()

        self.var_reference.set(
            self.format_triple(
                snap.reference.yaw if snap.reference else None,
                snap.reference.pitch if snap.reference else None,
                snap.reference.roll if snap.reference else None,
            )
        )
        self.var_current.set(
            self.format_triple(
                snap.current.yaw if snap.current else None,
                snap.current.pitch if snap.current else None,
                snap.current.roll if snap.current else None,
            )
        )

        if snap.theta_deg is None:
            self.var_theta.set("--")
        else:
            self.var_theta.set("{:6.2f}".format(snap.theta_deg))

        if snap.axis is None:
            self.var_axis.set("(--, --, --)")
        else:
            self.var_axis.set(
                "({:6.3f},{:6.3f},{:6.3f})".format(snap.axis[0], snap.axis[1], snap.axis[2])
            )

        self._set_status(snap)
        self.var_rate.set(
            "{:.1f} Hz".format(snap.rate_hz) if snap.connected and snap.rate_hz else ""
        )
        self.var_message.set(snap.error or "")
        self._set_connect_button(snap)

    def _set_status(self, snap):
        """状态：文字 + 颜色双通道表达。"""
        now = time.time()
        if now < self._flash_until and self._flash_text:
            self.var_status.set(self._flash_text)
            self.lbl_status.config(fg=COLOR_HINT)
            return
        if snap.connecting:
            text, color = "● 连接中…", COLOR_WARN
        elif not snap.connected:
            text, color = "● 未连接", COLOR_OFF
        elif not snap.fresh:
            text, color = "● 数据超时", COLOR_WARN
        else:
            text, color = "● 已连接", COLOR_OK
        if snap.temperature is not None and snap.connected:
            text += "  {:.1f}℃".format(snap.temperature)
        self.var_status.set(text)
        self.lbl_status.config(fg=color)

    def _set_connect_button(self, snap):
        if snap.connecting:
            self.btn_connect.config(text="连接中…", state="disabled")
        elif snap.connected:
            self.btn_connect.config(text="断开", state="normal")
        else:
            self.btn_connect.config(text="连接", state="normal")

    def _flash(self, text, seconds=2.0):
        self._flash_text = text
        self._flash_until = time.time() + seconds

    def _poll(self):
        """UI 线程轮询：有更新就刷新，另外每 20 个周期兜底刷新一次。"""
        self._ticks += 1
        try:
            if self.service.drain_ui_events() or self._ticks % 20 == 0:
                self.refresh()
        except Exception:
            pass
        try:
            self.after(self.poll_ms, self._poll)
        except tk.TclError:
            pass

    # ── 交互 ───────────────────────────────────────────────────────

    def on_record(self) -> bool:
        """记录：锁定当前姿态为初始姿态 (Yaw1, Pitch1, Roll1)。"""
        ok = self.service.record_reference()
        if ok:
            self._flash("已记录初始角度")
        self.refresh()
        if not ok and not self.var_message.get():
            self.var_message.set("暂无陀螺仪数据，无法记录")
        return ok

    def on_clear(self):
        """清零：丢弃基准，theta 回到未测量状态。"""
        self.service.clear_reference()
        self._flash("已清零")
        self.refresh()

    def on_connect_toggle(self):
        """连接/断开。串口操作在服务层子线程完成，这里立即返回。"""
        snap = self.service.get_snapshot()
        if snap.connected or snap.connecting:
            self.service.stop()
            self.refresh()
            return
        port = self.var_port.get().strip()
        if not port:
            self.var_message.set("请先填写或选择串口号，例如 COM3")
            return
        self.var_message.set("")
        self.btn_connect.config(text="连接中…", state="disabled")
        self.service.start(port)

    # ── 端口枚举（子线程 + after 回主线程） ────────────────────────

    def _load_ports_async(self):
        def worker():
            ports = self.service.available_ports()
            try:
                self.after(0, self._apply_ports, ports)
            except (tk.TclError, RuntimeError):
                pass  # 窗口已销毁

        threading.Thread(target=worker, daemon=True, name="GyroPanel-Ports").start()

    def _apply_ports(self, ports):
        if not ports:
            return
        try:
            self.cmb_port.config(values=list(ports))
            if self.var_port.get().strip() not in ports:
                self.var_port.set(ports[0])
        except tk.TclError:
            pass
