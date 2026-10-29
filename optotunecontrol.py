import numpy as np
import time

#optotune MDC 的 Python 包随 Optotune MDC 软件一起安装，不是 pip 包。
#没装（或没插振镜）时不让整个程序起不来：退化为"振镜不可用"，其余功能照常。
try:
    import optoMDC  # type: ignore[import]  # 确保已安装此库
    MIRROR_SDK_AVAILABLE = True
    MIRROR_IMPORT_ERROR = None
except ImportError as _exc:  # pragma: no cover - 取决于本机是否装了 MDC 软件
    optoMDC = None
    MIRROR_SDK_AVAILABLE = False
    MIRROR_IMPORT_ERROR = _exc
    print("[振镜] 未检测到 Optotune MDC 的 Python 包 optoMDC：振镜控制不可用，其余功能正常。")

def deg_to_xy(mech_deg):         
    return np.tan(np.deg2rad(2 * mech_deg)) / np.tan(np.deg2rad(50))

def xy_to_deg(xy):                
    return 0.5 * np.rad2deg(np.arctan(xy * np.tan(np.deg2rad(50))))

class optotune:
    def __init__(self):
        #没有 SDK 时进入"未连接"模式：读回 0°，设置目标为无操作
        self.available = False
        self.mre2 = None
        if not MIRROR_SDK_AVAILABLE:
            return
        try:
            self.mre2 = optoMDC.connect()
        except Exception as exc:  # 装了 MDC 软件但没接振镜/被占用
            self.mre2 = None
            print("[振镜] 连接 Optotune MRE2 失败（%s）：振镜控制不可用，其余功能正常。" % exc)
            return
        self.mre2.Mirror.Channel_0.SetControlMode(2)
        self.mre2.Mirror.Channel_1.SetControlMode(2)
        self.mre2.Mirror.Channel_0.StaticInput.SetAsInput()
        self.mre2.Mirror.Channel_1.StaticInput.SetAsInput()
        self.available = True
    def setdegreeX(self,target_x):
        """原始通道接口：把角度直接写到 channel_0（不含用户坐标标定，界面请用 setxy）。"""
        if not self.available:
            return
        x = deg_to_xy(target_x)
        self.mre2.Mirror.Channel_0.StaticInput.SetXY(x)
    def setdegreeY(self,target_y):
        """原始通道接口：把角度直接写到 channel_1（不含用户坐标标定，界面请用 setxy）。"""
        if not self.available:
            return
        y = deg_to_xy(target_y)
        self.mre2.Mirror.Channel_1.StaticInput.SetXY(y)
    
    def getdegreex(self):
        if not self.available:
            return 0.0
        actual_x = self.mre2.Mirror.Channel_0.StaticInput.GetXY()
        
        return xy_to_deg(float(actual_x[0]))


    def getdegreey(self):
        if not self.available:
            return 0.0
        actual_y = self.mre2.Mirror.Channel_1.StaticInput.GetXY()
        return xy_to_deg(float(actual_y[0]))
    
    def setxy(self,target_degreex,target_degreey):
        #用户坐标 (x, y) 与振镜两路通道按实机标定互换并取反（相当于坐标轴旋转 90°）：
        #   channel_0 = -y，channel_1 = x
        #这样输入 SETX=2.2 / SETY=-0.7 打出的振镜状态，
        #与旧版直接输入 SETX=0.7 / SETY=2.2 完全相同。
        self.setdegreeX(-target_degreey)
        self.setdegreeY(target_degreex)
    def setzero(self):
        self.setdegreeX(0)
        self.setdegreeY(0)
    def getxy(self):
        #把两路通道读数换算回用户坐标，与 setxy 的输入约定一致：
        #   x = channel_1，y = -channel_0
        channel_0 = self.getdegreex()
        channel_1 = self.getdegreey()
        return round(channel_1,1), round(-channel_0,1)
    



