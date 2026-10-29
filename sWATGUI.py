import cv2
import threading
import queue
import time
from PIL import Image, ImageTk
import tkinter as tk
import configpara
import capprocess
from collections import deque
import os
import optotunecontrol as mirctl
from services.gyro_service import GyroService
from views.gyro_panel import GyroPanel
from utils.resources import resource_path

# 界面上点阵图片的显示尺寸：保持 4:3，与相机 1600x1200 一致
IMAGE_WIDTH = 520
IMAGE_HEIGHT = 390
# 结果队列只用于“保存数据”和 SaveFlag，不再显示在界面上
RESULT_POLL_MS = 500
# 窗口最小尺寸；内容更大时由 _fit_window() 按实际需求放大
MIN_WINDOW_WIDTH = 480
MIN_WINDOW_HEIGHT = 560


class App:
    def __init__(self,window,window_title,video_source = 0):
        #信号灯
        self.openflag = False
        self.saveflag = False
        #新建Frame
        self.root = window
        self.root.title(window_title)
        #窗口尺寸先给一个下限，__init__ 末尾再按内容自适应（见 _fit_window）
        self.root.geometry("{}x{}".format(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT))
        self.root.resizable(False,False)
        #关闭窗口时释放陀螺仪串口（关闭动作在子线程执行，主线程不卡顿）
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        #统一外边距：三块面板左右对齐，窗口边缘留白一致
        self.container = tk.Frame(self.root)
        self.container.pack(fill="both",expand=True,padx=12,pady=10)

        self.fr_configpara = tk.LabelFrame(self.container,text="实验数据",relief="solid",bd = 2)
        self.fr_configpara.pack(fill="x")

        #面板内部三列：光学参数 / 模式与操作 / 振镜，下面再横跨一整行放陀螺仪
        self.box_grid = tk.Frame(self.fr_configpara)
        self.box_grid.pack(fill="x",padx=8,pady=6)
        self.box_optics = tk.Frame(self.box_grid)
        self.box_optics.grid(row=0,column=0,sticky="nw")
        self.box_actions = tk.Frame(self.box_grid)
        self.box_actions.grid(row=0,column=1,sticky="nw",padx=(16,16))
        self.box_mirror = tk.Frame(self.box_grid)
        self.box_mirror.grid(row=0,column=2,sticky="nw")

        self.fr_image = tk.LabelFrame(self.container,text="点阵图片",relief="solid",bd = 2)
        self.fr_image.pack(pady=(10,0))
        self.framelabel = tk.Label(self.fr_image)
        self.framelabel.pack(padx=8,pady=8)

        #共享参数与线程控制
        self.para = configpara.configparameters()

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.para_queue = queue.Queue(maxsize=1)

        #存储管理
        self.resfifo = deque(maxlen = 10)
        #最近一帧画面：没有相机时保持 None，“保存图像”会据此提示而不是抛异常
        self.currentframe = None
        #振镜初始化
        self.mir = mirctl.optotune()
        #打开摄像头
        self.capture = cv2.VideoCapture(video_source, cv2.CAP_ANY)  # 打开内置摄像头
        target_width = 1600  # 目标图像宽度
        target_height = 1200  # 目标图像高度
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        #相机打不开不再让程序直接崩掉：界面照常起来（参数、陀螺仪面板可用），
        #图像区保持"no frame!"，方便在没有相机的机器上调试界面。
        self.camera_available = bool(self.capture.isOpened())
        if not self.camera_available:
            print("[相机] 未检测到可用相机（索引 %s）：图像采集不会更新，其余功能正常。"
                  % video_source)

        #启动数据处理线程

        self.thread = threading.Thread(
            target=capprocess.captureprocess,
            args=(self.capture,self.frame_queue,self.result_queue,self.para_queue),
            daemon=True
        )

        self.thread.start()

        #实验参数控制
        self.configGUI()

        #实验模式与操作按钮
        self.stateGUI()

        #振镜控制显示
        self.mirrorGUI()

        #状态显示（保存指示灯）
        self.flagGUI()

        #陀螺仪角度与转动角 theta 面板（设备访问/计算都在 services/ 里，按钮回调不阻塞）
        self.gyro_service = GyroService()
        self.gyro_panel = GyroPanel(self.box_grid, self.gyro_service)
        self.gyro_panel.grid(row=1,column=0,columnspan=3,sticky="we",pady=(8,0))

        #实验图像显示
        self.update_frame()

        #后台结果只入队（供“保存数据”与 SaveFlag 使用），不显示在界面上
        self._poll_results()

        #按内容自适应窗口尺寸：不同 DPI/字体下不会把控件裁掉
        self._fit_window()



    def getinputparameters(self):
        ri = float(self.__ent_ri.get())
        ro = float(self.__ent_ro.get())
        #界面上的“坐标x/坐标y”对应内部配置里的 angle/theta（theta 决定 direction）
        viewx = float(self.__ent_x.get()) #angle ->x
        viewy = float(self.__ent_y.get()) #theta ->y
        sph = float(self.__ent_inputsph.get())
        cyl = float(self.__ent_inputcyl.get())
        axis = float(self.__ent_inputaxis.get())
        lens = [sph,cyl,axis]
        self.para.setconfigparameters(ri,ro,viewy,viewx,lens)
        self.para.printpara()
        self.openflag = True
        try:
            self.para_queue.put_nowait(self.para)
        except queue.Full:
            # 丢弃旧帧
            try:
                self.para_queue.get_nowait()
            except queue.Empty:
                pass
            self.para_queue.put_nowait(self.para)

    def saveresults(self):
        savefolder = "res"
        timestamp = time.strftime("%Y-%m-%d %H-%M-%S")

        if not os.path.exists(savefolder):
                os.makedirs(savefolder)

        savepath = os.path.join(savefolder,timestamp)


        if len(self.resfifo) == self.resfifo.maxlen :
            for res in self.resfifo:
                res.save(savepath)
            self.resfifo.clear()
            self.saveflag = False
            print("结果已保存！")
        else :
            print("数据量不足，不能保存！")



    def saveframe(self):
        if self.currentframe is None:
            print("没有可用画面，不能保存图像！")
            return

        savefolder = "res"
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        #文件名与界面一致：坐标x（内部 angle）、坐标y（内部 theta）
        #例：shuiping_x3.0_y90.0_input 1.00 0.50 10.00.jpg
        name = "image_x{}_y{}_input {} {} {}.jpg".format(
            self.para.viewx,self.para.viewy,
            self.para.inputsph,self.para.inputcyl,self.para.inputaxis)
        savepath = os.path.join(savefolder,name)
        cv2.imwrite(savepath,self.currentframe)
    def cleardataqueue(self):
        self.resfifo.clear()
        self.saveflag = False


    def setmirrortarget(self):
        degreetargetx = float(self.__ent_inputx.get())
        degreetargety = float(self.__ent_inputy.get())
        self.mir.setxy(degreetargetx,degreetargety)

    def setmirrorzero(self):
        self.mir.setzero()

    def configGUI(self):
        """光学参数列：入瞳/出瞳半径、坐标 x/y、试镜片 sph/cyl/axis。"""
        ent_w = 6

        def add_row(row,text,var,unit=None):
            tk.Label(self.box_optics,text=text,anchor="w").grid(
                row=row,column=0,sticky="w",pady=2)
            entry = tk.Entry(self.box_optics,textvariable=var,width=ent_w,justify="right")
            entry.grid(row=row,column=1,sticky="w",pady=2)
            if unit:
                tk.Label(self.box_optics,text=unit,anchor="w").grid(
                    row=row,column=2,sticky="w",padx=(4,0))
            return entry

        self.__var_ri = tk.DoubleVar(value=self.para.ri)
        self.__ent_ri = add_row(0,"入瞳半径：",self.__var_ri,"mm")

        self.__var_ro = tk.DoubleVar(value=self.para.ro)
        self.__ent_ro = add_row(1,"出瞳半径：",self.__var_ro,"mm")

        #视场角angle -> 坐标x，视场角theta -> 坐标y
        self.__var_x = tk.DoubleVar(value=self.para.viewx)
        self.__ent_x = add_row(2,"坐标x：",self.__var_x,"deg")

        self.__var_y = tk.DoubleVar(value=self.para.viewy)
        self.__ent_y = add_row(3,"坐标y：",self.__var_y,"deg")

        self.__var_inputsph = tk.DoubleVar(value=self.para.inputsph)
        self.__ent_inputsph = add_row(4,"sph(D)：",self.__var_inputsph)

        #cyl 与 axis 同一行：两个输入框
        tk.Label(self.box_optics,text="cyl(D),axis(°)：",anchor="w").grid(
            row=5,column=0,sticky="w",pady=2)
        self.__var_inputcyl = tk.DoubleVar(value=self.para.inputcyl)
        self.__ent_inputcyl = tk.Entry(
            self.box_optics,textvariable=self.__var_inputcyl,width=5,justify="right")
        self.__ent_inputcyl.grid(row=5,column=1,sticky="w",pady=2)
        self.__var_inputaxis = tk.DoubleVar(value=self.para.inputaxis)
        self.__ent_inputaxis = tk.Entry(
            self.box_optics,textvariable=self.__var_inputaxis,width=5,justify="right")
        self.__ent_inputaxis.grid(row=5,column=2,sticky="w",padx=(4,0),pady=2)

    def stateGUI(self):
        """模式选择与四个操作按钮。"""
        self.mode = tk.IntVar()
        tk.Radiobutton(self.box_actions,text="本征像差",variable=self.mode,value=1,
                       anchor="w").grid(row=0,column=0,sticky="w")
        tk.Radiobutton(self.box_actions,text="测试模式",variable=self.mode,value=2,
                       anchor="w").grid(row=1,column=0,sticky="w")

        tk.Button(self.box_actions,text="写入参数",command=self.getinputparameters,
                  width=10).grid(row=2,column=0,sticky="w",pady=(10,3))
        tk.Button(self.box_actions,text="保存数据",command=self.saveresults,
                  width=10).grid(row=3,column=0,sticky="w",pady=3)
        tk.Button(self.box_actions,text="保存图像",command=self.saveframe,
                  width=10).grid(row=4,column=0,sticky="w",pady=3)
        tk.Button(self.box_actions,text="清空数据",command=self.cleardataqueue,
                  width=10).grid(row=5,column=0,sticky="w",pady=3)

    def mirrorGUI(self):
        """振镜列：SETX/SETY 目标、确定/置零、GETX/GETY 读数。"""
        m = self.box_mirror

        tk.Label(m,text="SETX：",anchor="w").grid(row=0,column=0,sticky="w",pady=2)
        self.__var_inputx = tk.DoubleVar(value=0.0)
        self.__ent_inputx = tk.Entry(m,textvariable=self.__var_inputx,width=6,justify="right")
        self.__ent_inputx.grid(row=0,column=1,sticky="w",pady=2)
        tk.Label(m,text="°",anchor="w").grid(row=0,column=2,sticky="w",padx=(2,8))
        tk.Button(m,text="确定",command=self.setmirrortarget,width=8).grid(
            row=0,column=3,sticky="w",pady=2)

        tk.Label(m,text="SETY：",anchor="w").grid(row=1,column=0,sticky="w",pady=2)
        self.__var_inputy = tk.DoubleVar(value=0.0)
        self.__ent_inputy = tk.Entry(m,textvariable=self.__var_inputy,width=6,justify="right")
        self.__ent_inputy.grid(row=1,column=1,sticky="w",pady=2)
        tk.Label(m,text="°",anchor="w").grid(row=1,column=2,sticky="w",padx=(2,8))
        tk.Button(m,text="置零",command=self.setmirrorzero,width=8).grid(
            row=1,column=3,sticky="w",pady=2)

        #读振镜读数：GETX / GETY 的取值见 update_mirror()（与界面上的“坐标x/坐标y”不是同一套量）
        self.getx_value = tk.StringVar(value="--")
        self.gety_value = tk.StringVar(value="--")
        tk.Label(m,text="GETX：",anchor="w").grid(row=2,column=0,sticky="w",pady=(6,2))
        tk.Label(m,textvariable=self.getx_value,anchor="w",width=7).grid(
            row=2,column=1,sticky="w",columnspan=2,pady=(6,2))
        tk.Label(m,text="GETY：",anchor="w").grid(row=3,column=0,sticky="w",pady=2)
        tk.Label(m,textvariable=self.gety_value,anchor="w",width=7).grid(
            row=3,column=1,sticky="w",columnspan=2,pady=2)

        self.update_mirror()

    def flagGUI(self):
        """保存指示灯：攒够 10 组结果后变绿。"""
        tk.Label(self.box_mirror,text="SaveFlag：",anchor="w").grid(
            row=4,column=0,sticky="w",pady=(8,2))
        self.indicator = tk.Label(self.box_mirror,text="●",font=("Arial",14),fg="red",
                                  anchor="w")
        self.indicator.grid(row=4,column=1,sticky="w",columnspan=2,pady=(8,2))

        self.update_flag()

    def update_frame(self):

        try:
            self.currentframe = self.frame_queue.get_nowait()

            img = Image.fromarray(self.currentframe)
            img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            self.framelabel.imgtk = imgtk
            self.framelabel.config(image=imgtk)
        except queue.Empty:
            pass
        if not self.openflag:
            #用动态资源路径，兼容 PyInstaller 打包（sys._MEIPASS）
            img = Image.open(resource_path("no frame.png")).convert("L")  # 关键：convert("L")
            img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            self.framelabel.imgtk = imgtk
            self.framelabel.config(image=imgtk)
        self.root.after(30, self.update_frame)
        # 30ms 约 33 fps


    def _poll_results(self):
        """后台结果只入队到 resfifo（供“保存数据”用）与 SaveFlag，界面上不再显示。"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self.resfifo.append(result)
                if len(self.resfifo) == self.resfifo.maxlen:
                    self.saveflag = True
        except queue.Empty:
            pass
        self.root.after(RESULT_POLL_MS, self._poll_results)

    def update_mirror(self):
        #振镜读数与界面上的“坐标x/坐标y”不是同一套量，这里是按实机标定定下来的对应关系：
        #GETX 显示 getxy() 第一路读数取反，GETY 显示第二路读数。
        #（这不是笔误，别再按“x 对 x、y 对 y”去“修正”）
        mirror_ch0,mirror_ch1 = self.mir.getxy()

        #取反后 0 会变成 -0.0，显示上归零
        getx = 0.0 if mirror_ch0 == 0 else -mirror_ch0
        self.getx_value.set(f"{getx}°")
        self.gety_value.set(f"{mirror_ch1}°")

        self.root.after(50,self.update_mirror)

    def update_flag(self):
        if self.saveflag :
            self.indicator.config(fg="green")
        else:
            self.indicator.config(fg = "red")

        self.root.after(50, self.update_flag)

    def on_close(self):
        """关闭窗口：先让服务在子线程释放串口，再销毁窗口，避免卡顿与句柄残留。"""
        try:
            self.gyro_service.stop()
        except Exception:
            pass
        self.root.destroy()

    def _fit_window(self):
        """按控件实际需求尺寸调整窗口，并让窗口在屏幕上居中。"""
        try:
            self.root.update_idletasks()
            width = max(MIN_WINDOW_WIDTH, self.root.winfo_reqwidth())
            height = max(MIN_WINDOW_HEIGHT, self.root.winfo_reqheight())
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            x = max(0, (screen_w - width) // 2)
            y = max(0, (screen_h - height) // 3)
            self.root.geometry("{}x{}+{}+{}".format(width, height, x, y))
        except Exception:
            pass
