import cv2
import threading
import queue
import time
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import configpara
import capprocess
from collections import deque
import os
import optotunecontrol as mirctl
from services.gyro_service import GyroService
from views.gyro_panel import GyroPanel
from utils.resources import resource_path


class App:
    def __init__(self,window,window_title,video_source = 0):
        #信号灯
        self.openflag = False
        self.saveflag = False
        #新建Frame
        self.root = window
        self.root.title(window_title)
        #窗口尺寸先给一个下限，__init__ 末尾再按内容自适应（见 _fit_window）
        self.root.geometry("520x920")
        self.root.resizable(False,False)
        #关闭窗口时释放陀螺仪串口（关闭动作在子线程执行，主线程不卡顿）
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.fr_configpara = tk.LabelFrame(self.root,text="实验数据",relief="solid",bd = 2)
        self.fr_configpara.pack(anchor="w",padx=20)
        self.makemirrorStringVar()

        self.fr_image = tk.LabelFrame(self.root,text="点阵图片",relief="solid",bd = 2)
        self.fr_image.pack()
        self.framelabel = tk.Label(self.fr_image)
        self.framelabel.pack()

        self.fr_result = tk.LabelFrame(self.root,text="测试结果",bg="#9EE2EB",relief="solid",bd = 2)
        self.fr_result.pack(pady=10)
        self.makeresultsStringVar()
        #共享参数与线程控制
        self.para = configpara.configparameters()

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.para_queue = queue.Queue(maxsize=1)

        #存储管理
        self.resfifo = deque(maxlen = 20)
        #振镜初始化
        self.mir = mirctl.optotune()
        #打开摄像头
        self.capture = cv2.VideoCapture(video_source, cv2.CAP_ANY)  # 打开内置摄像头
        target_width = 1600  # 目标图像宽度
        target_height = 1200  # 目标图像高度
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        #相机打不开不再让程序直接崩掉：界面照常起来（参数、陀螺仪面板可用），
        #图像与测试结果区保持"no frame!"，方便在没有相机的机器上调试界面。
        self.camera_available = bool(self.capture.isOpened())
        if not self.camera_available:
            print("[相机] 未检测到可用相机（索引 %s）：图像采集与测试结果不会更新，其余功能正常。"
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

        #振镜控制显示
        self.mirrorGUI()

        #陀螺仪角度与转动角 theta 面板（设备访问/计算都在 services/ 里，按钮回调不阻塞）
        self.gyro_service = GyroService()
        self.gyro_panel = GyroPanel(self.fr_configpara, self.gyro_service)
        self.gyro_panel.grid(row=6,column=0,columnspan=9,sticky="we",padx=6,pady=(6,8))

        #实验图像显示
        self.update_frame()

        #显示测试结果
        self.update_result()

        #状态显示
        self.stateGUI()

        #标志位
        self.flagGUI()

        #按内容自适应窗口尺寸：不同 DPI/字体下不会把右侧控件裁掉
        self._fit_window()



    def getinputparameters(self):
        ri = float(self.__ent_ri.get())
        ro = float(self.__ent_ro.get())
        theta = float(self.__ent_theta.get())
        angle = float(self.__ent_angle.get())
        lens = float(self.__ent_inputD.get())
        self.para.setconfigparameters(ri,ro,theta,angle,lens)
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
    def makemirrorStringVar(self):
        self.actualdegreex = tk.StringVar(value="--")
        self.actualdegreey = tk.StringVar(value="--")
        tk.Label(self.fr_configpara,text="GETY:").grid(row=3,column=5)
        #固定宽度，避免振镜读数从 "--" 变成数值时撑宽面板
        tk.Label(self.fr_configpara,textvariable=self.actualdegreex,width=6,anchor="w").grid(row=3,column=6)
        

        tk.Label(self.fr_configpara,text="GETX:").grid(row=2,column=5)
        tk.Label(self.fr_configpara,textvariable=self.actualdegreey,width=6,anchor="w").grid(row=2,column=6)

    def makeresultsStringVar(self):
            self.beforedemod_a0 = tk.StringVar(value="--")
            self.beforedemod_a1 = tk.StringVar(value="--")
            self.beforedemod_a2 = tk.StringVar(value="--")
            self.beforedemod_a3 = tk.StringVar(value="--")
            self.beforedemod_a4 = tk.StringVar(value="--")
            self.beforedemod_a5 = tk.StringVar(value="--")

            self.afterdemod_b3 = tk.StringVar(value="--")
            self.afterdemod_b4 = tk.StringVar(value="--")
            self.afterdemod_b5 = tk.StringVar(value="--")

            self.afterdemod_Mx = tk.StringVar(value="--")
            self.afterdemod_My = tk.StringVar(value="--")

            self.afterdemod_sph = tk.StringVar(value="--")
            self.afterdemod_cyl = tk.StringVar(value="--")
            self.afterdemod_axis = tk.StringVar(value="--")


            tk.Label(self.fr_result,text="zernike a: ",bg="#EEA9B8",width=15).grid(row=0,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a0,bg="#9EE2EB").grid(row=1,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a1,bg="#9EE2EB").grid(row=2,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a2,bg="#9EE2EB").grid(row=3,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a3,bg="#9EE2EB").grid(row=4,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a4,bg="#9EE2EB").grid(row=5,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a5,bg="#9EE2EB").grid(row=6,column=0)

            tk.Label(self.fr_result,text="zernike b: ",bg="#EEA9B8",width=15).grid(row=0,column=1)
            tk.Label(self.fr_result,textvariable=self.afterdemod_b3,bg="#9EE2EB").grid(row=1,column=1)
            tk.Label(self.fr_result,textvariable=self.afterdemod_b4,bg="#9EE2EB").grid(row=2,column=1)
            tk.Label(self.fr_result,textvariable=self.afterdemod_b5,bg="#9EE2EB").grid(row=3,column=1)

            tk.Label(self.fr_result,text="Mx,My: ",bg="#EEA9B8",width=15).grid(row=0,column=2)
            tk.Label(self.fr_result,textvariable=self.afterdemod_Mx,bg="#9EE2EB").grid(row=1,column=2)
            tk.Label(self.fr_result,textvariable=self.afterdemod_My,bg="#9EE2EB").grid(row=2,column=2)

            tk.Label(self.fr_result,text="sph,cyl,axis: ",bg="#EEA9B8",width=15).grid(row=0,column=3)
            tk.Label(self.fr_result,textvariable=self.afterdemod_sph,bg="#9EE2EB").grid(row=1,column=3)
            tk.Label(self.fr_result,textvariable=self.afterdemod_cyl,bg="#9EE2EB").grid(row=2,column=3)
            tk.Label(self.fr_result,textvariable=self.afterdemod_axis,bg="#9EE2EB").grid(row=3,column=3)

    
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
        savefolder = "res"
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        name = str(self.para.direction)+"_angle"+str(self.para.angle)+"_"+str(self.para.inputD)+"D.jpg"
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


        lab_ent_ri1 = tk.Label(self.fr_configpara,text="入瞳半径：").grid(row=0,column=0)
        ri_default = tk.DoubleVar(value=self.para.ri)
        self.__ent_ri = tk.Entry(self.fr_configpara,textvariable=ri_default,width=5)
        self.__ent_ri.grid(row=0,column=1)
        lab_ent_ri2 = tk.Label(self.fr_configpara,text="mm").grid(row=0,column=2)

        lab_ent_ro1 = tk.Label(self.fr_configpara,text="出瞳半径：").grid(row=1,column=0)
        ro_default = tk.DoubleVar(value=self.para.ro)
        self.__ent_ro = tk.Entry(self.fr_configpara,textvariable=ro_default,width=5)
        self.__ent_ro.grid(row=1,column=1)
        lab_ent_ro2 = tk.Label(self.fr_configpara,text="mm").grid(row=1,column=2)

        lab_ent_theta1 = tk.Label(self.fr_configpara,text="视场角theta：").grid(row=2,column=0)
        theta_default = tk.DoubleVar(value=self.para.theta)
        self.__ent_theta = tk.Entry(self.fr_configpara,textvariable=theta_default,width=5)
        self.__ent_theta.grid(row=2,column=1)
        lab_ent_theta2 = tk.Label(self.fr_configpara,text="deg").grid(row=2,column=2)

        lab_ent_angle1 = tk.Label(self.fr_configpara,text="视场角angle：").grid(row=3,column=0)
        angle_default = tk.DoubleVar(value=self.para.angle)
        self.__ent_angle = tk.Entry(self.fr_configpara,textvariable=angle_default,width=5)
        self.__ent_angle.grid(row=3,column=1)
        lab_ent_angle2 = tk.Label(self.fr_configpara,text="deg").grid(row=3,column=2)

        lab_ent_inputD1 = tk.Label(self.fr_configpara,text="试镜片：").grid(row=4,column=0)
        inputD_default = tk.DoubleVar(value=self.para.inputD)
        self.__ent_inputD = tk.Entry(self.fr_configpara,textvariable=inputD_default,width=5)
        self.__ent_inputD.grid(row=4,column=1)
        lab_ent_inputD2 = tk.Label(self.fr_configpara,text="D").grid(row=4,column=2)
        
        Bt = tk.Button(self.fr_configpara,text="确定",command=self.getinputparameters,width=10).grid(row=5,column=1,columnspan=2)

    def stateGUI(self):
        self.mode = tk.IntVar()
        mode1 = tk.Radiobutton(self.fr_configpara,text = "本征像差",variable=self.mode,value=1)
        mode1.grid(row=0,column=3)
        mode2 = tk.Radiobutton(self.fr_configpara,text = "测试模式",variable=self.mode,value=2)
        mode2.grid(row=1,column=3)
        
        tk.Button(self.fr_configpara,text="保存数据",command=self.saveresults,width=10).grid(row=2,column=3)
        tk.Button(self.fr_configpara,text="保存图像",command=self.saveframe,width=10).grid(row=3,column=3)
        tk.Button(self.fr_configpara,text="清空数据",command=self.cleardataqueue,width=10).grid(row = 4, column=3)

    def mirrorGUI(self):
        #振镜输入
        tk.Label(self.fr_configpara,text="SETY:").grid(row=1,column=5)
        targetx_default = tk.DoubleVar(value=0.0)
        self.__ent_inputx = tk.Entry(self.fr_configpara,textvariable=targetx_default,width=5)
        self.__ent_inputx.grid(row=1,column=6)
        tk.Label(self.fr_configpara,text="° ").grid(row=1,column=7)

        tk.Label(self.fr_configpara,text="SETX:").grid(row=0,column=5)
        targety_default = tk.DoubleVar(value=0.0)
        self.__ent_inputy = tk.Entry(self.fr_configpara,textvariable=targety_default,width=5)
        self.__ent_inputy.grid(row=0,column=6)
        tk.Label(self.fr_configpara,text="° ").grid(row=0,column=7)

        tk.Button(self.fr_configpara,text="确定",command=self.setmirrortarget,width=10).grid(row=0,column=8)
        tk.Button(self.fr_configpara,text="置零",command=self.setmirrorzero,width=10).grid(row=1,column=8)
        #读振镜
        self.update_mirror()
    def flagGUI(self):
        tk.Label(self.fr_configpara,text="SaveFlag:").grid(row=5,column=3)
        self.indicator = tk.Label(self.fr_configpara,text="●",font=("Arial",16),fg="red")
        self.indicator.grid(row= 5,column = 4)

        self.update_flag()
    def update_frame(self):

        try:
            self.currentframe = self.frame_queue.get_nowait()          
   
            img = Image.fromarray(self.currentframe)
            img = img.resize((450,337),Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.framelabel.imgtk = imgtk
            self.framelabel.config(image=imgtk)
        except queue.Empty:
            pass
        if not self.openflag:
            #用动态资源路径，兼容 PyInstaller 打包（sys._MEIPASS）
            img = Image.open(resource_path("no frame.png")).convert("L")  # 关键：convert("L")
            img = img.resize((450, 337), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)

            self.framelabel.imgtk = imgtk
            self.framelabel.config(image=imgtk)
        self.root.after(30, self.update_frame)
        # 30ms 约 33 fps
         


    def update_result(self):


        try:
            result = self.result_queue.get_nowait()
            self.beforedemod_a0.set(f"{result.beforedemod['a0']:.3f}") 
            self.beforedemod_a1.set(f"{result.beforedemod['a1']:.3f}") 
            self.beforedemod_a2.set(f"{result.beforedemod['a2']:.3f}") 
            self.beforedemod_a3.set(f"{result.beforedemod['a3']:.3f}") 
            self.beforedemod_a4.set(f"{result.beforedemod['a4']:.3f}") 
            self.beforedemod_a5.set(f"{result.beforedemod['a5']:.3f}") 

            self.afterdemod_b3.set(f"{result.afterdemod['b3']:.3f}") 
            self.afterdemod_b4.set(f"{result.afterdemod['b4']:.3f}") 
            self.afterdemod_b5.set(f"{result.afterdemod['b5']:.3f}") 

            self.afterdemod_Mx.set(f"{result.Mx:.3f}")
            self.afterdemod_My.set(f"{result.My:.3f}")

            self.afterdemod_sph.set(f"{result.sph:.3f}")
            self.afterdemod_cyl.set(f"{result.cyl:.3f}")
            self.afterdemod_axis.set(f"{result.axis:.3f}")

            self.resfifo.append(result)
            if len(self.resfifo) == self.resfifo.maxlen:
                self.saveflag = True

        except queue.Empty:
            pass
        self.root.after(1000, self.update_result)
        # 30ms 约 33 fps

    def update_mirror(self):
        actualdegreex,actualdegreey = self.mir.getxy()

        self.actualdegreex.set(f"{actualdegreex}°")
        self.actualdegreey.set(f"{actualdegreey}°")

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
        """按控件实际需求尺寸调整窗口，保证各面板完整可见。"""
        try:
            self.root.update_idletasks()
            width = max(520, self.root.winfo_reqwidth())
            height = max(920, self.root.winfo_reqheight())
            self.root.geometry("{}x{}".format(width, height))
        except Exception:
            pass
    
