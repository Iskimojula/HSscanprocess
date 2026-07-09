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
class App:
    def __init__(self,window,window_title,video_source = 0):
        #信号灯
        self.openflag = False
        self.saveflag = False
        #新建Frame
        self.root = window
        self.root.title(window_title)
        self.root.geometry("500x800")
        self.root.resizable(False,False)

        self.fr_configpara = tk.LabelFrame(self.root,text="实验数据",relief="solid",bd = 2)
        self.fr_configpara.pack(anchor="w",padx=20)

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

        #打开摄像头
        self.capture = cv2.VideoCapture(video_source, cv2.CAP_ANY)  # 打开内置摄像头
        target_width = 1600  # 目标图像宽度
        target_height = 1200  # 目标图像高度
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        if not self.capture.isOpened():
            raise ValueError("无法打开摄像头")

        #启动数据处理线程
        
        self.thread = threading.Thread(
            target=capprocess.captureprocess,
            args=(self.capture,self.frame_queue,self.result_queue,self.para_queue),
            daemon=True
        )

        self.thread.start()
        
        #实验参数控制
        self.configGUI()

        #实验图像显示
        self.update_frame()

        #显示测试结果
        self.update_result()

        #状态显示
        self.stateGUI()



    def getinputparameters(self):
        ri = float(self.__ent_ri.get())
        ro = float(self.__ent_ro.get())
        theta = float(self.__ent_theta.get())
        angle = float(self.__ent_angle.get())
        self.para.setconfigparameters(ri,ro,theta,angle)
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
            tk.Label(self.fr_result,textvariable=self.beforedemod_a0).grid(row=1,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a1).grid(row=2,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a2).grid(row=3,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a3).grid(row=4,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a4).grid(row=5,column=0)
            tk.Label(self.fr_result,textvariable=self.beforedemod_a5).grid(row=6,column=0)

            tk.Label(self.fr_result,text="zernike b: ",bg="#EEA9B8",width=15).grid(row=0,column=1)
            tk.Label(self.fr_result,textvariable=self.afterdemod_b3).grid(row=1,column=1)
            tk.Label(self.fr_result,textvariable=self.afterdemod_b4).grid(row=2,column=1)
            tk.Label(self.fr_result,textvariable=self.afterdemod_b5).grid(row=3,column=1)

            tk.Label(self.fr_result,text="Mx,My: ",bg="#EEA9B8",width=15).grid(row=0,column=2)
            tk.Label(self.fr_result,textvariable=self.afterdemod_Mx).grid(row=1,column=2)
            tk.Label(self.fr_result,textvariable=self.afterdemod_My).grid(row=2,column=2)

            tk.Label(self.fr_result,text="sph,cyl,axis: ",bg="#EEA9B8",width=15).grid(row=0,column=3)
            tk.Label(self.fr_result,textvariable=self.afterdemod_sph).grid(row=1,column=3)
            tk.Label(self.fr_result,textvariable=self.afterdemod_cyl).grid(row=2,column=3)
            tk.Label(self.fr_result,textvariable=self.afterdemod_axis).grid(row=3,column=3)

    def saveresults(self):
        savefolder = "res"
        timestamp = time.strftime("%Y-%m-%d %H-%M-%S")
        
        if not os.path.exists(savefolder):
                os.makedirs(savefolder)
        
        savepath = os.path.join(savefolder,timestamp)

        
        if len(self.resfifo) == self.resfifo.maxlen :
            for res in self.resfifo:
                res.save(savepath)
            
            print("结果已保存！")
        else :
            print("数据量不足，不能保存！")

        

    def saveframe(self):
        pass
        
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
        
        tk.Button(self.fr_configpara,text="保存数据",command=self.saveresults,width=10).grid(row=0,column=4)
        tk.Button(self.fr_configpara,text="保存图像",command=self.saveframe,width=10).grid(row=1,column=4)
    def update_frame(self):

        try:
            frame = self.frame_queue.get_nowait()          
   
            img = Image.fromarray(frame)
            img = img.resize((450,337),Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.framelabel.imgtk = imgtk
            self.framelabel.config(image=imgtk)
        except queue.Empty:
            pass
        if not self.openflag:
            img = Image.open("no frame.png").convert("L")  # 关键：convert("L")
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



    