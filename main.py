import cv2
import threading
import queue
import time
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import configpara
import capprocess
class App:
    def __init__(self,window,window_title,video_source = 0):
        self.window = window
        self.window.title(window_title)

        #共享参数与线程控制
        self.para = configpara.configparameters()

        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.para_queue = queue.Queue(maxsize=1)

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
        

        #-----GUI-----
        self.root = tk.Tk()
        self.root.title("扫描式HS波前像差仪")
        self.root.geometry("500x800")
        self.root.resizable(False,False)

        #实验参数控制
        self.configGUI()

        #实验图像显示
        self.update_frame()

        #显示测试结果
        self.update_result()



    def getinputparameters(self):
        ri = float(self.__ent_ri.get())
        ro = float(self.__ent_ro.get())
        theta = float(self.__ent_theta.get())
        angle = float(self.__ent_angle.get())
        self.para.setconfigparameters(ri,ro,theta,angle)
        self.para.printpara()
        try:
            self.para_queue.put_nowait(self.para)
        except queue.Full:
            # 丢弃旧帧
            try:
                self.para_queue.get_nowait()
            except queue.Empty:
                pass
            self.para_queue.put_nowait(self.para)

    def configGUI(self):
        fr_configpara = tk.LabelFrame(self.root,text="实验参数",relief="solid",bd = 2)
        fr_configpara.pack(anchor="w")

        lab_ent_ri1 = tk.Label(fr_configpara,text="入瞳半径：").grid(row=0,column=0)
        ri_default = tk.DoubleVar(value=self.para.ri)
        self.__ent_ri = tk.Entry(fr_configpara,textvariable=ri_default,width=5)
        self.__ent_ri.grid(row=0,column=1)
        lab_ent_ri2 = tk.Label(fr_configpara,text="mm").grid(row=0,column=2)

        lab_ent_ro1 = tk.Label(fr_configpara,text="出瞳半径：").grid(row=1,column=0)
        ro_default = tk.DoubleVar(value=self.para.ro)
        self.__ent_ro = tk.Entry(fr_configpara,textvariable=ro_default,width=5)
        self.__ent_ro.grid(row=1,column=1)
        lab_ent_ro2 = tk.Label(fr_configpara,text="mm").grid(row=1,column=2)

        lab_ent_theta1 = tk.Label(fr_configpara,text="视场角theta：").grid(row=2,column=0)
        theta_default = tk.DoubleVar(value=self.para.theta)
        self.__ent_theta = tk.Entry(fr_configpara,textvariable=theta_default,width=5)
        self.__ent_theta.grid(row=2,column=1)
        lab_ent_theta2 = tk.Label(fr_configpara,text="deg").grid(row=2,column=2)

        lab_ent_angle1 = tk.Label(fr_configpara,text="视场角angle：").grid(row=3,column=0)
        angle_default = tk.DoubleVar(value=self.para.angle)
        self.__ent_angle = tk.Entry(fr_configpara,textvariable=angle_default,width=5)
        self.__ent_angle.grid(row=3,column=1)
        lab_ent_angle2 = tk.Label(fr_configpara,text="deg").grid(row=3,column=2)

        lab_ent_inputD1 = tk.Label(fr_configpara,text="试镜片：").grid(row=4,column=0)
        inputD_default = tk.DoubleVar(value=self.para.inputD)
        self.__ent_inputD = tk.Entry(fr_configpara,textvariable=inputD_default,width=5)
        self.__ent_inputD.grid(row=4,column=1)
        lab_ent_inputD2 = tk.Label(fr_configpara,text="D").grid(row=4,column=2)
        
        Bt = tk.Button(fr_configpara,text="确定",command=self.getinputparameters,width=10).grid(row=5,column=1,columnspan=3)


    def update_frame(self):
        fr_image = tk.LabelFrame(tk.root,text="点阵图片",relief="solid",bd = 2)
        fr_image.pack(anchor="w")
        try:
            frame = self.frame_queue.get_nowait()          
   
            img = Image.fromarray(frame)
            img.resize((400,300),Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            tk.Label(fr_image,image=img).pack()
        except self.frame_queue.empty():
            pass
        # 30ms 约 33 fps
         


    def update_result(self):
        fr_result = tk.LabelFrame(self.root,text="测试结果",bg="#9EE2EB",relief="solid",bd = 2)
        fr_result.pack(anchor="w")

        try:
            result = self.result_queue.get_nowait() 
            tk.Label(fr_result,text="zernike a: ",bg="#EEA9B8").grid(row=0,column=0)
            tk.Label(fr_result,text=str(result.beforedemod['a0'])).grid(row=1,column=0)
            tk.Label(fr_result,text=str(result.beforedemod['a1'])).grid(row=2,column=0)
            tk.Label(fr_result,text=str(result.beforedemod['a2'])).grid(row=3,column=0)
            tk.Label(fr_result,text=str(result.beforedemod['a3'])).grid(row=4,column=0)
            tk.Label(fr_result,text=str(result.beforedemod['a4'])).grid(row=5,column=0)
            tk.Label(fr_result,text=str(result.beforedemod['a5'])).grid(row=6,column=0)

            tk.Label(fr_result,text="zernike b: ",bg="#EEA9B8").grid(row=0,column=1)
            tk.Label(fr_result,text=str(result.beforedemod['b3'])).grid(row=1,column=1)
            tk.Label(fr_result,text=str(result.beforedemod['b4'])).grid(row=2,column=1)
            tk.Label(fr_result,text=str(result.beforedemod['b5'])).grid(row=3,column=1)

            tk.Label(fr_result,text="Mx,My: ",bg="#EEA9B8").grid(row=0,column=2)
            tk.Label(fr_result,text=str(result.Mx)).grid(row=1,column=2)
            tk.Label(fr_result,text=str(result.My)).grid(row=2,column=2)

            tk.Label(fr_result,text="sph,cyl,axis: ",bg="#EEA9B8").grid(row=0,column=3)
            tk.Label(fr_result,text=str(result.sph)).grid(row=1,column=3)
            tk.Label(fr_result,text=str(result.cyl)).grid(row=2,column=3)
            tk.Label(fr_result,text=str(result.axis)).grid(row=3,column=3)

            tk.Button(fr_result,text="保存",command=self.getinputparameters,width=10).grid(row=5,column=1,columnspan=3)
        except self.result_queue.empty():
            pass
        # 30ms 约 33 fps



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "扫描式波前像差系统")
    root.mainloop()