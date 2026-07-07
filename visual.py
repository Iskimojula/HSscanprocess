import configpara
from tkinter import *
import cv2
from PIL import Image, ImageTk
root = Tk()
root.title("扫描式HS波前像差仪")
root.geometry("500x800")
root.resizable(False,False)

##输入实验参数
ri_default = DoubleVar()
g_para = configpara.configparameters()

fr_configpara = LabelFrame(root,text="实验参数",relief="solid",bd = 2)
fr_configpara.pack(anchor="w")

lab_ent_ri1 = Label(fr_configpara,text="入瞳半径：").grid(row=0,column=0)
ri_default = DoubleVar(value=g_para.ri)
ent_ri = Entry(fr_configpara,textvariable=ri_default,width=5)
ent_ri.grid(row=0,column=1)
lab_ent_ri2 = Label(fr_configpara,text="mm").grid(row=0,column=2)

lab_ent_ro1 = Label(fr_configpara,text="出瞳半径：").grid(row=1,column=0)
ro_default = DoubleVar(value=g_para.ro)
ent_ro = Entry(fr_configpara,textvariable=ro_default,width=5)
ent_ro.grid(row=1,column=1)
lab_ent_ro2 = Label(fr_configpara,text="mm").grid(row=1,column=2)

lab_ent_theta1 = Label(fr_configpara,text="视场角theta：").grid(row=2,column=0)
theta_default = DoubleVar(value=g_para.theta)
ent_theta = Entry(fr_configpara,textvariable=theta_default,width=5)
ent_theta.grid(row=2,column=1)
lab_ent_theta2 = Label(fr_configpara,text="deg").grid(row=2,column=2)

lab_ent_angle1 = Label(fr_configpara,text="视场角angle：").grid(row=3,column=0)
angle_default = DoubleVar(value=g_para.angle)
ent_angle = Entry(fr_configpara,textvariable=angle_default,width=5)
ent_angle.grid(row=3,column=1)
lab_ent_angle2 = Label(fr_configpara,text="deg").grid(row=3,column=2)

lab_ent_inputD1 = Label(fr_configpara,text="试镜片：").grid(row=4,column=0)
inputD_default = DoubleVar(value=g_para.inputD)
ent_inputD = Entry(fr_configpara,textvariable=inputD_default,width=5)
ent_inputD.grid(row=4,column=1)
lab_ent_inputD2 = Label(fr_configpara,text="D").grid(row=4,column=2)

def getinputparameters():
    ri = float(ent_ri.get())
    ro = float(ent_ro.get())
    theta = float(ent_theta.get())
    angle = float(ent_angle.get())
    g_para.setconfigparameters(ri,ro,theta,angle)
    g_para.printpara()


Bt = Button(fr_configpara,text="确定",command=getinputparameters,width=10).grid(row=5,column=0,columnspan=3)

##显示每帧图片
fr_image = LabelFrame(root,text="点阵图片",relief="solid",bd = 2)
fr_image.pack(anchor="w")

dotimgfromcap = Image.open("sample.jpg")
dotimgfromcap = dotimgfromcap.resize((400,300),Image.Resampling.LANCZOS)

img = ImageTk.PhotoImage(dotimgfromcap)
Label(fr_image,image=img).pack()

##显示计算结果

fr_result = LabelFrame(root,text="调试结果",relief="solid",bd = 2)
fr_result.pack(anchor="w")
mainloop()