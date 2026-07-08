import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from math import sqrt
from scipy.spatial import cKDTree
import math
import os
import optometry as opt
from imageproc import lvpyfun
import configpara

capture = cv2.VideoCapture(0, cv2.CAP_ANY)  # 打开内置摄像头

target_width = 1600  # 目标图像宽度
target_height = 1200  # 目标图像高度

# 设置摄像头的分辨率
capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

if not capture.isOpened():
    raise ValueError("摄像头错误")

while capture.isOpened():  # 摄像头被打开   
    para = configpara.configparameters()
    retval, image = capture.read()
    if retval == True:
        cv2.imshow("ori", image)
        cv2.imwrite('C:\\Users\\Dell\\Desktop\\ori\\ori.bmp', image)
        r = lvpyfun("back_1600_1200_20240731.bmp", image, 1.637 , 13.11878520128891, 3.45, "DMM1600_1200交大校准完成后的数据.txt", 4)  # 画圆半径单位毫米，透镜阵列焦距单位毫米，一个像素几微米,标准点坐标
        
        print(f"phy: {para.angle},direction: {para.direction}")
        print(f"修正前：z(0,0): {r[0]:.3f}, z(1,-1): {r[1]:.3f}, z(1,1): {r[2]:.3f}, z(2,-2): {r[3]:.3f}, z(2,0): {r[4]:.3f}, z(2,2): {r[5]:.3f}")
        print(f"修正前：z(3,-3): {r[6]:.3f}, z(3,-1): {r[7]:.3f}, z(3,1): {r[8]:.3f}, z(3,-3): {r[9]:.3f}")
        print(f"修正前：z(4,-4): {r[10]:.3f}, z(4,-2): {r[11]:.3f}, z(4,0): {r[12]:.3f}, z(4,2): {r[13]:.3f}, z(4,4): {r[14]:.3f}")
        #修正模式
        #opt.demodulation(r,para)
 
    key = cv2.waitKey(1000)
    if key == 32:
        break
capture.release()
cv2.destroyAllWindows() 