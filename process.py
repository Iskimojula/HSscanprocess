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


while capture.isOpened():  # 摄像头被打开
    para = configpara.configparameters()
    retval, image = capture.read()
    if retval == True:
        cv2.imshow("ori", image)
        cv2.imwrite('C:\\Users\\Dell\\Desktop\\ori\\ori.bmp', image)
        r = lvpyfun("back_1600_1200_20240731.bmp", image, 1.637 , 13.11878520128891, 3.45, "DMM1600_1200交大校准完成后的数据.txt", 4)  # 画圆半径单位毫米，透镜阵列焦距单位毫米，一个像素几微米,标准点坐标
        opt.demodulation(r,para)

    key = cv2.waitKey(1000)
    if key == 32:
        break
capture.release()
cv2.destroyAllWindows()