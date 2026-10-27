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
import queue
def captureprocess(capture,frame_queue,result_queue,para_queue):
    #提示信息只在状态变化时打印一次，避免每 500ms 刷屏（旧版会不停打印 para_queue.empty！）
    last_note = None
    note = None
    while  True:
        if not para_queue.empty():
            para = para_queue.get_nowait()
            para_queue.put_nowait(para)
            retval, image = capture.read()
            if retval == True and para.checkvaild() == True:
                note = "采集运行中"
                #cv2.imshow("ori", image)
                #cv2.imwrite('C:\\Users\\Dell\\Desktop\\ori\\ori.bmp', image)
                r = lvpyfun("back_1600_1200_20240731.bmp", image, 1.637 , 13.11878520128891, 3.45, "DMM1600_1200交大校准完成后的数据.txt", 4)  # 画圆半径单位毫米，透镜阵列焦距单位毫米，一个像素几微米,标准点坐标
                
                print(f"phy: {para.angle},direction: {para.direction}")
                print(f"修正前：z(0,0): {r[0]:.3f}, z(1,-1): {r[1]:.3f}, z(1,1): {r[2]:.3f}, z(2,-2): {r[3]:.3f}, z(2,0): {r[4]:.3f}, z(2,2): {r[5]:.3f}")
                
                #修正模式
                results = opt.demodulation(r,para)

                #数据传输---传原图
                try:
                    frame_queue.put_nowait(image)
                except queue.Full:
                # 丢弃旧帧
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    frame_queue.put_nowait(image)

                #数据传输---传结果
                try:
                    result_queue.put_nowait(results)
                    print("transfer result_queue")
                except queue.Full:
                # 丢弃旧帧
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    result_queue.put_nowait(results)
            else:
                note = "未取到图像或参数无效（检查相机是否可用、入瞳/出瞳半径是否为 0）"
        else:
            note = "等待参数下发（点“确定”后开始采集）"
        if note != last_note:
            print("[采集] " + note)
            last_note = note
        cv2.waitKey(500)

