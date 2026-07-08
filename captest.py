import cv2
capture = cv2.VideoCapture(0)
if  capture.isOpened():
    print("摄像头已经打开")

else:
    print("摄像头未打开")
# 尝试 0, 1, 2 ... 直到找到为止