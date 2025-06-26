

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from math import sqrt
from scipy.spatial import cKDTree
import math
import os
import HSscan.optometry as opo

#外推法代码定义
def find_adjacent_points(point, distance=87):   #给入一个标定的点  distance为标定点间距
    """
    寻找给定点上下左右特定距离的点。
    """
    x, y = point
    return [[x, y - distance], [x, y + distance], [x - distance, y], [x + distance, y]]

def calculate_shift(measurement, calibration):       #测量点与标定点的偏移量
    """
    计算测量点和标定点之间的偏移。
    """
    return measurement[0] - calibration[0], measurement[1] - calibration[1]

def find_closest_point(guessed_point, points):#猜测点和测量点，一个猜测点与所有测量点对比
    """
    在提供的点列表中找到最接近猜测点的点。
    """
    closest_point = None
    min_distance =  70
    for point in points: #测量点列表
        distance = math.sqrt((guessed_point[0] - point[0])**2 + (guessed_point[1] - point[1])**2) #测量点与猜测点的距离
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point     #返回一个最近的测量点

def process_points(start_measurement, start_calibration,calibration_points,measurement_points,mapping):
    """
    根据起始测量点和起始标定点处理并更新映射关系。
    """
    # 计算起始测量点与起始标定点之间的偏移量
    x_shift, y_shift = calculate_shift(start_measurement, start_calibration)
    # 寻找起始标定点四周（上、下、左、右）的相邻标定点
    adjacent_calibrations = find_adjacent_points(start_calibration)

    # 用于存储更新后的相邻标定点(真实标定点)
    updated_adjacent_calibrations = []

    #让相邻的四个理论标定点与标定点列表中所有点进行匹配，找出真实的相邻标定点
    # 检查每个理论相邻点与calibration_points中的点的距离
    for adj_point in adjacent_calibrations:
        for cal_point in calibration_points:
            if distance(adj_point, cal_point) <= 0.5:                  #理论点与实际点距离
                # 如果距离小于等于0.1，则加入calibration_points中的点
                updated_adjacent_calibrations.append(cal_point)
                break

    #遍历每一个相邻标定点
    for calib_point in updated_adjacent_calibrations:
        # 检查相邻标定点是否仍在标定点列表中，这个if防止updated_adjacent_calibrations为空从的报错
        if calib_point in calibration_points:
            # 计算根据偏移量预测的测量点位置
            guessed_point = [calib_point[0] + x_shift, calib_point[1] + y_shift]
            # 在所有测量点中找到与预测点最接近的实际测量点
            closest_measurement = find_closest_point(guessed_point, measurement_points)
            # 如果找到最接近点，并且该测量点尚未被添加到映射中
            if closest_measurement and tuple(closest_measurement) not in mapping:
                # 建立测量点与标定点之间的映射关系
                mapping[tuple(closest_measurement)] = tuple(calib_point)
                # 从测量点列表和标定点列表中移除已经匹配的点
                measurement_points.remove(closest_measurement)
                calibration_points.remove(calib_point)
                # 递归调用函数，以当前匹配的测量点和标定点作为新的起始点继续匹配过程
                process_points(closest_measurement, calib_point,calibration_points,measurement_points,mapping)

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


#以上为外推法代码定义

#在两个列表中找到最近的坐标
def calculate_distance(point1, point2):
    """计算两点之间的欧式距离"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_closest_list_point(list1, list2):
    """找到两个列表中距离最近的两个点"""
    min_distance = float('inf')
    closest_points = [(0,0),(0,0)]

    for point1 in list1:
        for point2 in list2:
            distance = calculate_distance(point1, point2)
            if distance < min_distance:
                min_distance = distance
                closest_points = (point1, point2)

    return closest_points, min_distance


#每个子孔径采用自适应阈值
def adaptive_threshold(image, subaperture_centers, aperture_size):
    """
    对于每个子孔径应用自适应阈值。

    :param image: 输入包含光斑的图像。
    :param subaperture_centers: 子孔径中心点的坐标列表，每个元素是一个元组，代表 (x, y)。
    :param aperture_size: 子孔径的大小，表示为 (width, height)。
    :param n: 用作阈值的灰度值的索引。
    :return: 字典，键为子孔径索引，值为阈值化光斑信息。
    """
    # 创建一个与原始图像同样大小的空白图像
    final_image = np.zeros_like(image)

    for i, (x_center, y_center) in enumerate(subaperture_centers):
        # 计算子孔径区域的左上角坐标
        w, h = aperture_size
        x =round(x_center - w // 2)
        y =round(y_center - h // 2)
        #提取子孔径，增加判断越界如何处理
        if x<0:
            w=w+x
            x=0
        if y<0:
            h=h+y
            y=0

        sub_img = image[y:y+h, x:x+w]

        threshold_value, dst0 = cv2.threshold(sub_img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)  # 使用Otsu的方法对图像进行自适应阈值


        # 应用阈值
        # 测试
        if threshold_value>99:
            threshold_value=50
            _, tozero_sub_img = cv2.threshold(sub_img, threshold_value, 255, cv2.THRESH_TOZERO)
        else:
            _, tozero_sub_img = cv2.threshold(sub_img, threshold_value, 255, cv2.THRESH_TOZERO)

        # print(f"第{i + 1}个子孔径当前选用的阈值为", threshold_value)
        # 减去当前阈值
        # 将 threshold_value 转换为与 tozero_sub_img 相同的数据类型
        threshold_value_converted = np.array([threshold_value], dtype=tozero_sub_img.dtype)

        # 创建一个与 tozero_sub_img 尺寸和类型都相同的数组
        threshold_array = np.full(tozero_sub_img.shape, threshold_value_converted, dtype=tozero_sub_img.dtype)

        # 应用减法操作
        new_sub_img = cv2.subtract(tozero_sub_img, threshold_array)

        # 将二值化后的图像填充到最终图像的相应位置
        final_image[y:y + h, x:x + w] = new_sub_img


    return final_image


#图片相减+图像预处理，用于插分消背景
def fft_sub_open(image1_filepath,image2_filepath):

    image1 = cv2.imread(image1_filepath,cv2.IMREAD_GRAYSCALE)
    image3 = image2_filepath
    image2 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
    # 图像相减
    subtracted_image = cv2.subtract(image2, image1)
    # cv2.imwrite("988.4A.bmp", subtracted_image)

    mid_image = cv2.medianBlur(subtracted_image, 3)                               # 使用中值滤波去除噪点
    k = np.ones((3, 3), np.uint8)                                                 # 创建3X3的数组作为核
    open = cv2.morphologyEx(mid_image, cv2.MORPH_OPEN, k, iterations=1)           # 开运算
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, k, iterations=1)              # 闭运算
    # cv2.imshow("close",close)
    # cv2.imwrite("result.bmp",close)
    return close

def sph_cyl_axis(fit_list, r, X):
    r = r * X / 1000
    M = (-4 * sqrt(3) * fit_list[4] + 12 * sqrt(5) * fit_list[12]) / r ** 2
    J_0 = (-4 * sqrt(6) * fit_list[5] + 12 * sqrt(10) * fit_list[13]) / r ** 2
    J_45 = (-4 * sqrt(6) * fit_list[3] + 12 * sqrt(10) * fit_list[11]) / r ** 2
    cyl = -1 * sqrt(J_0 ** 2 + J_45 ** 2)
    sph = M - cyl / 2
    axis = math.degrees(0.5 * math.atan(fit_list[3] / fit_list[5]))
    if J_0 < 0:
        axis = axis + 90
    elif J_0 > 0 and J_45 < 0:
        axis = axis + 180
    elif J_0 > 0 and J_45 < 0:
        axis = axis
    return sph, cyl, axis

#寻找位于图像中心的点
def find_spot_centers(image_path):
    # 读取图像
    image = image_path          #进来的图片就已经是二值化图像了

    # 寻找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化一个列表用于存储光斑点阵的中心坐标
    spot_centers = []
    sorted_centroids = []
    if len(contours)>0:
        # 遍历每个轮廓
        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            # 通过设置一个阈值来排除小面积的轮廓，以排除噪音
            if area > 10:
                # 计算轮廓的矩
                M = cv2.moments(contour)

                # 计算中心坐标
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    spot_centers.append((center_x, center_y))



        # 按y坐标排序
        sorted_by_y = sorted(spot_centers, key=lambda x: x[1])  # 对centroid_list中的元素的第二个值作为依据进行排序

        # 将具有相似y坐标的质心分组
        groups = []
        group = []
        prev_centroid = sorted_by_y[0]                     # 存放的是相似坐标中最小的y

        for centroid in sorted_by_y:
            if abs(centroid[1] - prev_centroid[1]) < 20:   # 把列表中的每一个y元素与转折最小值进行比较
                group.append(centroid)                     # 当坐标相减的绝对值小于一个数，那么为相似存入group中
            else:
                groups.append(group)                       # 当if不成立的时候，则y的最小值发生了转折,把之前全部相似的(x,y)坐标导入groups
                group = [centroid]                         # 把这个新的(x,y)用来初始化group，此时group中只包含这个新的(x,y)
            prev_centroid = centroid                       # 把这个新的值，重新付给prev_centroid作为相似的判断依据
        groups.append(group)                               # 因为最后的值判断完毕后无法进入else的groups.append(group)

                                                               # 所以需要另外传递


        # 对每个组按照x坐标进行排序，并将它们连接在一起
        list_cc=[]                #初始化方便查看矩阵的列表
        for group in groups:  # 需要注意的是groups中存放的是一个相似y的(x,y)坐标列表
            sorted_centroids.extend(sorted(group, key=lambda x: x[0]))  # 把group中的相似y按照x重新排序。
            list_cc.append(sorted(group, key=lambda x: x[0]))


        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for idx, centroid in enumerate(sorted_centroids, 1):
            # 绘制质心和编号
            cv2.putText(color, str(idx), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)  # 对不同质心的坐标进行标号

        row_spot=len(list_cc)//2                      #行

        col_spot=len(list_cc[row_spot])//2                 #列
        spot_center=list_cc[row_spot][col_spot]

        return spot_center
    else:
        return 0

# 定义一个函数，创建一个大的圆形mask
def create_large_circular_mask(image, radius,center_spot):
    center = (int(image.shape[1] / 2), int(image.shape[0] / 2))  # shape[1]横坐标列。shape[2]纵坐标行
    if center_spot!=0:
        center=center_spot
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))  # 返回二维数组的横坐标和纵坐标存放在x，y中
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)  # 每个像素到中心的距离
    mask = distance <= radius                                         #当每个像素坐标到中心距离小于半径返回真给mask
    image_circular = np.where(mask, image, 0)  # mask为判断条件，里面为真的值执行image的原像素值，除了真以外的像素值全部置为0

    return image_circular,center            #返回遮掩后的图像，返回中心坐标

#计算质心坐标
def calculate(img,radius):                                                   #H为True时，进行可调区域计算，否则正常计算
    img0 = img.copy()                                                             # 原图
    img1 = img.copy()                                                             # 标号图
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)                                 # 变成彩色图像
    img2 = img.copy()                                                             # 原图画圈
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)  # 变成彩色图像
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                  # 原图从彩图变成单通道灰度图像


    # center_spot=find_spot_centers(img)                        #img为滤波后的图片,这个代码以后重写可能与276行代码有冲突
    dst1, center = create_large_circular_mask(img, radius,0)  # 输出可调区域的光斑图像 另center_spot等于0则采用图片中心作为圆心

    cv2.circle(img, center, radius,  (255,0,0), 2)                             # 框框图画圈
    cv2.circle(img2, center, radius, (255, 0, 0), 2)                          # 在原图上画圈

    # 二值化
    t10, bin_img = cv2.threshold(dst1, 0, 255, cv2.THRESH_BINARY)       # dst1的基础上二值化
    # cv2.imshow("bin",bin_img)

    # 用作绘图
    cic_dst1 = dst1.copy()
    cic_dst1 = cv2.cvtColor(cic_dst1, cv2.COLOR_GRAY2BGR)
    cv2.circle(cic_dst1, center, radius, (255, 0, 0), 2)  # 在原图上画圈
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓坐标信息


    Mid_Area=0
    valid_contours = []  # 创建一个新的函数，存储有效轮廓坐标信息
    sorted_centroids = []
    s = 0  # 光斑的总数初始值为0
    if len(contours) > 0:
        Area_contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)  # 对新的函数从大到小进行面积排序
        middle_index = len(Area_contours_sorted) // 2  # 取中间的标号
        Mid_Area = cv2.contourArea(Area_contours_sorted[middle_index])

        for i, number in enumerate(contours, 1):
            if cv2.contourArea(number) < Mid_Area * (1/3) or cv2.contourArea(number)>Mid_Area*2.5:  #比较与中间的光斑面积大小做出判别
                continue
            valid_contours.append(number)  # 把number里面的坐标存入新的函数中
        #        sum=sum+1                                           # 记录有效光斑总数
        # 增加图像的判别，先对图像进行筛选，对有效的光斑按照面积从大到小进行排序，如果任意一个光斑的面积大于排在中间光斑的3倍则图片是无效的
        # for i, number in enumerate(Area_contours_sorted):  # 遍历没有经过筛选的面积列表
        #     if cv2.contourArea(number) > cv2.contourArea(Area_contours_sorted[middle_index]) * 3:  # 大小对比
        #         print('所取的图像无效,请重新采集')
        #         #exit()
        # 初始化光斑质心列表

        centroid_list = []

        # 获取图像中心位置

        center_x = center[0]
        center_y = center[1]

        #计算中间光斑的半径

        r = np.sqrt(Mid_Area / np.pi)

        # 质心的计算坐标的过程
        for num, contour in enumerate(valid_contours, 1):  # num表示下标 contours数组当括号内数字为1时第一个下标为1否则默认从0开始
            x, y, w, h = cv2.boundingRect(contour)         # contour表示contours里面单个图像的坐标的存放数据
            if w > h*2.5 or h > w*2.5:                     # 限制光斑长宽比例，以有效去除杂光
                continue
            sum_1 = 0
            sum_2 = 0
            sum_3 = 0
            s = s + 1                                      # 当进行一个光斑轮廓的坐标赋值时，光斑的数量+1
            for i in range(y, y + h):                      # i表示纵坐标
                for j in range(x, x + w):                  # j表示横坐标
                    sum_1 = sum_1 + j * dst1[i, j]         # 计算x坐标的分子
                    sum_2 = sum_2 + i * dst1[i, j]         # 计算Y坐标的分子
                    sum_3 = sum_3 + dst1[i, j]             # 分母是一样的
            sum_x = (sum_1 / sum_3)                     # 计算X的横坐标
            sum_y = (sum_2 / sum_3)                     # 计算Y的纵坐标

            distance = abs(math.sqrt((sum_x - center_x) ** 2 + (sum_y - center_y) ** 2))
            if (radius - distance) <= r :          #大圆半径-光斑与中心的距离与中间光斑半径进行比较
                continue
            cv2.rectangle(cic_dst1, (x, y), (x + w, y + h), (0, 0, 255), 1)         # 绘制矩形框框
            centroid_list.append((sum_x, sum_y))                               # 把质心的坐标依次放入一个新的列表中其中(x,y)为一个元素

        #阈值与总数
        # print(f"当前图片的光斑总数：{s} 当前阈值为：{t0}")

        if len(centroid_list)>0:

            # 按y坐标排序
            sorted_by_y = sorted(centroid_list, key=lambda x: x[1])  # 对centroid_list中的元素的第二个值作为依据进行排序

            # 将具有相似y坐标的质心分组
            groups = []
            group = []
            prev_centroid = sorted_by_y[0]                     # 存放的是相似坐标中最小的y

            for centroid in sorted_by_y:
                if abs(centroid[1] - prev_centroid[1]) < 20:   # 把列表中的每一个y元素与转折最小值进行比较
                    group.append(centroid)                     # 当坐标相减的绝对值小于一个数，那么为相似存入group中
                else:
                    groups.append(group)                       # 当if不成立的时候，则y的最小值发生了转折,把之前全部相似的(x,y)坐标导入groups
                    group = [centroid]                         # 把这个新的(x,y)用来初始化group，此时group中只包含这个新的(x,y)
                prev_centroid = centroid                       # 把这个新的值，重新付给prev_centroid作为相似的判断依据
            groups.append(group)                               # 因为最后的值判断完毕后无法进入else的groups.append(group)

                                                               # 所以需要另外传递


            # 对每个组按照x坐标进行排序，并将它们连接在一起
            # list_cc=[]                #初始化方便查看矩阵的列表
            for group in groups:  # 需要注意的是groups中存放的是一个相似y的(x,y)坐标列表
                sorted_centroids.extend(sorted(group, key=lambda x: x[0]))  # 把group中的相似y按照x重新排序。
                # list_cc.append(sorted(group, key=lambda x: x[0]))           # 方便查看的列表
            #遍历方便查看的列表
            # print(list_cc)               #查看整个列表，方便复制
            # for cc in list_cc:
            #     print(cc)

            # 遍历排序后的质心
            for idx, centroid in enumerate(sorted_centroids, 1):
                # 绘制质心和编号
                cv2.putText(img1, str(idx), (int(centroid[0]), int(centroid[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)     # 对不同质心的坐标进行标号
                # canvas = cv2.line(cic_dst1, (int(centroid[0]) - 2, int(centroid[1])), (int(centroid[0]) + 2, int(centroid[1])), (255, 0, 0),
                #                   1)  # 质心标记的横线
                # canvas = cv2.line(cic_dst1, (int(centroid[0]), int(centroid[1]) - 2), (int(centroid[0]), int(centroid[1]) + 2), (255, 0, 0),
                #                   1)  # 质心标记的竖线
            # 输出光斑质心坐标（idx + 1）

            # for i, centroid in enumerate(sorted_centroids, 1):
            #     print(f"光斑{i} ({centroid[0]}, {centroid[1]})",end="")
            # print("")
            #
            flag=True
        else:
            flag=False
    else:
        flag = False
    return flag, sorted_centroids, img0, cic_dst1, img1, img2, s,center  # flag,返回质心坐标列表、原图、框框图、序号图、画圆圈的原图、光斑总数、自适应阈值
#计算D矩阵
def compute_D_matrix(coordinates,n):            #定义4阶X,Y方向上的zernike展开
    matrix = []
    for x, y in coordinates:                  #X方向
        if n == 0:
            matrix.append([
                1 * 0
            ])
        elif n == 1:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1
            ])
        elif n == 2:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1,
                sqrt(6) * 2 * y,
                sqrt(3) * 4 * x,
                sqrt(6) * 2 * x
            ])
        elif n == 3:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1,
                sqrt(6) * 2 * y,
                sqrt(3) * 4 * x,
                sqrt(6) * 2 * x,
                sqrt(8) * 6 * x * y,
                sqrt(8) * 6 * x * y,
                sqrt(8) * (-2 + 9 * x ** 2 + 3 * y ** 2),
                sqrt(8) * (3 * x ** 2 - 3 * y ** 2)
            ])
        elif n == 4:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1,   #1
                sqrt(6) * 2 * y,
                sqrt(3) * 4 * x,
                sqrt(6) * 2 * x,
                sqrt(8) * 6 * x * y,
                sqrt(8) * 6 * x * y,
                sqrt(8) * (-2 + 9 * x ** 2 + 3 * y ** 2),
                sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                sqrt(10) * (12 * x ** 2 * y - 4 * y ** 3),
                sqrt(10) * (-6 * y + 24 * x ** 2 * y + 8 * y ** 3),
                sqrt(5) * (-12 * x + 24 * x ** 3 + 24 * x * y ** 2),
                sqrt(10) * (-6 * x + 16 * x ** 3),
                sqrt(10) * (4 * x ** 3 - 12 * x * y ** 2)
            ])
    for x, y in coordinates:              #Y方向
        if n == 0:                        #1
            matrix.append([
                1 * 0
            ])
        elif n == 1:                      #3
            matrix.append([
                1 * 0,
                2 * 1,
                2 * 0
            ])
        elif n == 2:                      #6
            matrix.append([
                1 * 0,
                2 * 1,
                2 * 0,
                sqrt(6) * 2 * x,
                sqrt(3) * 4 * y,
                sqrt(6) * -2 * y
            ])
        elif n == 3:                      #10
            matrix.append([
                1 * 0,
                2 * 1,
                2 * 0,
                sqrt(6) * 2 * x,
                sqrt(3) * 4 * y,
                sqrt(6) * -2 * y,
                sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                sqrt(8) * (-2 + 3 * x ** 2 + 9 * y ** 2),
                sqrt(8) * 6 * x * y,
                sqrt(8) * -6 * x * y
            ])
        elif n == 4:                       #15
            matrix.append([
                1 * 0,
                2 * 1,    #1
                2 * 0,
                sqrt(6) * 2 * x,
                sqrt(3) * 4 * y,
                sqrt(6) * -2 * y,
                sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                sqrt(8) * (-2 + 3 * x ** 2 + 9 * y ** 2),
                sqrt(8) * 6 * x * y,
                sqrt(8) * -6 * x * y,
                sqrt(10) * (4 * x ** 3 - 12 * x * y ** 2),
                sqrt(10) * (-6 * x + 8 * x ** 3 + 24 * x * y ** 2),
                sqrt(5) * (-12 * y + 24 * x ** 2 * y + 24 * y ** 3),
                sqrt(10) * (6 * y - 16 * y ** 3),
                sqrt(10) * (-12 * x ** 2 * y + 4 * y ** 3)
            ])
    return np.array(matrix)       #把一行的列表导入数组

def compute_C(D, J):             #最小二乘法，广义逆矩阵计算
    D_plus = np.linalg.pinv(D)   # D的广义逆矩阵
    C = np.dot(D_plus, J)
    return C

def rms(c):                      #误差RMS计算
    s=0
    for i in c:
        s = s + i ** 2
    s=sqrt(s)
    return s


def getcentraldiffs(non_standard_spots, list_standard_spots):



    # 计算偏移量并输出偏移量的结果
    x_diffs = []
    y_diffs = []

    # 匹配
    for (x1, y1), (x2, y2) in zip(non_standard_spots, list_standard_spots):
        x_diff = x1 - x2
        y_diff = y1 - y2
        x_diffs.append(x_diff)  # 存放x的偏移量
        y_diffs.append(y_diff)  # 存放y的偏移量

    # 显示X与Y方向的平均偏移距离
    # average_XY=sum(sum_distance)/len(list_standard_spots)
    # average_X=sum(x_diffs)/len(list_standard_spots)
    # average_Y=sum(y_diffs)/len(list_standard_spots)
    # print("XY方向光斑平均偏移量",average_XY)
    # print("X方向光斑平均偏移量",average_X)
    # print("Y方向光斑平均偏移量",average_Y)

    # x与y的偏移量合并
    xy_diffs = []
    xy_diffs.extend(x_diffs)
    xy_diffs.extend(y_diffs)



    return xy_diffs,list_standard_spots

#计算平均斜率
def averageslope(xy_diffs,radius,F,X):
    j=[]
    for i in xy_diffs:
        x=i*X
        f=F*1000
        r=radius*X
        j.extend([(x/f)*r])
    # J矩阵
    J = np.array(j)
    return J

def getstandard_spots(list_standard_spots,radius,n,center):
    # #定义归一化后的R列表
    R = []
    # 计算图像中心坐标
    zx = center
    for central in list_standard_spots:          #遍历与非标准点对应的标准点坐标
        x = (central[0]-zx[0])/radius
        y = (central[1]-zx[1])/radius
        R.append([x,y])


    coordinates = R

    # 计算D矩阵
    D = compute_D_matrix(coordinates,n)
    return D

def read_function(file_path):


    # 打开文本文件
    with open(file_path, 'r') as file:
        # 读取文件内容
        content = file.read()
    # 使用eval函数将文本内容转换为列表
    central = eval(content)

    return central

def canvas(list_standard,list_nostandard):

    # 创建一个白色画布
    canvas = np.ones((1200, 1600,3), dtype=np.uint8) * 255
    cv2.circle(canvas, (800, 600), 5, (0, 255, 0), -1)  # 标记坐标
    # 在画布上标记坐标并画正方形矩阵
    square_size = 87  # 正方形矩阵的边长

    for point in list_standard:
        x, y = round(point[0]), round(point[1])
        cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)  # 标记坐标
        # 计算正方形矩阵的边界框
        x1, y1 = x - square_size // 2, y - square_size // 2
        x2, y2 = x + square_size // 2, y + square_size // 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 画正方形矩阵

    for point in list_nostandard:
        x, y = round(point[0]), round(point[1])
        cv2.circle(canvas, (x, y), 3, (255, 0, 0), -1)  # 标记坐标
    # 显示画布
    # cv2.imshow("Canvas", canvas)
    return canvas

# 定义子函数用于绘制连线
def draw_lines(canvas, list_standard, list_nostandard):
    for (point_std, point_nstd) in zip(list_standard, list_nostandard):
        x_std, y_std = round(point_std[0]), round(point_std[1])
        x_nstd, y_nstd = round(point_nstd[0]), round(point_nstd[1])
        cv2.line(canvas, (x_std, y_std), (x_nstd, y_nstd), (0, 0, 0), 1)  # 绘制连线
    cv2.imshow("Canvas_line", canvas)


def rotate_point(point, angle_rad):
    # angle_rad = np.radians(angle_degrees)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_point = np.dot(rotation_matrix, point)
    return rotated_point.tolist()  # 将结果转换为列表

def zernike(r, theta, Zernike_coefficient):
    expressions = [
        np.ones_like(r),
        2*r*np.sin(theta),
        2*r*np.cos(theta),
        np.sqrt(6)*r**2*np.sin(2*theta),
        np.sqrt(3)*(2*r**2-1),
        np.sqrt(6)*r**2*np.cos(2*theta),
        np.sqrt(8)*r**3*np.sin(3*theta),
        np.sqrt(8)*(3*r**3-2*r)*np.sin(theta),
        np.sqrt(8)*(3*r**3-2*r)*np.cos(theta),
        np.sqrt(8)*r**3*np.cos(3*theta),
        np.sqrt(10)*r**4*np.sin(4*theta),
        np.sqrt(10)*(4*r**4-3*r**2)*np.sin(2*theta),
        np.sqrt(5)*(6*r**4-6*r**2+1),
        np.sqrt(10)*(4*r**4-3*r**2)*np.cos(2*theta),
        np.sqrt(10)*r**4*np.cos(4*theta)
    ]

    z_fit = np.zeros_like(r)
    for i, c in enumerate(Zernike_coefficient):
        z_fit += c * expressions[i]

    return z_fit

def calculate_pv_rms(z_fit,coefficient):
    coefficients_np=np.array(coefficient)
    z_max=np.max(z_fit)
    z_min=np.min(z_fit)
    pv = z_max-z_min
    coefficients_np[4]=0
    rms = np.sqrt(np.sum(coefficients_np**2))
    return pv, rms,z_max,z_min


def plot_zernike_wavefront(Zernike_coefficient):
    #倾斜项置0
    coefficient=Zernike_coefficient
    coefficient[1]=0
    coefficient[2]=0

    theta = np.linspace(0, 2.*np.pi, 100)
    r = np.linspace(0, 1, 100)
    r, theta = np.meshgrid(r, theta)

    # 转化为笛卡尔坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    Z = zernike(r, theta, coefficient)

    # fig = plt.figure(figsize=(12, 5))
    #
    # # 3D plot
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot_surface(x, y, Z, cmap='jet',rstride=2, cstride=2, alpha=1)
    # ax1.set_title("3D Wavefront")
    #
    # # 2D plot
    # ax2 = fig.add_subplot(122)
    # c = ax2.contourf(x, y, Z, cmap='jet', levels=100)  # 使用levels参数增加颜色层级
    # ax2.set_title("2D Wavefront")
    #
    # #Add colorbar for the 2D plot
    # fig.colorbar(c, ax=ax2, orientation='vertical')
    #
    # plt.tight_layout()

    pv, rms,z_max,z_min = calculate_pv_rms(Z,coefficient)
    print("PV value:", pv)
    print("RMS value:", rms/0.85)
    print("z_max and z_min :", z_max,z_min)
    plt.show()
    return 0

#过滤圆外的标准点
def filter_points_inside_circle(points, center, radius):
    """
    过滤掉在圆外的点

    :param points: 点坐标列表，每个元素格式为(x, y)
    :param center: 圆心坐标，格式为(x, y)
    :param radius: 圆的半径
    :return: 在圆内的点列表
    """
    # 在圆内的点列表
    points_inside_circle = []

    # 遍历所有点，判断每个点是否在圆内
    for point in points:
        # 计算点到圆心的距离
        distance = ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2) ** 0.5
        # 如果距离小于等于半径，则点在圆内
        if distance <= radius:
            points_inside_circle.append(point)

    return points_inside_circle

#查看单张图片的质心以及图像
# image0=cv2.imread("1D.bmp")
# flag,central0,img00,img01,img02,s0,t0=calculate(image0,256)                 #True时进行可调区域计算，否则正常计算
# cv2.imshow("img00",img00)                                                    #原图
# cv2.imshow("img01",img01)                                                    #序号
# cv2.imshow("img02",img02)                                                    #框框
# print("坐标",central0)
# cv2.waitKey()                                                                # 按下任何键盘按键后
# cv2.destroyAllWindows()                                                      # 释放所有窗体
# exit()


#                                          主程序
#***************************************************************************************************


def lvpyfun(image1_filepath,image2_filepath,radius,F,X,standard_spots_filepath,n):       #图片的地址，画圆半径单位毫米，透镜阵列焦距单位毫米，一个像素几微米，标准点坐标

    #对图片进行滤波处理，图片2-1
    filtering=fft_sub_open(image1_filepath,image2_filepath)
    # color_image = cv2.cvtColor(filtering, cv2.COLOR_GRAY2BGR)
    # # 读入图片的地址
    # image1 = cv2.imread(image_filepath)               #偏移的图像
    # 把毫米单位半径转化为像素单位半径
    radius = round(radius * 1000 / X)
    # 读入标准点坐标的的内容

    central2=read_function(standard_spots_filepath)
    #每个窗口采用不同阈值
    subaperture_centers=central2
    final_image=adaptive_threshold(filtering, subaperture_centers, (87,87))
    # cv2.imshow("final_image",final_image)

    #去底噪
    final_image= cv2.subtract(final_image, 5)                         # 去除底噪

    #给入一个图片，将会计算这个图片的质心坐标，和返回质心坐标的列表,TRUE为可调区域的选取，False为不选取可调区域
    flag,central1,img10,img11,img12,img13,s1,center=calculate(final_image,radius)       #central1为质心列表.10为原图，11为框框图.12为标号图
    # cv2.imshow("12",img12)
    # cv2.imshow("11",img11)
    #换砖坐标
    # 原始坐标数据
    coordinates = central1
    # 以左上角为原点旋转坐标逆时针旋转
    rotated_coordinates = [rotate_point(point, 0.0011758743062984633) for point in coordinates]
    # print(rotated_coordinates)
    #提取园内标准点
    standard_incircle=filter_points_inside_circle(central2,center,radius)

    #画布动态范围可视化
    canvas_picture=canvas(standard_incircle, rotated_coordinates) #(标准，不标准)
    #外推法

    # 映射关系字典
    mapping = {}
    #放入初始点
    # initial_measurement = [988.6003917011782, 437.3625982440384]
    # initial_calibration = [993.4607959828131, 435.4881940520131]

    ori_points,dis=find_closest_list_point(rotated_coordinates,standard_incircle)#不标准，标准

    initial_measurement = ori_points[0]
    initial_calibration = ori_points[1]
    #通过下方函数，输出映射关系
    process_points(initial_measurement, initial_calibration, standard_incircle, rotated_coordinates, mapping)



    standard_points = []
    no_standard_points_ratated = []
    # Print the mapping
    # print("测量点与标定点的对应关系：")
    for m_point, c_point in mapping.items():
        standard_points.append(c_point)
        no_standard_points_ratated.append(m_point)
        # print(f"测量点 {m_point} 对应 标定点 {c_point}")

    # #调用连线函数和画初始点
    x_ori=initial_calibration[0]
    y_ori=initial_calibration[1]
    cv2.circle(canvas_picture, (round(x_ori),round(y_ori)), 5, (255, 0, 255), -1)  # 标记坐标
    draw_lines(canvas_picture, standard_points, no_standard_points_ratated)


    if len(standard_points)==0 or no_standard_points_ratated==0:
        flag=False


    if flag:
        #保存框框图片地址
        cv2.imwrite('C:\\Users\\Administrator\\Desktop\\123\\low.bmp', img11)
        # # 显示框框图
        cv2.imshow("10",img10)
        cv2.imshow("img11",img11)
        #给入可调区域的偏移坐标central1与完整标准点的central2
        xy_diffs,list_standard_spots=getcentraldiffs(no_standard_points_ratated,standard_points)
        # print(xy_diffs)
        #给如偏移量，计算平均斜率
        J=averageslope(xy_diffs,radius,F,X)
        #给如标准点坐标，计算D矩阵
        D=getstandard_spots(list_standard_spots,radius,n,center)
        #计算系数C与保留三位小数
        C=compute_C(D,J)
        rounded_coefficients = np.round(C, decimals=3)*(-1)
        r_list = list(rounded_coefficients)
        #减去本征
        r_list = opo.initaberration_correction(r_list)

        #计算PV
        # plot_zernike_wavefront(r_list)
        #计算球镜度，柱镜度数，轴向
        sph,cyl,axis=sph_cyl_axis(r_list,radius,X)
        r_list.append(sph)
        r_list.append(cyl)
        r_list.append(axis)

        return r_list
    else:
        return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# # 二三项置0计算PV完毕后 又把第5项置0 然后计算RMS
# r=lvpyfun("black_1600_1200.bmp","sph1016.bmp",1.5,13.11878520128891,3.45,"DMM1600_1200交大校准完成后的数据.txt",4)          #画圆半径单位毫米，透镜阵列焦距单位毫米，一个像素几微米,标准点坐标
#
#
# print(r)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# #process_images('C:\\Users\\Dell\\Desktop\\optotune230810\\data\\2023_8_13_16_36_54','C:\\Users\\Dell\\Desktop\\optotune230810\\data\\2023_8_13_16_36_54\\sdbk.txt','C:\\Users\\Dell\\Desktop\\optotune230810\\data\\2023_8_13_16_36_54_2')
capture = cv2.VideoCapture(0, cv2.CAP_ANY)  # 打开内置摄像头

target_width = 1600  # 目标图像宽度
target_height = 1200  # 目标图像高度

# 设置摄像头的分辨率
capture.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)



while capture.isOpened():  # 摄像头被打开
    retval, image = capture.read()
    if retval == True:
        cv2.imshow("ori", image)
        cv2.imwrite('C:\\Users\\Dell\\Desktop\\ori\\ori.bmp', image)
        r = lvpyfun("back_1600_1200_20240731.bmp", image, 1.637 , 13.11878520128891, 3.45, "DMM1600_1200交大校准完成后的数据.txt", 4)  # 画圆半径单位毫米，透镜阵列焦距单位毫米，一个像素几微米,标准点坐标
        print(r)
    key = cv2.waitKey(1000)
    if key == 32:
        break
capture.release()
cv2.destroyAllWindows()