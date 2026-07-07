import math
import numpy as np
import yaml
from math import sin,cos
import configpara
def printversion():
    print()
    print("****** opo algo *******")
    print()
def zernike(n,m):
    return int((n*(n+2)+m)/2)

def get_angelstr(angle_num):
    if angle_num == 0:
        return 'init'
    if angle_num < 0:
        return 'minus'+ str(int(math.fabs(angle_num)))
    if angle_num > 0:
        return 'positive' + str(int(math.fabs(angle_num)))

def get_initaberration(dir,angle):
    with open("intrinsic_aberration.yaml","r",encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return list(map(float,config[dir][get_angelstr(angle)]))

def calc_M_J45_J180_noatchson(z_list,radius):
    M = -4*math.sqrt(3)*z_list[zernike(2,0)]*0.85
    J45 = -2*math.sqrt(6)*z_list[zernike(2,-2)]*0.85 #CX
    J180 = -2*math.sqrt(6)*z_list[zernike(2,2)]*0.85 #C+
    return M/(radius*radius),J45/(radius*radius),J180/(radius*radius)

def  calc_M_J45_J180(z3,z4,z5,radius):
    M = -4*math.sqrt(3)*z4*0.85
    J45 = -4*math.sqrt(6)*z3*0.85 #CX
    J180 = -4*math.sqrt(6)*z5*0.85 #C+
    return M/(radius*radius),J45/(radius*radius),J180/(radius*radius)

#修正公式,输入是角度,计算M,J45,J180
'''
def calc_M_J45_J180(z_list,radius,alpha,phy):
    alpha = math.radians(alpha)
    phy = math.radians(phy)

    M = -(2*math.sqrt(3)*z_list[zernike(2,0)]*(math.pow(math.cos(phy),2)+1) + 
          math.sqrt(6)*z_list[zernike(2,-2)]*math.sin(2*alpha)*math.pow(math.sin(phy),2) + 
          math.sqrt(6)*z_list[zernike(2,2)]*math.cos(2*alpha)*math.pow(math.sin(phy),2)
          )/math.pow((radius*math.cos(phy)),2)
    
    J45 = - (2*math.sqrt(3)*z_list[zernike(2,0)]*math.sin(2*alpha)*math.pow(math.sin(phy),2)+
             math.sqrt(6)*z_list[zernike(2,-2)]*(2*math.pow(math.cos(2*alpha),2)*math.cos(phy)+ math.pow(math.sin(2*alpha),2)*(1+math.pow(math.cos(phy),2)))+
             math.sqrt(6)*z_list[zernike(2,2)]*math.cos(2*alpha)*math.sin(2*alpha)*math.pow((1+math.cos(phy)),2)
             )/math.pow((radius*math.cos(phy)),2)
    
    J180 = - (2*math.sqrt(3)*z_list[zernike(2,0)]*math.cos(2*alpha)*math.pow(math.sin(phy),2)+
             math.sqrt(6)*z_list[zernike(2,-2)]*math.cos(2*alpha)*math.sin(2*alpha)*math.pow((1+math.cos(phy)),2)+
             math.sqrt(6)*z_list[zernike(2,2)]*(math.pow(math.cos(2*alpha),2)*(math.pow(math.cos(phy),2)+1)+2*math.pow(math.sin(2*alpha),2)*math.cos(phy))
             )/math.pow((radius*math.cos(phy)),2)
    

    return M,J45, J180
'''
#自推导的椭圆修正公式
#@arg 1: z_list 出瞳处的Zernike系数
#@arg 2: Ms、Mt 水平、数值方向的放大倍率
#@arg 2: ri、ro 入瞳、出瞳半径
def algo_elli_correction(z_list,Ms,Mt,ri,ro):
    c_list = []
    c_list.append(z_list[zernike(0,0)])
    c_list.append(z_list[zernike(1,-1)])
    c_list.append(z_list[zernike(1,1)])

    ratio = (ri/ro)*(ri/ro)
    A = (1/(Ms*Ms)+1/(Mt*Mt))
    B = (1/(Ms*Ms)-1/(Mt*Mt))
    #计算Z(2,-2)
    c_list.append(ratio*(z_list[zernike(2,-2)]/(Ms*Mt)))
    #计算Z(2,0)
    c_list.append(ratio*( A*z_list[zernike(2,0)]/2 + math.sqrt(2)*B*z_list[zernike(2,2)]/4))
    #计算Z(2,2)
    c_list.append(ratio*( math.sqrt(2)*B*z_list[zernike(2,0)]/2 + A*z_list[zernike(2,2)]/2))
    return c_list

#b3:Z(2,-2), b4:Z(2,0), b5:Z(2,2),theta是角度
def algo_elli_correction(b3,b4,b5,Ms,Mt,ri,ro,theta):
    rad = math.radians(theta)
    ratio = (ri/ro)*(ri/ro)
    A = (1/(Ms*Ms)+1/(Mt*Mt))
    B = (1/(Ms*Ms)-1/(Mt*Mt))
    C = cos(rad)*cos(rad) - sin(rad)*sin(rad)
    D = cos(rad)*sin(rad)
    E = Mt*Ms


    a3 = ratio*(b3*C/E+b4*math.sqrt(2)*D*B+b5*D*A)
    a4 = ratio*(A*b4/2+ math.sqrt(2)*B*b5/4)
    a5 = ratio*(-b3*2*D/E+b4*math.sqrt(2)*C*B/2+b5*C*A/2)

    return a3,a4,a5  

#Thibos转换
def algo_Thibos_line(z_list,k):
    c_list = []
    c_list.append(z_list[zernike(0,0)])
    c_list.append(z_list[zernike(1,-1)])
    c_list.append(z_list[zernike(1,1)])

    
    c_list.append( (k)*z_list[zernike(2,-2)])
    c_list.append( (k*k+1)*z_list[zernike(2,0)]/2 + math.sqrt(2)*(k*k-1)*z_list[zernike(2,2)]/4)
    c_list.append( math.sqrt(2)*(k*k-1)*z_list[zernike(2,0)]/2 + (k*k+1)*z_list[zernike(2,2)]/2)

    return c_list

def algo_Thibos_mat(z_list,k):
    clist = np.zeros_like(z_list)
    M = [
        [1,0,0,0,-math.sqrt(3),0],
        [0,0,2,0,0,0],
        [0,2,0,0,0,0],
        [0,0,0,0,2*math.sqrt(3),math.sqrt(6)],
        [0,0,0,2*math.sqrt(6),0,0],
        [0,0,0,0,2*math.sqrt(3),-math.sqrt(6)]
    ]

    N = [
        [1,0,0,0,-math.sqrt(3),0],
        [0,0,2*k,0,0,0],
        [0,2,0,0,0,0],
        [0,0,0,0,2*math.sqrt(3)*k*k,math.sqrt(6)*k*k],
        [0,0,0,2*math.sqrt(6)*k,0,0],
        [0,0,0,0,2*math.sqrt(3),-math.sqrt(6)]
    ]

    clist = np.dot(np.dot(np.linalg.inv(np.array(N)),np.array(M)),np.transpose(z_list))
    return np.transpose(clist).tolist()


def calc_sph_cyl_theta(M,J45,J180):
    cyl = -math.sqrt(J180*J180+J45*J45)
    sph = M - cyl/2
    theta = 0.5*math.degrees(math.atan2(J45,J180)) #角度
    if J180 < 0:
        theta = theta + 90
    
    if J180 >= 0 and J45 <= 0:
        theta = theta + 180

    return sph,cyl,theta

def calc_Mx_My(sph,cyl,inputD = 99 ):
    if inputD != 99:
        sphx = sph+cyl
        sphy = sph
        return math.sqrt(math.fabs(sphx/inputD)), math.sqrt(math.fabs(sphy/inputD))
    
    else:
        
        return 0,0

def getMagnificationRatio(b3,b4,b5,ro,inputD = 99):
    M,J45,J180 = calc_M_J45_J180(b3,b4,b5,ro)
    sph,cyl,axis = calc_sph_cyl_theta(M,J45,J180)
    Mx,My = calc_Mx_My(sph,cyl,inputD)
    return Mx,My

def readMagnificationRatio(direction = 'shuiping',angle = 0):
    Mx = 0.0
    My = 0.0
    return Mx,My

#复制compute_D_matrix矩阵
def calc_D_matrix(coordinates,n):
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
                math.sqrt(6) * 2 * y,
                math.sqrt(3) * 4 * x,
                math.sqrt(6) * 2 * x
            ])
        elif n == 3:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1,
                math.sqrt(6) * 2 * y,
                math.sqrt(3) * 4 * x,
                math.sqrt(6) * 2 * x,
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * (-2 + 9 * x ** 2 + 3 * y ** 2),
                math.sqrt(8) * (3 * x ** 2 - 3 * y ** 2)
            ])
        elif n == 4:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1,   #1
                math.sqrt(6) * 2 * y,
                math.sqrt(3) * 4 * x,
                math.sqrt(6) * 2 * x,
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * (-2 + 9 * x ** 2 + 3 * y ** 2),
                math.sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                math.sqrt(10) * (12 * x ** 2 * y - 4 * y ** 3),
                math.sqrt(10) * (-6 * y + 24 * x ** 2 * y + 8 * y ** 3),
                math.sqrt(5) * (-12 * x + 24 * x ** 3 + 24 * x * y ** 2),
                math.sqrt(10) * (-6 * x + 16 * x ** 3),
                math.sqrt(10) * (4 * x ** 3 - 12 * x * y ** 2)
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
                math.sqrt(6) * 2 * x, 
                math.sqrt(3) * 4 * y,
                math.sqrt(6) * -2 * y
            ])
        elif n == 3:                      #10
            matrix.append([
                1 * 0,
                2 * 1,
                2 * 0,
                math.sqrt(6) * 2 * x,
                math.sqrt(3) * 4 * y,
                math.sqrt(6) * -2 * y,
                math.sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                math.sqrt(8) * (-2 + 3 * x ** 2 + 9 * y ** 2),
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * -6 * x * y
            ])
        elif n == 4:                       #15
            matrix.append([
                1 * 0,
                2 * 1,    #1
                2 * 0,
                math.sqrt(6) * 2 * x,
                math.sqrt(3) * 4 * y,
                math.sqrt(6) * -2 * y,
                math.sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                math.sqrt(8) * (-2 + 3 * x ** 2 + 9 * y ** 2),
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * -6 * x * y,
                math.sqrt(10) * (4 * x ** 3 - 12 * x * y ** 2),
                math.sqrt(10) * (-6 * x + 8 * x ** 3 + 24 * x * y ** 2),
                math.sqrt(5) * (-12 * y + 24 * x ** 2 * y + 24 * y ** 3),
                math.sqrt(10) * (6 * y - 16 * y ** 3),
                math.sqrt(10) * (-12 * x ** 2 * y + 4 * y ** 3)
            ])
    return np.array(matrix)       #把一行的列表导入数组

#计算放大倍率，Mx = sph 轴向，My = cyl+sph径向
'''
def calc_Mx_My(cyl,sph,input):
    return sph/input, (cyl+sph)/input
'''
#替换函数read_function，对标准点进行Mx,My变换
def calc_std_spots(standard_spots_filepath,Mx=1.0,My=1.0):
    with open(standard_spots_filepath, 'r') as file:
        content = file.read()
    central = eval(content)
    result = [[row[0] * Mx, row[1] * My] for row in central]
    return result

#替换函数getcentraldiffs,对diffx和diffy进行变换
def calc_xydiff(non_standard_spots,list_standard_spots,Mx = 1.0,My = 1.0):
    # 计算偏移量并输出偏移量的结果
    x_diffs = []
    y_diffs = []

    for (x1, y1), (x2, y2) in zip(non_standard_spots, list_standard_spots):
        x_diff = (x1 - Mx*x2)/(Mx*Mx)
        y_diff = (y1 - My*y2)/(My*My)
        x_diffs.append(x_diff)  # 存放x的偏移量
        y_diffs.append(y_diff)  # 存放y的偏移量

    xy_diffs = []
    xy_diffs.extend(x_diffs)
    xy_diffs.extend(y_diffs)
    return xy_diffs,list_standard_spots

#J = D*C
#替换函数averageslope，计算J
def calc_J(xy_diffs,radius,F,X):
    j=[]
    for i in xy_diffs:
        x=i*X #像素转距离，单位um
        f=F*1000 #单位转换，单位um
        r=radius*X #单位转换，单位um
        j.extend([(x/f)]) 
    # J矩阵
    J = np.array(j)
    return J

#替换函数getstandard_spots,计算D
def calc_D(no_standard_points,radius,n,center,Mx,My):
    # #定义归一化后的R列表
    R = []
    # 计算图像中心坐标
    zx = center
    for central in no_standard_points:          #遍历与非标准点对应的标准点坐标
        x = (central[0]/Mx-zx[0])/radius
        y = (central[1]/My-zx[1])/radius
        R.append([x,y])


    coordinates = R

    # 计算D矩阵
    D = calc_D_matrix(coordinates,n)
    return D


def initaberration_correction(z_ist_with_initaber,dir = 0, angle = 0):
    init_aber = get_initaberration(dir,angle)
    z_list_out_initaber = [x-y for x,y in zip(z_ist_with_initaber,init_aber)]

    return z_list_out_initaber


def demodulation(ori_z_list,para):
    z = initaberration_correction(ori_z_list,para.direction,para.angle)
    Mx,My = readMagnificationRatio(para.direction,para.angle)
    b3,b4,b5 = algo_elli_correction(z[3],z[4],z[5],Mx,My,para.ri,para.ro,para.theta)

    M,J45,J180 = calc_M_J45_J180(b3,b4,b5,radius=para.ri)
    sph,cyl,axis = calc_sph_cyl_theta(M,J45,J180)
    results = configpara.finalresults()
    results.setfinalresults(z,
                            round(b3,3),round(b4,3),round(b5,3),
                            round(Mx,2),round(My,2),
                            round(sph,2),round(cyl,2),round(axis,2))
    
    print(f"phy: {para.angle},direction: {para.direction}")
    print(f"修正前：z(2,-2): {z[3]:.3f},z(2,0): {z[4]:.3f},z(2,2): {z[5]:.3f},Mx: {Mx:.3f},My: {My:.3f}")
    print(f"修正后：b(2,-2): {b3:.3f},b(2,0): {b4:.3f},b(2,2): {b5:.3f},cyl: {cyl:.3f},sph: {sph:.3f},axis: {axis:.3f}")
    return results
    
    

def calibration(ori_z_list,para):
    z = initaberration_correction(ori_z_list,para.direction,para.angle)
    Mx,My = getMagnificationRatio(z[3],z[4],z[5],para.ro,para.inputD)
    a3,a4,a5 = algo_elli_correction(z[3],z[4],z[5],Mx,My,para.ri,para.ro,para.theta)

    M,J45,J180 = calc_M_J45_J180(a3,a4,a5,radius=para.ri)
    cyl,sph,axis = calc_sph_cyl_theta(M,J45,J180)

    print(f"phy: {para.angle},direction: {para.direction}")
    print(f"b(2,-2): {z[3]:.3f},b(2,0): {z[4]:.3f},b(2,2): {z[5]:.3f},Mx: {Mx:.3f},My: {My:.3f}")
    print(f"a3: {a3:.3f},a4: {a4:.3f},a5: {a5:.3f},cyl: {cyl:.3f},sph: {sph:.3f},axis: {axis:.3f}")
