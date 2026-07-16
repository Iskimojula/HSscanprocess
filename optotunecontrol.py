import numpy as np
import optoMDC  # 确保已安装此库
import time

def deg_to_xy(mech_deg):         
    return np.tan(np.deg2rad(2 * mech_deg)) / np.tan(np.deg2rad(50))

def xy_to_deg(xy):                
    return 0.5 * np.rad2deg(np.arctan(xy * np.tan(np.deg2rad(50))))

class optotune:
    def __init__(self):
        self.mre2 = optoMDC.connect()
        self.mre2.Mirror.Channel_0.SetControlMode(2)
        self.mre2.Mirror.Channel_1.SetControlMode(2)
        self.mre2.Mirror.Channel_0.StaticInput.SetAsInput()
        self.mre2.Mirror.Channel_1.StaticInput.SetAsInput()
    def setdegreeX(self,target_x):
        x = deg_to_xy(target_x)
        self.mre2.Mirror.Channel_0.StaticInput.SetXY(x)
    def setdegreeY(self,target_y):
        y = deg_to_xy(target_y)
        self.mre2.Mirror.Channel_1.StaticInput.SetXY(y)
    
    def getdegreex(self):
        actual_x = self.mre2.Mirror.Channel_0.StaticInput.GetXY()
        
        return xy_to_deg(float(actual_x[0]))


    def getdegreey(self):
        actual_y = self.mre2.Mirror.Channel_1.StaticInput.GetXY()
        return xy_to_deg(float(actual_y[0]))
    
    def setxy(self,target_degreex,target_degreey):
        self.setdegreeX(target_degreex)
        self.setdegreeY(target_degreey)
    def setzero(self):
        self.setdegreeX(0)
        self.setdegreeY(0)
    def getxy(self):
        return round(self.getdegreex(),1), round(self.getdegreey(),1)
    



