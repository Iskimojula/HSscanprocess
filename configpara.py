import threading

def setdirectionfromtheta(theta):
        direction = 'shuiping'
        if theta == 90:
            direction = 'chuizhi'
            return direction
        return direction
class configparameters():
    def __init__(self):
        self._lock = threading.Lock()
        #入瞳半径mm
        self.ri = 1.6
        #出瞳半径mm
        self.ro = 2
        #视场角theta和direction相对应
        self.theta = 0
        #视场角angle,phy
        self.angle = 0
        #视镜片的度数
        self.inputD = 0
        #视场角direction
        self.direction = setdirectionfromtheta(self.theta)

    def printpara(self):
        with self._lock:
            print("config parameters:")
            print(f"入瞳半径：{self.ri:.2f}")
            print(f"出瞳半径：{self.ro:.2f}")
            print(f"视场角theta,angle: {self.theta:.2f},{self.angle:.2f}")
            print(f"视场角方向direction:{self.direction}")
    
    def setconfigparameters(self,ri,ro,theta,angle):
        with self._lock:
            self.ri = ri
            self.ro = ro
            self.theta = theta
            self.angle = angle
            self.direction = setdirectionfromtheta(self.theta)

    def checkvaild(self):
        if self.ri==0 or self.ro==0:
            return False
        return True


class finalresults():
    def __init__(self):
        self._lock = threading.Lock()

        #已经去掉了本征像差
        self.beforedemod = {
            
            'a0': 0 ,#z(0,0)
            'a1': 0 ,#z(1,-1)
            'a2': 0 ,#z(1,1)
            'a3': 0 ,#z(2,-2)
            'a4': 0 ,#z(2,0)
            'a5': 0 ,#z(2,2)
        }

        self.afterdemod = {
            
            'b0': 0 ,#z(0,0)
            'b1': 0 ,#z(1,-1)
            'b2': 0 ,#z(1,1)
            'b3': 0 ,#z(2,-2)
            'b4': 0 ,#z(2,0)
            'b5': 0 ,#z(2,2)
        }


        #放大倍率
        self.Mx = 0
        #视镜片的度数
        self.My = 0
        #sph
        self.sph = 0
        self.cyl = 0
        self.axis = 0

    
    def setfinalresults(self,a,b3,b4,b5,Mx,My,sph,cyl,axis):
        with self._lock:
            
            self.beforedemod['a0'] = round(a[0],3)
            self.beforedemod['a1'] = round(a[1],3)
            self.beforedemod['a2'] = round(a[2],3)
            self.beforedemod['a3'] = round(a[3],3)
            self.beforedemod['a4'] = round(a[4],3)
            self.beforedemod['a5'] = round(a[5],3)

            self.afterdemod['b3'] = b3
            self.afterdemod['b4'] = b4
            self.afterdemod['b5'] = b5

            #放大倍率
            self.Mx = Mx
            #视镜片的度数
            self.My = My
            #sph
            self.sph = sph
            self.cyl = cyl
            self.axis = axis
