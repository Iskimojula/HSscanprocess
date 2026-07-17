import threading
import time
def setdirectionfromtheta(theta):
    if theta == 0:
        direction = 'shuiping'
    if theta == 90:
        direction = 'chuizhi'
    if theta == 45:
            direction == 'degree45'
    if theta == 130:
            direction == 'degree135'
    return direction

transdirstr = {
     'shuiping' : '水平',
     'chuizhi' :'垂直',
     'degree45' : '45度',
     'degree135' : '135度'
}
     
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
    
    def makestrforsave(self):
         return f"入瞳半径：{self.ri:.2f}，出瞳半径：{self.ro:.2f}，视场角：{transdirstr[self.direction]}，偏转X：{self.angle:.1f}，偏转Y：{self.angle:.1f}，试镜片：{self.inputD:.2f}"
    def setconfigparameters(self,ri,ro,theta,angle,lens):
        with self._lock:
            self.ri = ri
            self.ro = ro
            self.theta = theta
            self.angle = angle
            self.inputD = lens
            self.direction = setdirectionfromtheta(self.theta)

    def checkvaild(self):
        if self.ri==0 or self.ro==0:
            return False
        return True


class finalresults():
    def __init__(self):
        self._lock = threading.Lock()
        self.b = [0.0]*15
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
            self.a = a
            self.beforedemod['a0'] = round(a[0],3)
            self.beforedemod['a1'] = round(a[1],3)
            self.beforedemod['a2'] = round(a[2],3)
            self.beforedemod['a3'] = round(a[3],3)
            self.beforedemod['a4'] = round(a[4],3)
            self.beforedemod['a5'] = round(a[5],3)

            self.b[3] = b3
            self.b[4] = b4
            self.b[5] = b5
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
    def setinputdata(self,ori_z_list,para):
            with self._lock:
                 self.ori_a = ori_z_list
                 self.configpara = para
                
    def save(self,filepath):

         configdata = self.configpara.makestrforsave()
         enlargeratio = f"Mx：{self.Mx},My：{self.My}"
         refact = f"sph,cyl,ratio：{self.sph}，{self.cyl}，{self.axis}"
         zerniketitle = "z(0,0),z(1,-1),z(1,1),z(2,-2),z(2,0),z(2,2),z(3,-3),z(3,-1),z(3,1),z(3,3),z(4,-4),z(4,-2),z(4,0),z(4,2),z(4,4)"
         ori_a_data = ",".join(str(x) for x in self.ori_a[:15])
         a_data=",".join(str(x) for x in self.a[:15])
         b_data = ",".join(str(x) for x in self.b)

         with open(filepath,"a",encoding="utf-8") as f:
              f.write(configdata + "\n")
              f.write(enlargeratio + "\n")
              f.write(refact + "\n")
              f.write(zerniketitle + "\n")
              f.write("ori_a_data: "+ori_a_data + "\n")
              f.write("a_data: "+a_data + "\n")
              f.write("b_data"+b_data + "\n")
              f.write( "\n")
