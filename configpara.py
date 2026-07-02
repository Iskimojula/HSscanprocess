def setdirectionfromtheta(theta):
        direction = 'shuiping'
        if theta == 90:
            direction = 'chuizhi'
            return direction
        return direction
class configparameters():
    def __init__(self):
        #入瞳半径mm
        self.ri = 0
        #出瞳半径mm
        self.ro = 0
        #视场角theta和direction相对应
        self.theta = 0
        #视场角angle,phy
        self.angle = 0
        #视镜片的度数
        self.inputD = 0
        #视场角direction
        self.direction = setdirectionfromtheta(self.theta)

    def printpara(self):
        print("config parameters:")
        print(f"入瞳半径：{self.ri:.2f}")
        print(f"出瞳半径：{self.ro:.2f}")
        print(f"视场角theta,angle: {self.theta:.2f},{self.angle:.2f}")
        print(f"视场角方向direction:{self.direction}")
    
    def setconfigparameters(self,ri,ro,theta,angle):
        self.ri = ri
        self.ro = ro
        self.theta = theta
        self.angle = angle
        self.direction = setdirectionfromtheta(self.theta)
        self.printpara()


