#描述雷达威胁
from BaseClass.BaseThreaten import *
from BaseClass.CalMod import *
import random
class RD(BaseThreaten):
    def __init__(self,param,env=None) :
        super().__init__(param)   #初始化基类
        self.base_positon=Loc(int(param.get("position").get('x')),int(param.get("position").get('y')),int(param.get("position").get('z')))
        self.base_R=None2Value(float(param.get("_R")),10)
        self._R=None2Value(float(param.get("_R")),10)
        self.seta1=None2Value(float(param.get("seta1")),0)
        self.seta2=None2Value(float(param.get("seta2")),0.6*math.pi)
        self.env=env

        #随机生成的参数范围
        self.delt_position=None2Value(float(param.get("delt_position")),0)  #位置误差大小
        self.delt_R=None2Value(float(param.get("delt_R")),0)    #半径误差大小

    #重载构造函数(只考虑坐标和半径)
    def __init__(self,position:Loc,R:int) :
        self.position=position
        self._R=R

    def reset(self):
        self.position=self.position_
        self._R=self._R_

    def reset_random(self):
        self.position=Loc(self.base_positon.x+random.randint(-self.delt_position,self.delt_position),
                          self.base_positon.y+random.randint(-self.delt_position,self.delt_position),
                          0) #随机生成坐标位置
        self._R=self.base_R+random.uniform(-self.delt_R,self.delt_R)  
        self.seta1=random.uniform(0,2*math.pi)
        self.seta2=self.seta1+random.uniform(0.3*math.pi,0.6*math.pi)


    #根据坐标返回威胁
    def check_threaten(self,position:Loc):
        #根据坐标位置返回威胁
        if Eu_Loc_distance(position,self.position)<self._R:
            return 1
        else:
            return 0