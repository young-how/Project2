from BaseClass.BaseAgent import *
from BaseClass.CalMod import *
import random
import csv
import copy
import  os
root=os.path.dirname(os.path.abspath(__file__)) #当前根目录
from  torch.autograd import Variable 
class ZDJ(BaseAgent):
    def __init__(self,param:dict,env=None) -> None:
        #初始化
        #输入参数：
        #       param   :   初始化参数字典
        super().__init__(param)   #初始化基类
        #自定义附加参数添加在如下地方：
        self.param=param  #保存初始化参数信息
        #UAV特有属性
        self.name=param.get("name")  #无人机名称
        self.V_vector=Loc(0,0,0)  #速度矢量
        self.Min_V=float(param.get("Max_V"))   #最大速度
        self.Max_V=float(param.get("Max_V"))   #最大速度
        self.Steering_angle=float(param.get("Steering_angle"))/180*math.pi   #最大转向角
        seta=random.uniform(0,2*math.pi)   #随机初始化速度方向
        self.V_vector.x=self.Max_V*math.cos(seta)
        self.V_vector.y=self.Max_V*math.sin(seta)  #初始速度大小为Min_V
        self.V=self.Calc_V()     #计算速度大小
        self.ac=float(param.get("Acceration"))/1000   #加速度大小(恒定，km/s^2)
        self.Max_Step=int(param.get("Max_Step"))   #最大步长
        self.Step=0
        self.target_step=100
        #UAV动作空间（离散）
        self.act=[
            [self.ac,0],[self.ac,0.25*math.pi],[self.ac,0.5*math.pi],[self.ac,0.75*math.pi],[self.ac,math.pi],[self.ac,1.25*math.pi],[self.ac,1.5*math.pi],[self.ac,1.75*math.pi] ,
            [0.5*self.ac,0],[0.5*self.ac,0.25*math.pi],[0.5*self.ac,0.5*math.pi],[0.5*self.ac,0.75*math.pi],[0.5*self.ac,math.pi],[0.5*self.ac,1.25*math.pi],[0.5*self.ac,1.5*math.pi],[0.5*self.ac,1.75*math.pi],
            [0,0]
                ]             
        self.path=[]  #路径点
        self.V_record=[]  #记录速度大小
        self.R_record=[]  #记录奖励大小

        self.Trainer=None
        #样本集合
        self.transition_dict= {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        self.env=env  #设置所处的环境

        self.CsvWriter=None
        self.infos=[]   #训练的反馈列表
        self.Init_Record_Mod()
        #目标点
        self.goal=Loc(75,75,0)
        #数据收集情况
        self.data_num=0



    def Calc_V(self):
        #根据速度向量计算速度大小
        V=Eu_Loc_distance(Loc(0,0,0),self.V_vector)
        if V>self.Max_V:  #自动调整速度
            self.V_vector.x=self.V_vector.x*(self.Max_V/V)
            self.V_vector.y=self.V_vector.y*(self.Max_V/V)
            self.V=V
        return V
    
    def Set_Env(self,env):
        #设置所属环境类
        self.env=env

    #初始化记录模块
    def Init_Record_Mod(self):
        #初始化训练记录模块
        import datetime
        cur_time=datetime.datetime.now()
        cur_time= cur_time.strftime('%m_%d_%Y(%H_%M_%S)')
        path='logs/%s_%s.csv' %(self.name,cur_time)
        file=open(path,'a+',newline="")
        self.CsvWriter=csv.writer(file)
        self.CsvWriter.writerow(["sum_Episode","Episode"," Score"," Avg.Score","eps-greedy","success","failed","meet_threaten",'loss','step','Q-value','energy_cost','task_collect','Energy_Efficent','UE_waiting_time','Covered_rate'])

    #记录训练数据
    def record_list(self):
        Covered_rate=sum_epoch=score=average_score=eps=success=lose=meet_threaten=loss=step=energy_cost=task_collect=UE_waiting_time=Energy_Efficent=0
        list_len=len(self.infos)
        if self.infos!=[]:
            for line in self.infos:
                sum_epoch+=line['sum_epoch']
                score+=line['score']
                average_score+=line['average_score']
                loss+=0   #loss的绝对值
                step+=line['step']

                #无人机采集UE任务所需数据
                energy_cost+=line['energy_cost']
                task_collect+=line['task_collect']
                UE_waiting_time+=line['UE_waiting_time']
                Energy_Efficent+=line['Energy_Efficent']
                Covered_rate+=line['Covered_rate']
            
            self.CsvWriter.writerow([sum_epoch/list_len,self.Trainer.epoch,score/list_len,average_score/list_len,eps/list_len,success/list_len,lose/list_len,meet_threaten/list_len,loss/list_len,step/list_len,0,energy_cost/list_len,task_collect/list_len,Energy_Efficent/list_len,UE_waiting_time/list_len,Covered_rate/list_len])
        self.infos=[]  #清空列表

    def reset(self):
        self.Step=0
        self.score=0
        self.done=False
        self.path=[] 
        self.V_record=[]  #记录速度大小
        self.R_record=[]  #记录奖励大小
        self.position=Loc(int(self.param.get("position").get('x')),int(self.param.get("position").get('y')),int(self.param.get("position").get('z')))  #初始化在空间中的位置
        self.V_vector=Loc(0,0,0)  #速度矢量
        seta=random.uniform(0,2*math.pi)   #随机初始化速度方向
        self.V_vector.x=self.Max_V*math.cos(seta)
        self.V_vector.y=self.Max_V*math.sin(seta)  #初始速度大小为Min_V
        self.V=self.Calc_V()     #计算速度大小
        self.infos=[]   #训练的反馈列表
        #数据收集情况
        self.data_num=0
        #目标点
        seta=random.uniform(0,2*math.pi)
        self.goal=Loc(self.position.x+20*math.cos(seta),self.position.y+20*math.sin(seta),0)  #随机生成距离30km的终点
  
    def is_in_building(self,position):
        return False
    def Set_Max_Step(self,n:int):
        #设置任务的最大步长
        self.max_step=n

    def Set_State_Map(self,L:int,W:int,H:int):
        #设置状态图层形状
        #L图层长度，W图层宽度，H图层厚度
        self.state_map=np.zeros((1,H,L,W))

    def Set_Actions(self,Action_Dic):
        #通过设定好的动作集合初始化智能体的动作空间
        #Action_Dic为动作空间字典，包含了对应动作编号下的坐标（或姿态）变更规则
        #例如actions={'L':Loc(0,-1,0),'R':Loc(0,1,0),'U':Loc(1,0,0),'L':Loc(-1,0,0)}
        self.Actions=Action_Dic
    
    def Observation(self,env):
        #该函数实现环境状态感知，使智能体能感知到环境类中的部分或全部成员
        pass                                          

    # ██╗   ██╗██████╗ ██████╗  █████╗ ████████╗███████╗
    # ██║   ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
    # ██║   ██║██████╔╝██║  ██║███████║   ██║   █████╗  
    # ██║   ██║██╔═══╝ ██║  ██║██╔══██║   ██║   ██╔══╝  
    # ╚██████╔╝██║     ██████╔╝██║  ██║   ██║   ███████╗
    #  ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
    def update(self,action):
        #action: [转向角度，加速度的取值]
        global_r=0  #全局任务奖励
        self.Step+=1    #全局步长统计
        #速度大小变更(加速度可变情况下)
        # self.V=self.Calc_V()   #计算速度大小,并自适应调整
        # V=self.V+self.ac*action[1]   #变化速度
        # if V>self.Max_V:
        #     V=self.Max_V
        # elif V<self.Min_V:
        #     V=self.Min_V
        
        # self.V=self.Calc_V()   #计算新速度
        #速度方向变更
        seta_old=calculate_angle(Loc(0,0,0),self.V_vector)  #计算速度的方位角
        seta_new=seta_old+action[0]*self.Steering_angle   #单位弧度
        self.V_vector.x=self.Max_V*math.cos(seta_new)
        self.V_vector.y=self.Max_V*math.sin(seta_new)
        self.V=self.Calc_V()   #计算新速度
        #坐标变更
        dis_old=Eu_Loc_distance(self.position,self.goal)
        self.position.x+=self.V_vector.x   #变更x坐标
        self.position.y+=self.V_vector.y    #变更y坐标
        dis_new=Eu_Loc_distance(self.position,self.goal)
        global_r+=2*(dis_old-dis_new)
        self.path.append(action)
        self.V_record.append(Eu_Loc_distance(self.position,self.goal))
        #计算速度方位和终点方位的差值
        tri_goal=calculate_angle(self.position,self.goal)
        tri_V=calculate_angle(Loc(0,0,0),self.V_vector)
        global_r-=0.1*abs(tri_goal-tri_V)

        global_r-=0.5  #每走一步就惩罚一次
        self.R_record.append(global_r)  #记录奖励大小

        if self.Step>=self.Max_Step:
            self.done=True
            global_r+=(50-Eu_Loc_distance(self.position,self.goal))
            self.score+=global_r
            return global_r,True,'lose'
        elif Eu_Loc_distance(self.position,self.goal)<3:
            self.done=True
            global_r+=50  
            self.score+=global_r
            self.target_step=self.Step
            return global_r,True,'success'
        else:
            self.score+=global_r
            return global_r,False,'normal' 

    #简化版状态空间
    def state(self):
        #所有技能所需要的状态空间
        state_map=np.zeros((1,1,1,5))
        state_map[0,0,0,0]=self.Step/100
        state_map[0,0,0,1]=(self.goal.x-self.position.x)/10
        state_map[0,0,0,2]=(self.goal.y-self.position.y)/10
        state_map[0,0,0,3]=self.V   #速度大小
        state_map[0,0,0,4]=calculate_angle(Loc(0,0,0),self.V_vector) #速度方向
        return copy.copy(state_map[0,0,0,:])               

