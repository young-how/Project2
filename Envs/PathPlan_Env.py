from BaseClass.BaseEnv import *
from BaseClass.CalMod import *
import torch
import numpy as np
import math
import random
import csv
import threading
import copy
# use_cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor  
# device = torch.device("cuda" if use_cuda else "cpu")    #使用GPU进行训练
from  torch.autograd import Variable

class PathPlan_Env(BaseEnv):
    def __init__(self,param:dict) -> None:
        #初始化
        #输入参数：
        #       param   :   初始化参数字典
        super().__init__(param)   #初始化基类
        self.eps=float(None2Value(param.get('eps'),0.1))      #贪心概率
        self.Is_On_Policy=int(param.get('Is_On_Policy'))      #样本采样方式1：在线策略 2：离线策略

        #威胁描述图层
        self.map=np.zeros((self.len,self.width))   #道路图层

        #根据参数初始化Agent
        self.num_UAV=int(param.get('num_UAV'))  #UAV数目
        agents_params=param.get('Agent')
        for i in range(self.num_UAV):
            agents_params['name']='UAV_'+str(i)
            agents_params['j']=i
            obj_agent=self.AgentFactory.Create_Agent(agents_params,self)
            agent_trainer=agents_params.get('Trainer')
            agent_trainer['name']=obj_agent.name   #同一名称
            obj_agent.Trainer=self.TrainerFactory.Create_Trainer(agent_trainer)   #创建智能体专属的训练器
            self.Agents.append(obj_agent)

        #仿真推演结果读入与录入
        self.result={'success':0,'failed:':0,'meet_threaten':0,'normal':0,'loss':None,'sum_epoch':0,'eps':0.1} 

        #实验数据记录周期
        self.epoch=0
        self.print_loop=int(param.get('print_loop'))

        #多线程优化
        self.threads=[]

        

        
    def Reset_Thread_UAV(self,uav):
        uav.reset()   #随机重置状态


    #采用多线程优化uav的重置过程
    def Scene_Random_Reset(self):
        self.reset_threads=[]
        for indx,uav in enumerate(self.Agents):
            #普通调用
            #uav.reset()   #随机重置状态

            #多线程调用
            thread = threading.Thread(target=self.Reset_Thread_UAV, args=(uav,))
            self.reset_threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in self.reset_threads:
            thread.join()
            
    def Set_Space_Scale(self,len:int,width:int,h:int):
        #重新设置空间大小
        self.len=len
        self.width=width
        self.h=h

    def Draw_static_map(self):
        pass

    def Check_uav_Done(self):
        #检查环境中的所有智能体是否都进入任务完成状态
        #输出参数：
        #       flag    ：  状态1：所有Agent都完成任务  0：存在Agent没有完成任务
        for agent in self.Agents:
            if not agent.done:
                return False
        return True
    
    def check_threaten(self,p:Loc):
        pass
    
    def check_total_threaten(self,x,y,z):
        try:
            if self.map[int(x),int(y)]>0:
                return 1
            else:
                return 0
        except Exception as e:
            print(e.args)  #输出异常信息
            return 1
    
    def run(self):
        #环境智能体进行决策，并自定义环境中各类元素的状态变更规则
        pass

    def Choose_Action(self,index:int,eps=0.2):
        #epsilon贪心策略选取动作索引
        #输入参数
        #       index   ：  智能体所在Agents集合的索引
        #       eps     ：  eps贪心概率
        #输出参数
        #       action  ：  动作值
        if self.Is_AC==1:
            #actor-critic框架(包括SAC)
            sample = random.random()
            if  sample > eps:
                state = self.Agents[index].state()  #指定智能体的状态值
                tensor_state=FloatTensor(np.array(state))
                probs = self.Agents[index].Trainer.actor(tensor_state)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()
                return torch.tensor([[action.item()]], device=device) 
            else:
                #随机选取动作
                return torch.tensor([[random.randrange(self.Agents[index].act_num)]], device=device)   #随机选取动作
        elif self.Is_AC==2:  #DDPG算法
            sample = random.random()
            if  sample > eps:
                state = self.Agents[index].state()  #指定智能体的状态值
                tensor_state=FloatTensor(np.array(state))
                #state = torch.tensor([state], dtype=torch.float).to(self.device)
                action = self.Agents[index].Trainer.actor(tensor_state).item()
                # 给动作添加噪声，增加探索
                action = action + 0.01* np.random.randn(1)
            else:
                #随机选取动作
                return torch.tensor([random.uniform(-1,1)], device=device)   #随机选取动作
            return action
        elif self.Is_AC==3:  #SAC算法(离散动作空间)
            state = self.Agents[index].state()  #指定智能体的状态值
            tensor_state=FloatTensor(np.array(state))
            probs = self.Agents[index].Trainer.actor(tensor_state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return torch.tensor([[action.item()]], device=device) 
        elif self.Is_AC==4:  #SAC算法(连续动作空间)
            state = self.Agents[index].state()  #指定智能体的状态值
            state = torch.tensor([state], dtype=torch.float).to(device)
            action = self.Agents[index].Trainer.actor(state)[0]
            return torch.tensor([[action.item()]], device=device) 
        else:
            #DQN框架类型
            state=self.Agents[index].state()  #指定智能体的状态值
            tensor_state=FloatTensor(np.array(state))
            sample = random.random()
            if  sample > eps:
                with torch.no_grad():
                    y=self.Agents[index].Trainer.q_local(Variable(tensor_state).type(FloatTensor))
                    value=y.data.max(0)[1].view(1, 1) 
                    return  value   #根据Q值选择行为
            else:
                #随机选取动作
                return torch.tensor([[random.randrange(self.Agents[index].act_num)]], device=device)   #随机选取动作
    
    #动作选择方法2，将决定动作的权力交给训练器
    def Choose_Action2(self,index:int,eps=0.2):
        #epsilon贪心策略选取动作索引
        #输入参数
        #       index   ：  智能体所在Agents集合的索引
        #       eps     ：  eps贪心概率
        #输出参数
        #       action  ：  动作值
        state = self.Agents[index].state()  #获取状态
        return self.Agents[index].Trainer.get_action(state,eps) #返回动作值
        

    def Evaluation_Action(self,index:int):
        #对指定智能体执行动作后的状态进行评价
        pass
    
    
    def record(self,result):
        #统计训练信息
        data=[]
        pass
    
    def Reset_Result(self,eps_rate):
        #重置统计量
        self.result={'success':0,'lose':0,'meet_threaten':0,'normal':0,'loss':0,'sum_epoch':0,'eps':eps_rate,'score':0,'average_score':0,'step':0}   #返回的推演信息：success-完成任务的智能体数目，failed：失败的智能体数目，meet_threaten-遇到威胁的次数
    
    #线程启用函数（离线策略）
    def run_thread_OffPolicy(self,uav,ind,eps_rate):
        if uav.done:
            return #该智能体已完成任务
        #智能体决策
        state_test=uav.state()                          #当前的状态图
        #action=self.Choose_Action(ind,eps_rate)   #epsilon贪心策略选取动作值
        action=self.Choose_Action2(ind,eps_rate)   #采用新版的动作选择方式
        next_state, reward, done, info= self.Move_Agent(ind,action)  #根据选取的动作改变状态，获取收益
        self.Run_statistics(info)   #对智能体运行信息进行统计
        #存储交互经验（存放tensor版的数据）
        uav.Trainer.Push_Replay(
            (FloatTensor(np.array([state_test])), 
            action, 
            FloatTensor([[reward]]), 
            FloatTensor(np.array([next_state])), 
            FloatTensor([[done]])))

        #将普通格式的数据存放到回放池中
        uav.Trainer.replay_memory.add(state_test, action, reward, next_state, done)
        if  len(uav.Trainer.replay_memory.buffer)>uav.Trainer.Batch_Size:
            b_s, b_a, b_r, b_ns, b_d = uav.Trainer.replay_memory.sample2(uav.Trainer.Batch_Size)
            uav.transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}

    
    #线程启用函数(在线策略)
    def run_thread_OnPolicy(self,uav,ind,eps_rate):
        if uav.done:
            return #该智能体已完成任务
        #智能体决策
        state=uav.state()                          #当前的状态图
        #action=self.Choose_Action(ind,eps_rate)   #epsilon贪心策略选取动作值
        action=self.Choose_Action2(ind,eps_rate)   #采用第二种动作选择方式
        next_state, reward, done, info= self.Move_Agent(ind,action)  #根据选取的动作改变状态，获取收益
        self.Run_statistics(info)   #对智能体运行信息进行统计
        # uav.transition_dict['states'].append(FloatTensor(np.array([state])))
        # uav.transition_dict['actions'].append(action)
        # uav.transition_dict['next_states'].append(FloatTensor(np.array([next_state])))
        # uav.transition_dict['rewards'].append(FloatTensor([[reward]]))
        # uav.transition_dict['dones'].append(FloatTensor([[done]]))
        #存放非tensor格式数据
        uav.transition_dict['states'].append(state)
        uav.transition_dict['actions'].append(action)
        uav.transition_dict['next_states'].append(next_state)
        uav.transition_dict['rewards'].append(reward)
        uav.transition_dict['dones'].append(done)
                    

    def run_eposide(self,eps_rate=0.1):
        #环境智能体进行决策，并自定义环境中各类元素的状态变更规则
        #对一个任务场景进行推演，直到所有智能体到达终止状态
        #根据自定义规则进行仿真模拟推演
        #返回参数
        #       result    ：  推演结果，dict类型
        self.Reset_Result(eps_rate)     #重置统计量
        self.Scene_Random_Reset()  #计算机随机生成场景
        self.threads=[]
        if self.Is_On_Policy==1:
            #在线策略训练，AC类算法训练方式
            for ind,uav in enumerate(self.Agents):
                uav.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}          
            while(1):
                self.run()          #场景元素进行变更
                for ind,uav in enumerate(self.Agents):
                    #多线程调用
                    thread = threading.Thread(target=self.run_thread_OnPolicy, args=(uav,ind,eps_rate,))  #在线训练
                    self.threads.append(thread) 
                    thread.start()
                # 等待所有线程完成
                for thread in self.threads:
                    thread.join()

                #如果所有智能体完成任务，推出当前推演
                if self.Check_uav_Done():
                    break
            #train_info=self.train_on_policy()   #在线训练,运行完一幕才开始训练
            train_info=self.update()   #在线策略,运行完一幕才开始训练
        else:  
             #离线训练策略
            while(1):
                self.run()          #场景元素进行变更
                for ind,uav in enumerate(self.Agents):
                    #多线程调用
                    uav.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []} #重置
                    thread = threading.Thread(target=self.run_thread_OffPolicy, args=(uav,ind,eps_rate,))
                    self.threads.append(thread)
                    thread.start()

                # 等待所有线程完成
                for thread in self.threads:
                    thread.join()
                    
                #train_info=self.train_off_policy()   #执行一个动作训练一次
                train_info=self.update()   #离线策略，执行一个动作训练一次
                #如果所有智能体完成任务，推出当前推演
                if self.Check_uav_Done():
                    break
        
        #train_info=self.train_off_policy_FL()   #执行完一个任务训练一次,并附带联邦学习
        self.Train_statistics(train_info)  #统计训练结果，进行格式化保存
        self.epoch+=1   #运行了一个周期

        #记录每个轮次UAV的结算结果
        for uav in self.Agents:
            item={}
            item['sum_epoch']=self.epoch
            item['score']=uav.score
            item['average_score']=uav.score
            item['step']=uav.Step

            item['energy_cost']=Eu_Loc_distance(uav.position,uav.goal)
            item['task_collect']=0
            item['Energy_Efficent']=0
            item['UE_waiting_time']=0
            item['Covered_rate']=0
            item['UE_waiting_time']=0
            uav.infos.append(item) 
            

        #记录这段训练数据
        if self.epoch%self.print_loop==0:
            for uav in self.Agents:
                uav.record_list()
        
        return self.result
    
    def Run_statistics(self,info):
        #对运行结果进行统计
        self.result[info]=self.result[info]+1             #统计运行结果

    def Train_statistics(self,train_info):
        #对训练结果进行统计
        num=len(train_info)
        for index,info in enumerate(train_info):
            if type(info['loss'])!=int:
                self.result['loss']+=info['loss'].item()
            else:
                self.result['loss']+=info['loss']
            self.result['sum_epoch']+=info['sum_epoch']
            self.result['score']+=self.Cal_Score()[0]
            self.result['average_score']+=self.Cal_Score()[1]
            self.result['step']+=self.Agents[index].Step

        #平均
        if num!=0:
            self.result['loss']/=num
            self.result['sum_epoch']/=num
            self.result['score']/=num
            self.result['average_score']/=num
            self.result['step']/=num

    def Cal_Score(self):
        #计算环境中所有智能体的总得分与平均得分
        #返回参数
        #   sum_score       :   总体得分
        #   average_score   :   平均得分
        sum_score=0
        average_score=0
        for agent in self.Agents:
            sum_score+=agent.score
        return sum_score,sum_score/len(self.Agents)
    
    def run_XML_scene(self):
        #环境智能体进行决策，并自定义环境中各类元素的状态变更规则
        #对一个任务场景进行推演，直到所有智能体到达终止状态
        #根据自定义规则进行仿真模拟推演
        #根据XML生成场景
        pass 
    
    def render(self):
        #可视化接口，利用已有信息进行环境可视化
        pass
    
    def Load_Scene_FromXML(self,XML_path="../config/FMEC.xml"):
        #从xml文件中对环境成员进行配置，用于特定场景的强化学习训练或者训练效果测试
        #解析XML文件,从XML文件中解析成dict
        p_dict=XML2Dict(XML_path)
        if p_dict!=None:
            env_dict=p_dict.get('env')
            if env_dict!=None:
                #根据参数初始化Agent
                agents_params=env_dict.get('Agents')
                if agents_params != None:
                    agents=agents_params.get('Agent')
                    for agent_param in agents:
                        obj_agent=self.AgentFactory.Create_Agent(agent_param,self)
                        agent_trainer=agent_param.get('Trainer')
                        if obj_agent != None:
                            agent_trainer['name']=obj_agent.name   #同一名称
                            obj_agent.Trainer=self.TrainerFactory.Create_Trainer(agent_trainer)   #创建智能体专属的训练器
                            self.Agents.append(obj_agent)
                
                #根据参数初始化Threaten
                threaten_params=env_dict.get('Threatens')
                if threaten_params != None:
                    Threatens=threaten_params.get('Threaten')
                    for threaten_param in Threatens:
                        obj_threaten=self.ThreatenFactory.Create_Threaten(threaten_param,self)
                        if obj_threaten != None:
                            self.Threatens.append(obj_threaten)
        
                #根据参数初始化训练器
                trainer_params=env_dict.get('Trainer')
                if trainer_params != None:
                    self.Trainer=self.TrainerFactory.Create_Trainer(trainer_params)
    
    def train(self):
        #根据replay_buffer中的数据进行训练
        re=self.Trainer.learn()
        return re

    def train_off_policy(self):
        #离线训练（有经验回放池）
        re=[]
        for uav in self.Agents:
            #串行计算
            item=uav.Trainer.learn_off_policy()
            item['score']=uav.score
            item['average_score']=uav.score
            item['step']=uav.Step

            item['energy_cost']=uav.energy_cost_total
            item['task_collect']=uav.task_collect
            item['Energy_Efficent']=uav.task_collect/(uav.energy_cost_total+0.001)
            item['UE_waiting_time']=0
            for ue in uav.UEs:
                item['UE_waiting_time']+=ue.Caculate_wait_time  #统计每一个ue的等待时间
            item['UE_waiting_time']=item['UE_waiting_time']/len(uav.UEs)   #平均等待时间
            #uav.infos.append(item) 
            re.append(item)
        return re
    
    def train_on_policy(self):
        #在线训练（没有经验回放池）
        re=[]
        for uav in self.Agents:
            item=uav.Trainer.learn_on_policy(uav.transition_dict)
            item['score']=uav.score
            item['average_score']=uav.score
            item['step']=uav.Step

            item['energy_cost']=uav.energy_cost_total
            item['task_collect']=uav.task_collect
            item['Energy_Efficent']=uav.task_collect/(uav.energy_cost_total+0.001)
            item['UE_waiting_time']=0
            for ue in uav.UEs:
                item['UE_waiting_time']+=ue.Caculate_wait_time  #统计每一个ue的等待时间
            item['UE_waiting_time']=item['UE_waiting_time']/len(uav.UEs)   #平均等待时间
            #uav.infos.append(item) 
            re.append(item)
        return re
    
    #通用型更新方法，适用于离线和在线策略
    def update(self):
        #离线训练（有经验回放池）
        re=[]
        for uav in self.Agents:
            #串行计算
            item=uav.Trainer.update(uav.transition_dict)
            item['score']=uav.score
            item['average_score']=uav.score
            item['step']=uav.Step

            item['energy_cost']=0
            item['task_collect']=0
            item['Energy_Efficent']=0
            item['UE_waiting_time']=0
            re.append(item)
        return re
    
    def render(self):
        #可视化接口，利用已有信息进行环境可视化
        pass
    
    def store_experience(self,replay):
        #存放经验信息
        self.Trainer.Push_Replay(replay)
