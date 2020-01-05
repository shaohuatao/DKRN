'''
1表示横轴，方向从左到右；0表示纵轴，方向从上到下。
当axis=1时，数组的变化是横向的，而体现出来的是列的增加或者减少。
Q-target目标网络，作用是一种打乱相关性的机制，使用
双模型：Q（s,a,q1)表示当前网络的输出，用来评估当前状态动作对的值函数。
 Q(s,a,q2)表示target网络的输出，可以解出targetQ并根据lossFunction更新
 https://github.com/loserChen/TensorFlow-In-Practice,此网址包含多个源代码，FM,deepFM,等及数据
 https://www.cnblogs.com/wzyj/p/8974782.htmlNCF代码及数据
 https://github.com/robi56/Deep-Learning-for-Recommendation-Systems深度学习基本的文章
 https://paste.ubuntu.com/p/jWZFjbrn2F/   DQN

'''

import numpy as np
import gym
import pymysql
import copy
import random
import time
import math
from keras.optimizers import RMSprop,Adam
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,Activation
from collections import deque
import pickle
from DQN33 import Envionment as Env
from evaluate import eval_one
EPISODES = 3000  #3000

class DQN:
    def __init__(self,state_size,action_size):
        #早期训练的模型
        self.load_model = True
        self.load_episode = 150
        self.episode_setp =6000 #单个回合最大步数#6000
        self.action_size = action_size
        self.state_size = state_size  #多少个状态
        self.memory_size = deque(maxlen=2000)
        #self.memory_size=[]
        self.train_start = 64
        self.target_update=200
        self.gamma = 0.95 #折扣因子，计算reward.
        self.epsilon = 0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.01  # 学习速率
        self.batch_size = 64  # 或32
        self.model=self.build_model()  #网络模型构建，应该是评估网络eval_net,用于预测，q_value
        self.target_model = self.build_model() #target网络构建，这个是现实网络，也就是target_net,用于预测target net
        self.updateTargetModel()  # 更新网络模型

    def build_model(self):
        model=Sequential()
        dropout=0.2   #防止过拟合
        #添加一层全连接层，输入大小为input_shape=(self.state_size),输出大小为64，激活函数为relu，权值初始化方法为lecun_uniform
        model.add(Dense(64,input_shape=(self.state_size,),activation='relu',kernel_initializer='lecun_uniform'))
        # 添加一层全连接层，输出大小为64，激活函数为relu，权值初始化方法为lecun_uniform
        model.add(Dense(64,activation='relu',kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))
        # 添加一层全连接层，输出大小为action_size，权值初始化方法为lecun_uniform
        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform')) #输出大小为action, 或自己定义的的大小
        model.add(Activation('linear')) #添加一层linear激活层
        model.compile(loss='mse',optimizer=RMSprop(lr=self.learning_rate,rho=0.9,epsilon=1e-06 ))
        return model
    #计算Q值，用到reward(当前env 回馈），done, 以及有target_net网络计算得到的next_targe
    def getQvalue(self,reward, next_target,done):
        if done:
            return reward
        else:
            return reward+self.gamma*np.amax(next_target)
    def getmovievalue(self):
        Movievalue = env.getMovievalue()  # 电影的向量值
        return Movievalue
    def choose_action(self,state):  #选择动作,以概率ϵ随机选择动作at或者通过网络输出的Q（max）值选择动作at
        movievalue=self.getmovievalue()
        state = movievalue[state]
        state = np.array([state])
        # print("choose state",state)
        if np.random.rand() <= self.epsilon:  #随机选择一个动作# 贪婪策略随机探索动作
            self.q_value = np.zeros(self.action_size)
            return random.randrange(self.action_size)  #randrange() 方法返回指定递增基数集合中的一个随机数，基数缺省值为1
        else:# 相当于从q表中选择
            q_value = self.model.predict(state) #需要的是str
            # print("q_value",q_value)
            # print('cccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',type(state),state)
            self.q_value=q_value
            return np.argmax(q_value[0])   # 输出Q(max)值选择
    def Memory(self,state,action,reward,next_state,done): #在列表中，这四个元素在列表中存为一个元素。
        movievalue = self.getmovievalue()
        state = movievalue[state]
        state = np.array(state)
        next_state=movievalue[next_state]
        next_state=np.array(next_state)
        self.memory_size.append((state,action,reward,next_state,done))
        # print('memor_size',self.memory_size)
    def updateTargetModel(self):  #更新网络参数， 将q_value网络参数给了target net.
        self.target_model.set_weights(self.model.get_weights())

    def trainModel(self,target=False):
        mini_batch = random.sample(self.memory_size,self.batch_size)
        # print('mini_batch:',np.shape(mini_batch)) #mini_batch 的shape类似为(65,4)
        ####################################################################采样出来是列表，前一个量代表采样集合，后一个是采多少样本
        #X_Qbatch,Y_Tbatch 这两个为临时变量，应该是分别存储eval_net与，target_net最后输出的值；
        X_Qbatch = np.empty((1,50),dtype=np.float64)    #存储eval_net，初始值为1行3列，创建了空的列表用来存放states,X_Qbatch = np.empty((1,3),dtype=np.float64)
        # print(X_Qbatch)#这里的数据类型是Array([[],[],[]])
        # print('X_Qbatch',np.shape(X_Qbatch))
        # print('X_Qbatch',type(X_Qbatch)) #X_Qbatch <class 'numpy.ndarray'>
        Y_Tbatch = np.empty((1,self.action_size),dtype=np.float64)  #初始化tarnet_net的值
        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]
            # 计算q_value
            print("states",states) #states <class 'numpy.ndarray'>
            a = np.array(states.reshape(1,50), dtype=X_Qbatch.dtype) #states.reshape(1,3)
            b = np.array(next_states.reshape(1,50), dtype=X_Qbatch.dtype)
            #问题出在字符串不能用reshape
            q_value = self.model.predict(a) # state
            ##############################################################
            # print("q_value predict is",q_value)
            # 计算next_target, 由下一个状态值得到
            if target:
                next_target = self.target_model.predict(b)  #若结束了，则为rj
                print("next_target1",next_target)
            else:
                next_target = self.model.predict(b) #eval_net得到的,值没做完，则更新.
                print("next_target2", next_target)
            next_q_value =self.getQvalue(rewards,next_target,dones)  #计算next_target状态的Q值
            # print("next_q_value=",next_q_value)
            X_Qbatch = np.append(X_Qbatch,a,axis=0)  #将states的值拷贝到X_Qbatch中
            # print("X_Qbatch value",X_Qbatch)      #存放的为state的向量值
            ##########################
            Y_sample = q_value.copy()     #更新target网络参数的值，将eval_net的参数赋给target网络。?
            Y_sample[0][actions] = next_q_value #不管在下一个状态中，我们取了哪个action, 都需对应上q_eval中的action位置？
            Y_Tbatch = np.append(Y_Tbatch,np.array([Y_sample[0]]),axis =0) #
            if dones:
                X_Qbatch = np.append(X_Qbatch,b,axis =0)#
                Y_Tbatch = np.append(Y_Tbatch,np.array([[rewards]*self.action_size]),axis =0) #
            # print('X_Qbatch',X_Qbatch)
            # print("Y_batch", Y_Tbatch)
        print(len(X_Qbatch),len(Y_Tbatch))
        self.model.fit(X_Qbatch,Y_Tbatch,batch_size=self.batch_size,epochs=1,verbose=0)
        print('working')
#######################################################################################

if __name__ == '__main__':
       #1.只走一步，不走了，反复循环。
       start = time.clock() #计算开始时间
       state_size=50 #3,指的是输入层的维度数
       action_size=1 #输出 若有4个属性，则为4.
       allRecoLen=[]
       # userId = "103443334"
       #*******************************************对所有的用户遍历计算
       f = open("allUserId.data", 'rb')  # movieid -> “actor”, "director", "country", "type"
       alluserId = pickle.load(f)  # dict,
       f.close()
       allUserId = []
       # userId="xiaomandu"
       # env.getUserMovie(userId)
       # env.getUserId(userId)
       averagehr5, averagendcg5, averagemap5, averageprecision5, averagecoveage5 = 0, 0, 0, 0, 0
       averagehr15, averagendcg15, averagemap15, averageprecision15, averagecoveage15 = 0, 0, 0, 0, 0
       averagehr10, averagendcg10, averagemap10, averageprecision10, averagecoveage10 = 0, 0, 0, 0, 0
       averagehr20, averagendcg20, averagemap20, averageprecision20, averagecoveage20 = 0, 0, 0, 0, 0
       averagehr, averagendcg, averagemap, averageprecision, averagecoveage = 0, 0, 0, 0, 0
       laveragehr, laveragendcg, laveragemap, laverageprecision, laveragecoveage = 0, 0, 0, 0, 0
       for userId in alluserId:
           allUserId.append(userId)
           agent=DQN(state_size,action_size)
           scores,episodes = [] ,[]
           global_step = 0
           env=Env()
           recommend_list1=[]
           recommend_list2 = []
           gtItem = list(env.getUserMovie(userId)) #每一个用户的看过的电影的id集合
           # print("gtitem",gtItem)
           # action=[0,1,2,3]    # 任选一个类型，还是给定一个类型。从导演，演员，类型，国家中 选，假如选的action为 导演。
           for episode in range(agent.load_episode+1,EPISODES): #
               if len(env.getUserMovie(userId)) == 0:   #若用户看过的电影 为 空，则break
                   print("state is empty", next_state, action, reward, done)
                   break
               state = random.choice(list(env.getUserMovie(userId)))  # 从字典的key中随机选了一个用户看过的电影id
               env.currId=[]
               print("state",state) #为用户id值
               done = False
               score = 0
               for t in range(agent.episode_setp): #每个回合循环episode_setp步
                   action=agent.choose_action(state)
                   #####################
                   # print('aaaaaaaaaa',action)#从选择用户的id中的4个状态中状态
                   print('action=',action)
                   next_state,action1,reward,done=env.step(action,state,userId) # 这儿是这样传id ，吗？在列表中，下一个next_state,这个action1没有用
                   print(next_state,action1,reward,done)
                   if next_state==-1 or next_state ==0:
                       state=-1
                       break
                   if reward == 1:
                       break
                   recommend_list1.append(next_state)
                   print('recommend_list1', recommend_list1)
                   # 结束后输出推荐系列
                   # print('recommend_list length', len(recommend_list1))
                   lhr, lndcg, lprecision, lMaPP, lcoverage = eval_one(recommend_list1, gtItem)

                   laveragehr += lhr
                   laveragendcg += lndcg
                   laveragemap += lMaPP
                   laverageprecision += lprecision
                   laveragecoveage += lcoverage
                   # lalveragehr = laveragehr * 1.0 / len(allUserId)
                   # laveragendcg = laveragendcg * 1.0 / len(allUserId)
                   # laveragemap = laveragemap * 1.0 / len(allUserId)
                   # laverageprecision = laverageprecision * 1.0 / len(allUserId)
                   # laveragecoveage = laveragecoveage * 1.0 / len(allUserId)
                   print("Allaveragehr,Allaveragendcg,Allaverageprecision,Allaveragemap,Allaveragecoveage", laveragehr, laveragendcg,
                         laverageprecision, laveragemap, laveragecoveage)
                   print("userlength",len(allUserId))
                   agent.Memory(state,action,reward,next_state,done)
                   if len(agent.memory_size) >= agent.train_start:
                        if global_step <= agent.target_update: # 前面没有target_update的定义
                            agent.trainModel()
                        else:
                            agent.trainModel(True)
                   score += reward
                   state = next_state
                   if done:
                       break
               if state==-1:
                   break

           global_step += 1
           if global_step % agent.target_update == 0:
               agent.updateTargetModel()

           if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
           print('end')
           hr5,ndcg5,precision5,MaPP5,coverage5=eval_one(recommend_list1[:5],gtItem)
           hr10, ndcg10, precision10, MaPP10, coverage10 = eval_one(recommend_list1[:10], gtItem)
           hr15, ndcg15, precision15, MaPP15, coverage15 = eval_one(recommend_list1[:15], gtItem)
           hr20, ndcg20, precision20, MaPP20, coverage20 = eval_one(recommend_list1[:20], gtItem)
           hr, ndcg, precision, MaPP, coverage = eval_one(recommend_list1, gtItem)
           allRecoLen.append(len(recommend_list1))
           #*********************************************************求推荐前5，10的平均值,正确否？
           averagehr5 += hr5
           averagendcg5 += ndcg5
           averagemap5 += MaPP5
           averageprecision5 += precision5
           averagecoveage5 += coverage5
           #********************************************************求推荐前10的平均值
           averagehr10 += hr10
           averagendcg10 += ndcg10
           averagemap10 += MaPP10
           averageprecision10 += precision10
           averagecoveage10 += coverage10
           # ********************************************************求推荐前15的平均值
           averagehr15 += hr15
           averagendcg15 += ndcg15
           averagemap15 += MaPP15
           averageprecision15 += precision15
           averagecoveage15 += coverage15
           # ********************************************************求推荐前20的平均值
           averagehr20 += hr20
           averagendcg20 += ndcg20
           averagemap20 += MaPP20
           averageprecision20 += precision20
           averagecoveage20 += coverage20
           # ********************************************************求推荐的平均值
           averagehr += hr
           averagendcg += ndcg
           averagemap += MaPP
           averageprecision += precision
           averagecoveage += coverage

       averagehr5 = averagehr5 * 1.0 / len(allUserId)
       averagendcg5 = averagendcg5 * 1.0 / len(allUserId)
       averagemap5 = averagemap5 * 1.0 / len(allUserId)
       averageprecision5 = averageprecision5 * 1.0 / len(allUserId)
       averagecoveage5 = averagecoveage5 * 1.0 / len(allUserId)
       averagehr10 = averagehr10 * 1.0 / len(allUserId)
       averagendcg10 = averagendcg10 * 1.0 / len(allUserId)
       averagemap10 = averagemap10 * 1.0 / len(allUserId)
       averageprecision10 = averageprecision10 * 1.0 / len(allUserId)
       averagecoveage10 = averagecoveage10 * 1.0 / len(allUserId)
       averagehr15 = averagehr15 * 1.0 / len(allUserId)
       averagendcg15 = averagendcg15 * 1.0 / len(allUserId)
       averagemap15 = averagemap15 * 1.0 / len(allUserId)
       averageprecision15 = averageprecision15 * 1.0 / len(allUserId)
       averagecoveage15 = averagecoveage15 * 1.0 / len(allUserId)
       averagehr20 = averagehr10 * 1.0 / len(allUserId)
       averagendcg20 = averagendcg10 * 1.0 / len(allUserId)
       averagemap20 = averagemap10 * 1.0 / len(allUserId)
       averageprecision20 = averageprecision10 * 1.0 / len(allUserId)
       averagecoveage20 = averagecoveage10 * 1.0 / len(allUserId)
       averagehr = averagehr * 1.0 / len(allUserId)
       averagendcg = averagendcg * 1.0 / len(allUserId)
       averagemap = averagemap * 1.0 / len(allUserId)
       averageprecision = averageprecision * 1.0 / len(allUserId)
       averagecoveage = averagecoveage * 1.0 / len(allUserId)


       print("averagehr5,averagendcg5,averageprecision5,averagemap5,averagecoveage5", averagehr5, averagendcg5,
             averageprecision5, averagemap5, averagecoveage5)
       print("averagehr10,averagendcg10,averageprecision10,averagemap10,averagecoveage10", averagehr10, averagendcg10,
             averageprecision10, averagemap10, averagecoveage10)
       print("averagehr15,averagendcg15,averageprecision15,averagemap15,averagecoveage15", averagehr15, averagendcg15,
             averageprecision15, averagemap15, averagecoveage15)
       print("averagehr20,averagendcg20,averageprecision20,averagemap20,averagecoveage20", averagehr20, averagendcg20,
             averageprecision20, averagemap20, averagecoveage20)
       print("averagehr,averagendcg,averageprecision,averagemap,averagecoveage", averagehr, averagendcg,
             averageprecision, averagemap, averagecoveage)
       with open('allRecommendList11new.data', 'ab') as f:
           pickle.dump(allRecoLen, f)
           f.close()
       end = time.clock()
       print('finish all in %s' % str(end - start))































