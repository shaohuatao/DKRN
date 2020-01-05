import pickle
import collections
import json
import gensim
import pymysql
import collections
from numpy import *
import pickle
import numpy as np
import json
import random
import cmath
'''raise IndexError('Cannot choose from an empty sequence') from None
IndexError: Cannot choose from an empty sequence,偶尔出现的错误
这个程序是计算只有导演，演员，类型的三种情况的值。

'''
class Envionment:
    def __init__(self):
        self.currId=[]
    def getUserId(self,userId):
        db = pymysql.connect("localhost", "root", "root", "test")
        cursor = db.cursor()
        sql = "select collect_type from user where user_id='"+userId+"'"
        cursor.execute(sql)
        results = cursor.fetchall()
        keylist = []
        ratelist = {}
        colist = {}
        for row in results:
            coll = row[0]  # str类型
            colist = json.loads(coll.replace("'", "\""))  # json loads 将str转换为字典格式, 将用户看过的电影id 取出。
        # print(colist)
        UCMid = list(colist.keys())  # 用户看过的电影id 集合。
        # print("UCMID",UCMid)
        return UCMid

    def getUserMovie(self,userid):  #用户已看过的电影信息分类，如id: 导演，演员，类型
        userId = self.getUserId(userid)
        db = pymysql.connect("localhost", "root", "root", "test")
        CollectMovie = collections.defaultdict(list)
        cursor = db.cursor()
        for i in userId:
            sql = "select id,directors,actors,type,country from movienew where id=%s" % (i)  # 格式化输出
            cursor.execute(sql)
            results = cursor.fetchall()
            information = {}
            for row in results:
                information['director'] = []
                director = row[1].split("/")
                for di in director:
                    di = str.strip(di)
                    information['director'].append(di)
                    # print(information['director'])
                information['actor'] = []
                actor = row[2].split("/")
                for ac in actor:
                    ac = str.strip(ac)
                    information['actor'].append(ac)
                information['type'] = []
                type = row[3].split("/")
                for tc in type:
                    tc = str.strip(tc)
                    information['type'].append(tc)
                information['country'] = []
                country = row[4].split("/")
                for coun in country:
                    coun = str.strip(coun)
                    information['country'].append(coun)
                    CollectMovie[i] = information
        # print("collectmovie", CollectMovie)
        return CollectMovie

    def getUserWishMovie(self,userId):  #用户想看的电影id集合
        db = pymysql.connect("localhost", "root", "root", "test")
        cursor = db.cursor()
        sql = "select wish_type from user where user_id='"+userId+"'"
        cursor.execute(sql)
        results = cursor.fetchall()
        keylist = []
        ratelist = {}
        colist = {}
        for row in results:
            coll = row[0]  # str类型
            colist = json.loads(coll.replace("'", "\""))  # json loads 将str转换为字典格式, 将用户看过的电影id 取出。
        # print(colist)
        UserWishMovie = list(colist.keys())  # 用户想看的电影id 集合。
        UserWishMovie=UserWishMovie[:5]
        print("userwishMovie",UserWishMovie)
        # output = open('WCMid.data','wb')
        # pickle.dump(WCMid,output)
        # print(WCMid)
        return UserWishMovie
    def getAllMovie(self):  #全部电影信息，根据id分类
        db = pymysql.connect("localhost", "root", "root", "test")
        cursor = db.cursor()
        sql = "select id,directors,actors,type,country from movienew" #movienew
        cursor.execute(sql)
        results = cursor.fetchall()
        AllMovie = collections.defaultdict(list)
        # with open("information","wb")as f: 这个是操作文本内容的时候用的。
        # pickle是对数据进行序列化，反序列化用的，永久存储。
        for row in results:
            inoformation = {}
            id = row[0]
            # id = int(id)
            directorinfo = row[1].split("/")  # inoformation['director'] = row[1].split("/") # split 返回分割后的字符串列表
            inoformation['director'] = []
            for info in directorinfo:
                info = str.strip(info)
                inoformation['director'].append(info)
            # print(inoformation['director'])
            actorinfo = row[2].split("/")  # inoformation['actor']=row[2].split("/")
            inoformation['actor'] = []
            for actor in actorinfo:
                actor = str.strip(actor)
                inoformation['actor'].append(actor)
            # print(inoformation['actor'])
            inoformation['type'] = []
            typeinfo = row[3].split("/")  # 等同于：inoformation['type']=row[3].split("/")
            # print(typeinfo)
            for v in typeinfo:  #
                v = str.strip(v)
                inoformation['type'].append(v)
            # print(inoformation['type'])
            countryinfo = row[4].split("/")
            inoformation['country'] = []
            for country in countryinfo:
                country = str.strip(country)
                inoformation['country'].append(country)
            # print(inoformation['country'])
            AllMovie[id] = inoformation  # id 为key, information为value.
        # print("AllMovieList", AllMovie)
        # output = open('idinformation.data', 'wb')
        # pickle.dump(data, output)
        print("processing...")
        return AllMovie
    def getMovievalue(self): #全部电影的向量值, id:向量值，以字典方式存储的
        f = open("Allmovievalue.data", 'rb')
        AllmovieValue = pickle.load(f)
        f.close()
        return AllmovieValue

    def step(self,action,next,userId):

        AllMovieValue = self.getMovievalue() #获取电影的向量值

        if next in self.getUserMovie(userId).keys():
            infor = self.getUserMovie(userId)[next]
        else:
            infor=self.getAllMovie()[next]
        #print(infor)
        dir1 = infor['director']
        print('dir导演', dir1)
        actor1 = infor['actor']
        print('actor演员',
              actor1)  # actor有几种不同的情况：1.一个演员的集合列表{'巩俐': ['1306505', '1292365']}),2.多个演员有不同的电影列表:{'古天乐': ['25858785', '26035290'], '廖启智': ['26425063'], '张松枝': ['26425063']})
        type1 = infor['type']
        print('type类型', type1)
        country1 = infor['country']
        print('country国家', country1)
        #8888888888888888888888888888888888888888888
        AllMovie = self.getAllMovie()   #获取所有的电影
        LocalMovieList = collections.defaultdict(list)
        if action == 0:
            print('action导演=', action)
            for key, val in AllMovie.items():
                for value in dir1:
                    if value in val['director']:  # 同一导演的,导演中会出现只有一个导演的情况，这种
                        LocalMovieList[value].append(key)  # {'张艺谋': ['4864908', '1306505', '1292365']}

        elif action == 1:     
              print('action演员=',action)
              for key, val in AllMovie.items():
                  for a1 in actor1:          #同一演员的，{'古天乐': ['25858785', '26035290'], '廖启智': ['26425063'], '张松枝': ['26425063']})
                    # print('a1',a1)
                      if a1 in val['actor']:
                        # print('value',value)
                            LocalMovieList[a1].append(key)
        # elif action==2:
        #       print('action类型=',action)
        #       for key, val in AllMovie.items():
        #           for t1 in type1:                   #同一类型的：{'动作': ['25858785', '4864908', '25882296', '26425063', '26035290'], '剧情': ['4864908', '26425063', '26035290', '1306505', '1292365'], '犯罪': ['26425063']})                                                        # 多个type类型，从中任选一个类型，并根据这个类型，返回下一个nextstate, 同理，这个nextstate即为任选出的一个类型的最大值。将这个功能放在下一方法中
        #                if t1 in val['type']:
        #
        #                     LocalMovieList[t1].append(key)
        # else:
        #     print('action国家=', action)
        #     for key, val in AllMovie.items():
        #
        #         for counr in country1:              #同一国家的
        #             if counr in val['country']:
        #                     LocalMovieList[counr].append(key)

        UserWishMovie = self.getUserWishMovie(userId)  # 用户想看的电影集合（3）用户想看的电影集合, 这儿要加一个判断吧？？？？，若用户想看的电影为空，则break, 否则，程序会陷入死循环
        if len(UserWishMovie)==0:
            print("aaaaa");
            nextstate=-1
            reward = 0
            action = 0
            done = False
            print("ffffff", nextstate, action, reward, done)
            return nextstate, action, reward, done

        if len(LocalMovieList)>0:
            for key,value in LocalMovieList.items():
                print("LocalMoviekey=",key+","+ "Localmovievalue=",value)# key为导演，演员，国家或类型的名字，value为电影id
                for v in value: #对电影id进行遍历
                    print(v)
                    if v in UserWishMovie:
                        reward = 1
                        done = True
                        nextstate=v
                        print(nextstate)
                        return nextstate, action, reward, done
            nextstate=self.getNextState(next,LocalMovieList)
            reward=0
            action=0
            done=False
            print("rrrrr",nextstate, action, reward, done)
            return nextstate, action, reward, done
        else:
            done=False
            reward = 0
            action=action+1  #为什么用action+1呢？，选下一条边
            nextstate = next  #指向它自己（1）指向它自己, 这儿有问题。？？？？？？ recommend_list1 ['25964071', '27605698', '25964071', '27605698']，就在两个值上循环，不再往下走。
            if action>1:      # action>1  循环加超过3,还是没有下一个点，再重选一个用户。
                nextstate=random.choice(list(self.getUserMovie(userId)))   #此处有env.getUserMovie()改为self.getUserMovie()
                action=0
            print("bbbbbbbbb",nextstate, action, reward, done)
            return nextstate, action, reward, done

        print("locallist is",LocalMovieList)

        # print('userwishmovei',UserWishMovie)
        # print(type(UserWishMovie))
        score = 0
        if nextstate in UserWishMovie:
            reward = 1
            done = True
            print('reward1', reward)
        else:
            reward = 0
            done = False
            print('reward2', reward)

        return nextstate,reward,done
    def getNextState(self,moveId,movieList):  #每次的下一个节点要与AllMovieValue中的值去比较，不要包含它自身的比较，要不然，它的值永远是最大，只在这个圈子中兜转。
        # movieList=(list(set(movieList).difference(set(moveId))))  #新加的，去掉自身，与其它节点进行对比相似值
        AllMovieValue=self.getMovievalue()
        key1 = AllMovieValue[moveId]
        A = 0.0
        SumA = 0.0
        for k1 in key1:
            SumA = SumA + k1 * k1
        A = np.sqrt(SumA)
        SumB = 0.0
        B = 0.0
        Maxstate = collections.defaultdict(list)  #这儿是空值
        # 1.计算前先进行查询，看最大值有否在字典中，若有，则不用计算，直接取，否则计算。
        print("LocalMovieList size", len(movieList))  #若只有一个值，如导演
        # LocalMovieList: {'动作': ['25858785', '4864908', '25882296', '26425063', '26035290'], '剧情': ['4864908', '26425063', '26035290', '1306505', '1292365'], '犯罪': ['26425063']})
        for movies in movieList.values():  #
            # print("moviesvalues",movies)
            for movie in movies:
                # print("movie=\'",movie,"\'")
                # print("Maxstate",Maxstate)
                if movie in Maxstate.keys() and Maxstate[movie]>0:
                    Maxstate[movie]=Maxstate[movie]*1.5
                else:
                    if movie in AllMovieValue and movie!=moveId and movie not in self.currId:  #这儿有问题???  避免节点只选它自己，如果它也在全部电影中，则与它自己的匹配值最大，总是选它自己，但是总是循环为一个nextstate['27615233', '20356411']
                        key2 = AllMovieValue[movie]
                        for k2 in key2:
                            SumB = SumB + k2 * k2
                        B = np.sqrt(SumB)
                        x = np.dot(np.array(key1), np.array(key2)) / A * B
                        Maxstate[movie] = x
                        self.currId.append(movie)
                    else:
                        print("error")
        print('max',Maxstate)  # 结果：max defaultdict(<class 'list'>, {'4864908': 0.061599999999999995, '1306505': 0.454, '1292365': 0.1108})
        if len(Maxstate) ==0:
            return 0
        nextstate = max(Maxstate,key=Maxstate.get)
        print('nextstate is', nextstate)
        return nextstate








