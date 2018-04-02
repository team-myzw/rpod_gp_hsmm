#!/usr/bin/env python
# encoding: utf8
#from __future__ import unicode_literals
import GaussianProcessMiltiDim
import random
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
from threading import Lock
import time
import numpy
import glob
import os
import pylab
import gc
import copy
import tf
from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed,RotatingMarker, Timer
import rospy
from geometry_msgs.msg import Point
from joint_sampler import JointBagReader
from rosbag_database.srv import RosbagPlay, RosbagPlayRequest
from rpod_gp_hsmm.srv import JointBag, JointBagRequest, JointBagResponse
from rpod_gp_hsmm.srv import JointRecog, JointRecogResponse
from sampling_joint import sampling
from sparse import sparse_dim_n
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
import objects_from_db
import multiprocessing
import pandas as pd

USE_JOINT = ["x", "y", "z", "hand", "qx", "qy", "qz", "qw",
                      "power_x", "power_y","power_z"]#, "power"]
#                      "power_x_raw", "power_y_raw","power_z_raw", "power_raw",
TEST= False
TEST_FILE="test/"
TIME = 20
SP = 1
SP_DF=0.1
RATE = 1.0
DIM = 8#len(USE_JOINT)
SCORE = 0.75
#COR = [0,0,0,0,0,0,0,0,1,2,2,2,2]
NUMBER=0
COR = [0,0,0,0,1,2,2,2,2]
#COR = [0,0,0,0,2,2]
NCLASS = len(COR)
SAVE = "./save/exp_{0:f}/"#_{1:d}_{2:d}_{3:d}_{4:d}/"
DIST = 0.2
OBJ_DIST=0.5
DIS = 0.2
RECOG = "./recog/{}"
MAX = 700
MIN = -700
SPA=1
MINIX = -32700
ITE=10
GET_TIME=40
GET_DATA=False
#C_BETA=[[2.5,2.5,2.5,10.0,2.5,2.5,2.5,2.5],
#        [2.5,10.0,10.0,10.0,5.0,5.0,5.0,5.0],
#        [10.0,10.0,10.0,10.0,10.,10.,10.,10.]]
C_BETA=[[2.5,2.5,10.,10.,2.5,2.5,2.5,2.5],
        [10.0,10.0,10.,10.,10.0,10.0,10.0,10.],
        [10.0,10.0,10.,10.,10.0,10.0,10.0,10.],
        [10.0,10.0,10.,10.,10.0,10.0,10.0,10.]]

#CATEGORY=["pettbotle", "door", "fridge","box"]
CATEGORY=["button","table","fruit","doll","drink"]
OBJECTS={"pettbotle":[0,1,2,3,4,5,6],
         "box":[7],
         "door":"recognized_object/exp_door1/15",
         "fridge":"recognized_object/exp_friedge/19",
         "fruit": ["exp_fruit_0","exp_fruit_1"],
         "doll": ["exp_doll_2","exp_doll_3","exp_doll_4"],
         "drink": ["exp_drink_5","exp_drink_6","exp_drink_7"],
         "button": "exp_button_8",
         "table": ["exp_table_9","exp_table_10"],
         "object": ["exp_fruit_0","exp_fruit_1","exp_doll_2","exp_doll_3","exp_doll_4","exp_drink_5","exp_drink_6","exp_drink_7"],
         "equipment":["recognized_object/exp_door1/15","recognized_object/exp_friedge/19"]
         }

"""
Cythonのコンパイルできないときは，

  E:\Python27_64\Lib\distutils\msvc9compiler.py

のget_build_version()の

  majorVersion = int(s[:-2]) - 6

を使いたいコンパラのバージョンに書き換える．
VC2012の場合は

 majorVersion = 11
"""
def dump():
    import sys
    sys.exit(1)
    
def rotM(p):
    # 回転行列を計算する
    px = p[0]
    py = p[1]
    pz = p[2]
     #物体座標系の 3->2->1 軸で回転させる
    Rx = numpy.array([[1, 0, 0],
                    [0, numpy.cos(px), numpy.sin(px)],
                    [0, -numpy.sin(px), numpy.cos(px)]])
    Ry = numpy.array([[numpy.cos(py), 0, -numpy.sin(py)],
                    [0, 1, 0],
                    [numpy.sin(py), 0, numpy.cos(py)]])
    Rz = numpy.array([[numpy.cos(pz), numpy.sin(pz), 0],
                    [-numpy.sin(pz), numpy.cos(pz), 0],
                    [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)       
    return R

def calctransform(p,B,sP_B):
#    p = np.array([np.pi,np.pi/2, np.pi/3])
    R = rotM(p)
    A = R.T
    
    O = numpy.array([0, 0, 0])
    sP_O = numpy.dot(A, sP_B)
    rB_O = B - O
    rP_O = rB_O + sP_O
    return rP_O

def get_dataframe(df,i):
    land = df.loc[(df.id == i)]
    return land

        
    
def df2land(object_info):
    if len(object_info) != 0:       
        x = object_info.position_x.values
        y = object_info.position_y.values
        z = object_info.position_z.values
        ix = object_info.id.values
        return [x[0], y[0], z[0], ix[0]]
    else:
        return [0.,0.,0.,-1]
import numpy as np
def distance_check(position1, position2):
    pos1 = position1[0:3]
    if len(position2) >=1:
        pos2 = position2[:,0:3]
        dis2 = np.power(pos2[:,0]-pos1[0],2)+numpy.power(pos2[:,1]-pos1[1],2)+numpy.power(pos2[:,2]-pos1[2],2)
        dis = np.sqrt(dis2[0])
#    elif len(position2) == 1:
#        pos2 = position2[:,0:3]
#    
#        dis = numpy.sqrt(numpy.power(pos2[0]-pos1[0],2)+numpy.power(pos2[1]-pos1[1],2)+numpy.power(pos2[2]-pos1[2],2))
    else:
        return True
    check = dis[DIS>=dis]
    if len(check) == 0:
        return True
    else:
        return False

class GPSegmentation(object):
#    MAX_LEN = 17
#    MIN_LEN = 8
#    AVE_LEN = 11
    MAX_LEN = 11
    MIN_LEN = 6
    AVE_LEN = 9
    SKIP_LEN = 1
    MIN_STATE = 3
    CORD_TRA = 0
    CORD_LAND1 = 1
    CORD_LAND2 = 2
    CORD_LAND3 = 3
    CORD_MOV = 4
    FIELD_CORD = 5

    def __init__(self, ):#dim, nclass, cordinates):
        rospy.Service("joint_upload{}".format(NUMBER), JointBag, self.get_data)
        rospy.Service("start_gphsmm_learn{}".format(NUMBER),Trigger, self.learn_start)
        rospy.Service("start_gphsmm_recog",Trigger, self.recog_start)
        rospy.Service("start_gphsmm_recog_simple{}".format(NUMBER),Trigger, self.start_recognition)
        self._set_state() 

    def _set_state(self,):
        self.AVE_LEN = 9#numpy.random.randint(6,13)
        self.MIN_LEN = 5#self.AVE_LEN - numpy.random.randint(2,5)
        self.MAX_LEN = 13#self.AVE_LEN + numpy.random.randint(2,5)
        self.bag_name=[]
        self.dim = DIM

#        self.t_n = 0#numpy.random.randint(1,3)
#        self.h_n = 1#numpy.random.randint(2,4)
#        self.t_n = [0,0,0,0]
#        self.h_n = [2,1,1,2]#numpy.random.randint(2,4,len(CATEGORY))
#        self.h1_n = [2,0,0,2]#numpy.random.randint(0,3,len(CATEGORY))
#        self.h2_n = [0,4,4,0]#numpy.random.randint(2,5,len(CATEGORY))
        self.t_n =  [0,0,0,0,0]
        self.h_n =  [2,2,2,1,1]#numpy.random.randint(2,4,len(CATEGORY))
        self.h1_n = [1,1,1,0,2]#numpy.random.randint(0,3,len(CATEGORY))
        self.h2_n = [0,0,0,2,0]#numpy.random.randint(2,5,len(CATEGORY))
        
        cor = [[0]*self.t_n[i] + [1] * self.h1_n[i] + [2] * self.h_n[i] + [3] * self.h2_n[i] for i in range(len(CATEGORY))]
        self.cordinates = cor
        self.cord_beta = C_BETA
        self.numclass = [len(cor[i]) for i in range(len(CATEGORY))]
        self.segmlen = 3
        self.gps = [[GaussianProcessMiltiDim.GPMD(self.dim,self.cord_beta[self.cordinates[c][i]])
                    for i in range(self.numclass[c])] for c in range(len(CATEGORY))]
        
        self.segm_in_class = [[[] for i in range(self.numclass[j])] for j in range(len(CATEGORY))]
        self.segmclass = [{} for i in range(len(CATEGORY))]  
        self.segmlandmark = [{} for i in range(len(CATEGORY))]
        self.segments = [[]for i in range(len(CATEGORY))]
        self.landmarks = [[]for i in range(len(CATEGORY))]
        self.landmark_lists = [[]for i in range(len(CATEGORY))]
        self.trans_prob = [ numpy.ones((self.numclass[i], self.numclass[i])) for i in range(len(CATEGORY))]
        self.trans_prob_bos = [ numpy.ones(self.numclass[i]) for i in range(len(CATEGORY))]
        self.trans_prob_eos = [ numpy.ones(self.numclass[i]) for i in range(len(CATEGORY))]
        self.is_initialized = [ False for i in range(len(CATEGORY))]
        self.land_choice = [[] for i in range(len(CATEGORY))]
        self.jbr = JointBagReader()
        self.data = []
        self.names = []
        self.joint_states = []
        self.joint_state_stamp = []
        self.time_list = []


        self.object_list = [{} for i in range(len(CATEGORY))]
        self.field_list = [[] for i in range(len(CATEGORY))]

        self.gp_time = [ 0.0 for i in range(len(CATEGORY))]
        self.cordinate_time = [ 0.0 for i in range(len(CATEGORY))]
        self.outprob_time = [ 0.0 for i in range(len(CATEGORY))]
        self.alpha_time = [ 0.0 for i in range(len(CATEGORY))]
        self.land_choice_time = [ 0.0 for i in range(len(CATEGORY))]
        self.land_get_time = [ 0.0 for i in range(len(CATEGORY))]
        self.tf_r_time = [ 0.0 for i in range(len(CATEGORY))]
        self.tf_q_time = [ 0.0 for i in range(len(CATEGORY))]
        self.cordinate_num = [ 0 for i in range(len(CATEGORY))]
        self.cordinate_roq =[ 0.0 for i in range(len(CATEGORY))]
        self.cordinate_set = [ 0.0 for i in range(len(CATEGORY))]
        self.cordinate_buf = [ {} for i in range(len(CATEGORY))]
        self.lands_buf = [ {} for i in range(len(CATEGORY))]
        self.number = [0 for i in range(len(CATEGORY))]
        self.svdir = ["" for i in range(len(CATEGORY))]
        rospy.loginfo("get_database")
        self.object = objects_from_db.Objects()
#        rospy.loginfo("alrady start class {0:d} {1:d} {2:d} {3:d}".format(self.t_n,self.h1_n,self.h_n,self.h2_n)) 
        rospy.loginfo("len {0:d},{1:d},{2:d}".format(self.MIN_LEN,self.AVE_LEN,self.MAX_LEN))
        
    def get_data(self, data):
        req = RosbagPlayRequest()
        req.name = data.name
        self.bag_name.append(data.name.split("/")[-1])
        req.rosbag_time = data.rosbag_time
        req.start_time = data.start_time
        req.end_time = data.end_time
        req.count_number = data.count_number
        req.duration = data.duration_secs
        req.topics = data.read_topics[0]
        s = rospy.Time.now().to_sec()
        j = self.jbr.data_get(req, JointState())
        rospy.loginfo(rospy.Time.now().to_sec() - s)
        if j != None:
            self.names = j[0].name
            number = len(j)
            st,et = self.get_time_start_end_raw(j)
            sp = number/((float(et - st))/1000000000.0)/float(RATE)
            sp_joint_data = sparse_dim_n(j, int(sp))
            rospy.loginfo("load")
            stamps = self.get_stamp(sp_joint_data)
            self.joint_states.append(numpy.array(sp_joint_data,dtype="object"))
            self.joint_state_stamp.append(numpy.array(stamps,dtype="object"))
            self.time_list.append(st)
            self.time_list.append(et)
#            o,_o = self.get_object_from_time(0,sp_joint_data)
        res = JointBagResponse()
        if not None in [j]:
            res.success = True
            rospy.loginfo("I have {0:d} {1:d} length data".format(number,len(sp_joint_data)))
#            rospy.loginfo("{0:d} object in data".format(len(o)))
            rospy.loginfo("Now {0:d} data".format(len(self.joint_states)))
        return res
    
    def get_operation_object_data(self,):
        min_time = min(self.time_list)
        max_time = max(self.time_list)
        self.object.divid_df(min_time, max_time)
        self.df_data = []
        self.df_times = []
        self.index = []
        self.df_names = []
        for category in CATEGORY:
            score = SCORE
            param = OBJECTS[category]
            if category == "pettbotle":
                score = 0.95
            if category == "object":
                score = 0.9
            df_data, df_times, index, df_names = self.object.get_df_raw_data_extract_recogobj(param, score)
            self.df_data.append(df_data)
            self.df_times.append(df_times)
            self.index.append(index)
            self.df_names.append(df_names)
            
    def check_active_object(self,obj_data,obj_times,joint_list):
        n = len(obj_data)
        nums = []
        joint_data,times = self.get_joint_states_position_and_time(joint_list)
        count = 0
        ms = []
        for i in range(n):
            t = obj_times[i]
            joints = joint_data[numpy.where(((times >= t)&(times <= t+TIME)))[0],:]
            x = obj_data[i][0]
            y = obj_data[i][1]
            z = obj_data[i][2]
            jx = joints[:,0]
            jy = joints[:,1]
            jz = joints[:,2]
            if len(joints)==0:
                count +=1
                continue
            _dis = numpy.power(jx-x,2)+numpy.power(jy-y,2)+numpy.power(jz-z,2)
            dis = numpy.sqrt(_dis)            
            min_dis = numpy.min(dis)
            if min_dis<OBJ_DIST:
                ms.append(min_dis)
                nums.append(i)
        if len(nums)==0:
            import sys
            print "no"
            sys.exit(1)
        print count , n
        return nums
    
    def extract_object(self):
        joints = None
        for j in range(len(self.joint_states)):
            if np.all(joints == None):
                joints = self.joint_states[j]
            else:
                joints = numpy.c_[joints.reshape(1,-1),self.joint_states[j].reshape(1,-1)]
        joints = joints[0]
        for i in range(len(CATEGORY)):
            ob_data = self.df_data[i]
            ob_time = self.df_times[i]
            nums = self.check_active_object(ob_data,ob_time,joints)
            print len(self.df_data[i]),
            self.df_data[i] = self.df_data[i][nums,:]
            print len(self.df_data[i])
            self.df_times[i] = self.df_times[i][nums]
            self.index[i] = self.index[i][nums]
            self.df_names[i] = self.df_names[i][nums]

    def get_joint_states_position_and_time(self, joint_state_list):
        name = joint_state_list[0].name
        names = []
        times = []
        for j in range(len(name)):
            if not name[j] in USE_JOINT:
                names.append(j)
        position = []
        for joint_state in joint_state_list:
            p = joint_state.position
            t = joint_state.header.stamp.to_sec()
            times.append(t)
            position.append(p)        
        pos = numpy.array(position)        
        if names != []:
            pp = numpy.delete(pos, names, 1)
        return pp, times

    def check_active_time(self,):
        joints = None
        md = []
        for j in range(len(self.joint_states)):
            if np.all(joints) == None:
                joints = self.joint_states[j]
            else:
                joints = numpy.c_[joints.reshape(1,-1),self.joint_states[j].reshape(1,-1)]
        joints =joints[0]
        _joint_data,times = self.get_joint_states_position_and_time(joints)
        jc_d = []
#        jc_s = []
        for i in range(len(CATEGORY)):
            nums = []
            j_d = []
#            j_s = []
            for n in range(len(self.df_data[i])):
                if nums == []:
                    nums.append(n)
                else:
                    if (self.df_times[i][n] - self.df_times[i][n-1]) <= TIME and (self.df_times[i][n] - self.df_times[i][n-1]) > 0:
                        nums.append(n)
                        if n == len(self.df_data[i])-1:
                            st = self.df_times[i][nums[0]]
                            et = self.df_times[i][nums[-1]]+TIME
                            j = joints[numpy.where(((times >= st)&(times <= et)))[0]]
                            _j,_t = self.get_joint_states_position_and_time(j)
                            jx = _j[:,0]
                            jy = _j[:,1]
                            jz = _j[:,2]
                            lm = []
                            for nn in nums:
                                x = self.df_data[i][nn,0]
                                y = self.df_data[i][nn,1]
                                z = self.df_data[i][nn,2]
                                _dis = numpy.power(jx-x,2)+numpy.power(jy-y,2)+numpy.power(jz-z,2)
                                dis = numpy.sqrt(_dis)            
                                _min_dis = numpy.min(dis)
                                lm.append(_min_dis)
                            min_dis = numpy.min(lm)
                                
                            nums = []
                            nums.append(n)
                            if min_dis < OBJ_DIST:
                                md.append(min_dis)
                                j_d.append(j)
                            else:
                                pass
                            
                    else:
                        st = self.df_times[i][nums[0]]
                        et = self.df_times[i][nums[-1]]+TIME
                        j = joints[numpy.where(((times >= st)&(times <= et)))[0]]
                        _j,_t = self.get_joint_states_position_and_time(j)
                        try:
                            jx = _j[:,0]
                            jy = _j[:,1]
                            jz = _j[:,2]
                        except:
                            print _j
                            dump()
                        lm = []
                        for nn in nums:
                            x = self.df_data[i][nn,0]
                            y = self.df_data[i][nn,1]
                            z = self.df_data[i][nn,2]
                            _dis = numpy.power(jx-x,2)+numpy.power(jy-y,2)+numpy.power(jz-z,2)
                            dis = numpy.sqrt(_dis)            
                            _min_dis = numpy.min(dis)
                            lm.append(_min_dis)
                        min_dis = numpy.min(lm)
                        nums = []
                        nums.append(n)
                        if min_dis < OBJ_DIST:
                            md.append(min_dis)
                            j_d.append(j)
                        else:
                            pass
            jc_d.append(numpy.array(j_d,dtype="object"))
        self.joint_states = jc_d
    def get_dataframe2(self,number, df,i):
        ix = numpy.where(self.index[number] == i)[0]
        d = self.df_data[number][ix]
        return d
    def get_dataframename(self,i):
        ix = numpy.where(self.index == i)[0]
        name = self.df_names[ix]
        return name

    def df2land2(self,land):
        l = land[0]
        return l

    def first_land(self, segm, lands):
        n = len(lands)
        length_list = [0.] * n
        for i in range(n):
            if lands != 0 and lands[i][-1] == -1:
                continue
            len_sum = 0.0
            for s in segm:
                length = numpy.sqrt(numpy.power(s[0] - lands[i][0],2)+numpy.power(s[1] - lands[i][1],2)+numpy.power(s[2] - lands[i][2],2))
                len_sum += length
            length_list[i] = len_sum
        num = numpy.argmin(length_list)
        return lands[num]

    def dist_land(self, s, lands):
        n = len(lands)
        lands_list = []
        s = numpy.array(s)
        ss = s.T
        dis = DIST
        f_list = []
        for i in range(n):
            lx = numpy.power(ss[0] - lands[i][0],2)
            ly = numpy.power(ss[1] - lands[i][1],2)
            lz = numpy.power(ss[2] - lands[i][2],2)
            leng = lx+ly+lz
            f_lengths = numpy.sqrt(leng)
            f_length = numpy.min(f_lengths)
            if f_length <= dis:
                lands_list.append(i)
            f_list.append(f_length)
        if lands_list==[]:
#            print numpy.min(f_list)
            pass
        return lands_list


    def get_stamp(self, ros_data_list):
        stamp_list = []
        try:
            rospy.loginfo(ros_data_list[0].header.stamp)
        except:
            rospy.logerr("data has no Time stamp")
        for data in ros_data_list:
            stamp_list.append(data.header.stamp)

        return stamp_list

    def load_data(self,number,classfile=None):
        self.segments[number] = []
        self.is_initialized[number] = False
        for land_names in range(len(self.joint_states[number])):
            land_list = []
            for land in [0]:
                land_list.append([0] * DIM)
            self.land_choice[number].append(range(len(land_list)))
            self.landmark_lists[number].append(land_list)
        k = 0
#        print number
        for oj in range(len(self.joint_states[number])):
            y = self.joint_states[number][oj]
            segm = []
            """
            # ランダムに切る
            for i in range(len(y)/self.segmlen):
                segm.append( y[i*self.segmlen:i*self.segmlen+self.segmlen] )

            # 余りがあるか？
            remainder = len(y)%self.segmlen
            if remainder!=0:
                segm.append( y[len(y)-remainder:] )

            self.segments.append( segm )
            """

            i = 0
            joint_list = []
            while i < len(y):
                length = random.randint(self.MIN_LEN, self.MAX_LEN)
                    
#                lands, _ = self.get_object_from_time2(number,k, y[i: i+length + 1])
#                length_c1 = self.check_quat(self.get_joint_states_position(y[i: i+length + 1]),lands)
#                length_c3 =self.dist_land(self.get_joint_states_position(y[i: i+length + 1]), lands)                
#                if length_c1 == [] and length_c3==[]:
#                    continue
                if i + length + 1 >= len(y):
                    length = len(y)-i
                segm.append(self.get_joint_states_position(y[i: i+length + 1]))
                joint_list.append(y[i: i+length + 1])
            
                i += length

            self.segments[number].append(segm)
            
            # ランダムに割り振る
            land_list = []
#            p_bar = ProgressBar(maxval=len(segm)).start()
#            print "check"
            
            for i, s in enumerate(segm):
#                p_bar.update(p_bar.currval + 1)
                lands, _ = self.get_object_from_time2(number, k, joint_list[i])
                while not rospy.is_shutdown():
                    c = random.randint(0, self.numclass[number] -1)
                    if self.cordinates[number][c] != self.CORD_TRA:
                        if self.cordinates[number][c] != self.CORD_LAND1:
                            if len(lands)==0:
                                if TEST:
                                    c = 2
                                else:
                                    continue
    #                                print "out"
                            if self.cordinates[number][c]==self.CORD_LAND3:
                                length = self.check_quat(s,lands)
                            else:
                                length=self.dist_land(s, lands)
                            if length==[]:
                                if TEST:
                                    c = 2
                                else:
    #                                print "no obj"
    #                                rospy.sleep(1.)
                                    continue
                    if self.cordinates[number][c] == self.CORD_TRA:
                        self.segmlandmark[number][id(s)] = [0,0,0,0,0,0,-1]
                        break
                    elif self.cordinates[number][c] == self.CORD_LAND1:
#                        self.segmlandmark[id(s)] = self.first_land(s, lands)
                        numbers = numpy.random.choice(range(len(lands)))
                        self.segmlandmark[number][id(s)] = lands[numbers]
                        break
                    else:
                        numbers = numpy.random.choice(length)
                        self.segmlandmark[number][id(s)] = lands[numbers]                        
                        break
                self.segmclass[number][id(s)] = c
#            print number
            k += 1

    def check_quat(self,s, lands):
        n = len(lands)
        lands_list = []
        s = numpy.array(s)
        ss = s.T
        f_list = []
#        e_list = []
        for i in range(n):
            lx = numpy.power(ss[0] - lands[i][0],2)
            ly = numpy.power(ss[1] - lands[i][1],2)
            lz = numpy.power(ss[2] - lands[i][2],2)
            leng = lx+ly+lz
            f_lengths = numpy.sqrt(leng)
            f_length = f_lengths.min()
            if lands[i][4]==None:
                lands[i][4] = numpy.nan
            if math.isnan(lands[i][4]):
                continue
            lands_list.append(i)
            f_list.append(f_length)
        return lands_list
                

    def get_field_pos_from_point(self,):
        field = self.field_list
        f = []
        for p in field:
            f.append([p.x,p.y,p.z])
        return f
    def get_time_start_end(self,joint_list):
        st = joint_list[0].header.stamp.to_sec()
        st = st - TIME*2 
        et = joint_list[0].header.stamp.to_sec()
        et = et + 1.0
        return st, et
        
    def get_time_start_end_raw(self,joint_list):
        st = joint_list[0].header.stamp
        sts = (st.secs) * 1000000000
        stns = st.nsecs
        st = sts+stns
        et = joint_list[-1].header.stamp
        ets = (et.secs) * 1000000000
        etns = et.nsecs 
        et = ets+etns
        return st, et        
        
    def get_object_from_time(self, k, joint_list):
        st, et = self.get_time_start_end(joint_list)
        object_db = self.object.get_landmark_list(st,et)
        objects = object_db
        obj_pos = self.get_object_infos_pos(object_db)
        return obj_pos, objects

    def get_object_from_time2(self, number, k, joint_list):
        st, et = self.get_time_start_end(joint_list)
        key = str([st,et])
        if key in self.lands_buf[number].keys():
            obs = self.lands_buf[number][key]
            return list(obs[0]), obs[1]
        ix = numpy.where((self.df_times[number] > st) & (self.df_times[number] < et))[0]
        obj_pos = self.df_data[number][ix]
        obj = []
        for o in obj_pos:
            pos1 = numpy.array(o)
            pos2 = numpy.array(obj)
            keys = distance_check(pos1, pos2)
            if keys:
                obj.append(o)
        obj_pos = numpy.array(obj)
        objects = obj_pos
        obs = [obj_pos, objects]
        self.lands_buf[number][key] = obs
        return list(obj_pos), objects
        
        
    def quat2quat(self, quat1, quat2):
        """
        quat1 --> quat2 = quat
        """
        key = False
        qua1 = -quat1[0:3]
        quv1 = quat1[3]
        qua2 = quat2[0:3]        
        quv2 = quat2[3]
        if not key:
            qua = quv1 * qua2 + quv2 * qua1 + numpy.cross(qua1, qua2)
            quv = quv1 * quv2 - numpy.dot(qua1, qua2)
            if quv < 0.0:
                qua = qua * -1.
                quv = quv * -1.
        quat = numpy.r_[qua, numpy.array([quv])]
        return quat
        
# 遷移確率更新

    def get_joint_states_position(self, joint_state_list):
        name = joint_state_list[0].name
        names = []
        for j in range(len(name)):
            if not name[j] in USE_JOINT:
                names.append(j)
        position = []
        for joint_state in joint_state_list:
            p = joint_state.position
            position.append(p)

        
        pos = numpy.array(position)
        
        if names != []:
            pp = numpy.delete(pos, names, 1)

        return pp

    def get_object(self,data):
        objects = data
        point_list = []        
        for i in range(1):
            o = objects
            oo = [o.point.x, o.point.y, o.point.z]
            point_list.append(oo)
        return point_list


    def get_object_infos_pos(self, object_info):
        lands = []
        x = object_info.position_x.values
        y = object_info.position_y.values
        z = object_info.position_z.values
        ix = object_info.id.values
        for i in range(len(object_info)):
            xx = x[i]
            yy = y[i]
            zz = z[i]
            ii = ix[i]
            land = [xx, yy, zz,ii]
            lands.append(land)
        return lands
        

    def normlize_time(self, num_step, max_time):
        step = float(max_time)/(num_step+1)
        time_stamp = []

        for n in range(num_step):
            time_stamp.append((n + 1) * step)

        return time_stamp

    def normalize_samples(self, d, nsamples):
        if len(d) == 1:
            return numpy.ones(nsamples) * d[0]
        else:
            return numpy.interp(range(nsamples),
                                numpy.linspace(0, nsamples - 1, len(d)), d)

    def load_model(self, number,basename):
        # GP読み込み
        rospy.loginfo("now load model data")
        for c in range(self.numclass[number]):
            filename = os.path.join(basename, "class%03d.npy" % c)
            self.segm_in_class[number][c] = [s for s in numpy.load(filename)]

            landmarks = numpy.load(os.path.join(basename,
                                                "landmarks%03d.npy" % c))

            for s, l in zip(self.segm_in_class[number][c], landmarks):
                self.segmlandmark[number][id(s)] = l

            self.update_gp(number,c)

        # 遷移確率更新
        self.trans_prob[number] = numpy.load(os.path.join(basename, "trans.npy"))
        self.trans_prob_bos[number] = numpy.load(os.path.join(basename,
                                                      "trans_bos.npy"))
        self.trans_prob_eos[number] = numpy.load(os.path.join(basename,
                                                      "trans_eos.npy"))
    def generate_class(self, number, basename):

        for i in range(self.numclass[number]):
            gendata_lo,gendata_mi,gendata_hi, gendata_sigma=self.gps[number][i].generate2(numpy.arange(0, self.MAX_LEN, 1))
            gendata_hi = numpy.array(gendata_hi)
            gendata_lo = numpy.array(gendata_lo)

            gendata_mi = numpy.array(gendata_mi)
            gendata_sigma = numpy.array(gendata_sigma)
            numpy.savetxt(basename + "GP_m{0:d}.csv".format(i), gendata_mi,delimiter=",")
            numpy.savetxt(basename + "GP_sigma{0:d}.csv".format(i), gendata_sigma,delimiter=",")

    def gp_curve_old(self, number, basename):
        # GP読み込み
        gendata_lo_l = []
        gendata_mi_l = []
        gendata_hi_l = []
        for c in range(self.numclass[number]):
            gendata_lo,gendata_mi,gendata_hi,_si=self.gps[number][c].generate2(numpy.arange(0, self.MAX_LEN,0.1))
            gendata_lo_l.append(gendata_lo)
            gendata_mi_l.append(gendata_mi)
            gendata_hi_l.append(gendata_hi)

        plt.figure(figsize=(20,20))
        for c in range(len(self.gps[number])):
            for d in range(self.dim):
                plt.subplot(self.dim, self.numclass[number], c+d*self.numclass[number]+1)
                if self.dim == 1:
#                    plt.plot(numpy.linspace(0, self.MAX_LEN, len(gendata_lo_l[c])) , gendata_lo_l[c], "r-")
                    plt.plot(numpy.linspace(0, self.MAX_LEN, len(gendata_mi_l[c])) , gendata_mi_l[c], "r-")
                    plt.fill_between(numpy.linspace(0, self.MAX_LEN, len(gendata_hi_l[c])) , gendata_hi_l[c],gendata_lo[c], color="C0",alpha=.3)
                    plt.ylim(gendata_mi_l[c][8]-1.0, gendata_mi_l[c][8]+1.0)
                else:
#                    plt.plot(numpy.linspace(0, self.MAX_LEN, len(gendata_lo_l[c][:,d])) , gendata_lo_l[c][:,d], "r-")
                    plt.plot(numpy.linspace(0, self.MAX_LEN, len(gendata_mi_l[c][:,d])) , gendata_mi_l[c][:,d], "b-")
                    plt.fill_between(numpy.linspace(0, self.MAX_LEN, len(gendata_hi_l[c][:,d])) , gendata_hi_l[c][:,d], gendata_lo_l[c][:,d], color="r", alpha=.3)
        plt.tight_layout()
        plt.savefig( basename+"test.png" )
        print("save")        
    def update_gp(self,number, c):
        datay = []
        datax = []
        for s in self.segm_in_class[number][c]:
            try:
                s = self.cordinate_transform(number,s, self.segmlandmark[number][id(s)],
                                             self.cordinates[number][c])
            except:
                print self.segmlandmark[number].keys()
                print s
                dump()

            datay += [y for y in s]
            datax += range(len(s))
        # 間引く,ひとところに固まることのないように
        s = rospy.Time.now()
        self.gps[number][c].learn(numpy.array(datax), datay)
        st = (rospy.Time.now()-s).to_sec()
        self.gp_time[number] += st

    def update_gp2(self, c):
        datay = []
        datax = []
        for s in self.segm_in_class[c]:
            s = self.cordinate_transform(s, self.segmlandmark[id(s)],
                                         self.cordinates[c])
            datay += [y for y in s]
            datax += range(len(s))
        # 間引く,ひとところに固まることのないように
        self.gps[c].learn2(numpy.array(datax), datay)        
        
        
    def sample_class(self, landmark, segm):
        prob = []

        for c, gp in enumerate(self.gps):
            slen = len(segm)
            plen = 1.0
            if len(segm) > 2:
                plen = (self.AVE_LEN**slen * math.exp(-slen) /
                        math.factorial(self.AVE_LEN))

                cord = self.cordinates[c]
                s = self.cordinate_transform(segm, landmark, cord)
                p = gp.calc_lik(range(len(s)), s, self.MAX_LEN)
                prob.append((math.exp(p) * plen))
            else:
                prob.append(0)

        accm_prob = [0]*self.numclass
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i, prob[i]

        print "aaaaaaaaaaaaaaaaaa"

    def calc_output_prob(self, number, c, segm, landmark):
        gp = self.gps[number][c]

        slen = len(segm)

        plen = 1.0
        if len(segm) > 2:
#            plen = (self.AVE_LEN**slen * math.exp(-slen) /
#                    math.factorial(self.AVE_LEN))
            plen = (self.AVE_LEN**slen * math.exp(-self.AVE_LEN) /
                    math.factorial(slen))

            cord = self.cordinates[number][c]
            s = self.cordinate_transform(number, segm, landmark, cord)
            ss = rospy.Time.now()
            p = gp.calc_lik(range(len(s)), s, self.MAX_LEN)
            st = (rospy.Time.now()-ss).to_sec()
            self.outprob_time[number] += st
            return p + numpy.log(plen)
        else:
            return 0

    def save_model(self, number, basename):
        for n, segm in enumerate(self.segments[number]):
            joints = self.joint_states[number][n]
            classes = []
            clen = []
            stamps=[]
#            names = []
            t=0
            for s in segm:
                c = self.segmclass[number][id(s)]
                ts = joints[t].header.stamp.to_sec()
                classes += [c for i in range(len(s))]
                te = joints[t+len(s)-1].header.stamp.to_sec()
                t += len(s)
                dd = [c, len(s)]
                clen.append(dd)
                stamps.append([ts,te])
            numpy.savetxt(basename+"segm%03d.txt" % n, classes, fmt="%d")
            numpy.savetxt(basename+"slen%03d.txt" % n, numpy.array(clen,dtype=numpy.int))
            numpy.savetxt(basename+"stamps%03d.txt" % n, numpy.array(stamps))
              
            
        numpy.savetxt(basename+"cord.txt", numpy.array(self.cordinates[number]), fmt="%d")
        plt.figure()
        for c in range(len(self.gps[number])):
            for d in range(self.dim):
                plt.subplot(self.dim, self.numclass[number], c + d * self.numclass[number] + 1)
                for data in self.segm_in_class[number][c]:
                    trans_data = self.cordinate_transform(number, data,
                                                          self.segmlandmark[number][id(data)],
                                                          self.cordinates[number][c])

                    if self.dim == 1:
                        plt.plot(range(len(trans_data)), trans_data, "b-")
                    else:
                        plt.plot(range(len(trans_data)),
                                 trans_data[:, d], "b-")
                    plt.ylim(-1.1, 1.1)

        plt.savefig(basename+"class.png")
        numpy.savetxt(basename+"cordinate.txt", self.cordinates[number])
        # テキストでも保存
        numpy.save(basename + "trans.npy", self.trans_prob[number])
        numpy.save(basename + "trans_bos.npy", self.trans_prob_bos[number])
        numpy.save(basename + "trans_eos.npy", self.trans_prob_eos[number])
        numpy.savetxt(basename + "lik.txt", [self.calc_lik(number)])

        for c in range(self.numclass[number]):
            numpy.save(basename+"class%03d.npy" % c,
                       self.segm_in_class[number][c])
            numpy.save(basename+"landmarks%03d.npy" % c,
                       [self.segmlandmark[number][id(s)]
                        for s in self.segm_in_class[number][c]])

        numpy.save(basename + "cordinates.npy", self.cordinates[number])


    def cordinate_transform(self,number, s, land_pos, cord):
        s_time = rospy.Time.now()
        land_pos = numpy.array(land_pos)
        if cord == self.CORD_TRA:
            set_time = rospy.Time.now()
            ss = numpy.array(s).T
            ss_xyz = ss[:3]
            ss_h = ss[3]
            ss_qxyzw = ss[4:8]
            ss_power = ss[8:]
            ss_h = ss_h.T
            ss_qxyzw = ss_qxyzw.T
            ss_power = ss_power.T
            offset = numpy.zeros(len(ss_xyz))
            for i in range(len(ss_xyz)):
                offset[i] = ss_xyz[i][0]
            s_xyz = ss_xyz.T
            self.cordinate_set[number] += (rospy.Time.now()-set_time).to_sec()

            q = ss_qxyzw[0]
            R = tf.transformations.quaternion_matrix(q)

            v = []


            r_xyzw = ss_qxyzw[0]
            r_xyzw_inv = numpy.array([-r_xyzw[0],-r_xyzw[1],-r_xyzw[2],r_xyzw[3]])
            rot_inv = tf.transformations.euler_from_quaternion(r_xyzw_inv)
            pos = -s_xyz[0]
            pos = numpy.r_[pos,1.0]
            R_inv = tf.transformations.quaternion_matrix(r_xyzw_inv)
            pos_inv = numpy.dot(R_inv,pos)[0:3]


            for i, sss in enumerate(s_xyz):
                sr = calctransform(rot_inv, pos_inv, sss)
                x = sr[0]
                y = sr[1]
                z = sr[2]
                v.append([x, y, z])
            s_xyz = numpy.array(v)


            q = ss_qxyzw[0]
            ls = []
            for qq in ss_qxyzw:
                qt = self.quat2quat(q, qq)
                ls.append(qt)
            ls = numpy.array(ls)
            ss_h = numpy.array(ss_h)
            ss_power = ss_power/10.0
            try:
                ss = numpy.c_[s_xyz, ss_h]
                if self.dim > 4:
                    ss = numpy.c_[ss, ls]
                if self.dim > 8:
                    ss = numpy.c_[ss, ss_power]
            except:
                print s_xyz.shape
                print ss_h.shape
                print ls.shape
                print raw_input()

        elif cord == self.CORD_LAND1:
            set_time = rospy.Time.now()
            ss = numpy.array(s).T
            ss_xyz = ss[:3]
            ss_h = ss[3]
            ss_qxyzw = ss[4:8]
            ss_power = ss[8:]
            ss_h = ss_h.T
            ss_qxyzw = ss_qxyzw.T
            s_xyz = ss_xyz.T
            ss_power = ss_power.T

            v = []



            s_xyz = s_xyz - land_pos[0:3]



            t = -math.atan2(s_xyz[0][1], s_xyz[0][0])
            R = numpy.array([[numpy.cos(t), -numpy.sin(t)],
                             [numpy.sin(t), numpy.cos(t)]])
            zaw = t
            v = []
            for i, sss in enumerate(s_xyz):
                x = R[0][0] * sss[0] + R[0][1] * sss[1]
                y = R[1][0] * sss[0] + R[1][1] * sss[1]
                z = sss[2]
                v.append([x, y, z])

            s_xyz = numpy.array(v)
            if land_pos[4]==None:
                land_pos[4] = numpy.nan
            if math.isnan(land_pos[4]):
                if land_pos[2] > 0.25:
                    rxyzw = tf.transformations.quaternion_from_euler(0.0,0.0,zaw)
                else:
                    rxyzw = tf.transformations.quaternion_from_euler(0.0,-1.57,zaw)
            else:
                rxyzw = tf.transformations.quaternion_from_euler(land_pos[3],land_pos[4],land_pos[5])

            q = numpy.array(rxyzw)
            ls = []
            trqs = rospy.Time.now()
            for qq in ss_qxyzw:
                qt = self.quat2quat(q, qq)
                ls.append(qt)
            self.cordinate_roq[number] += (rospy.Time.now()-trqs).to_sec()
            ls = numpy.array(ls)
            ss_power = ss_power/10.0
            ss_h = numpy.array(ss_h)
            ss = numpy.c_[s_xyz, ss_h]           
            if self.dim > 4:
                ss = numpy.c_[ss, ls]
            if self.dim > 8:
                ss = numpy.c_[ss, ss_power]
                
        elif cord == self.CORD_LAND2:
            set_time = rospy.Time.now()
            ss = numpy.array(s).T
            ss_xyz = ss[:3]
            ss_h = ss[3]
            ss_qxyzw = ss[4:8]
            ss_power = ss[8:]
            ss_h = ss_h.T
            ss_qxyzw = ss_qxyzw.T
            s_xyz = ss_xyz.T
            ss_power = ss_power.T

            v = []
            s_xyz = s_xyz - land_pos[0:3]





            t = -math.atan2(s_xyz[0][1], s_xyz[0][0])
            R = numpy.array([[numpy.cos(t), -numpy.sin(t)],
                             [numpy.sin(t), numpy.cos(t)]])
            zaw = t
            v = []
            for i, sss in enumerate(s_xyz):
                x = R[0][0] * sss[0] + R[0][1] * sss[1]
                y = R[1][0] * sss[0] + R[1][1] * sss[1]
                z = sss[2]
                v.append([x, y, z])

            t = -math.atan2(s_xyz[0][2], numpy.sqrt(s_xyz[0][0]**2 + s_xyz[0][1]**2))
            R = numpy.array([[numpy.cos(t), -numpy.sin(t)],
                             [numpy.sin(t), numpy.cos(t)]])
            vv = []
            for i, sss in enumerate(v):
                x = R[0][0] * sss[0] + R[0][1] * sss[2]
                z = R[1][0] * sss[0] + R[1][1] * sss[2]
                y = sss[1]
                vv.append([x, y, z])
            s_xyz = numpy.array(vv)
            if land_pos[4]==None:
                land_pos[4] = numpy.nan
            if math.isnan(land_pos[4]):
                if land_pos[2] > 0.25:
                    rxyzw = tf.transformations.quaternion_from_euler(0.0,0.0,zaw)
                else:
                    rxyzw = tf.transformations.quaternion_from_euler(0.0,-1.57,zaw)                    
            else:
                rxyzw = tf.transformations.quaternion_from_euler(land_pos[3],land_pos[4],land_pos[5])

            q = numpy.array(rxyzw)
            ls = []
            trqs = rospy.Time.now()
            for qq in ss_qxyzw:
                qt = self.quat2quat(q, qq)
                ls.append(qt)
            self.cordinate_roq[number] += (rospy.Time.now()-trqs).to_sec()
            ss_power = ss_power/10.0
            ls = numpy.array(ls)
            ss_h = numpy.array(ss_h)
            ss = numpy.c_[s_xyz, ss_h]           
            if self.dim > 4:
                ss = numpy.c_[ss, ls]
            if self.dim > 8:
                ss = numpy.c_[ss, ss_power]

        elif cord == self.CORD_LAND3:
            set_time = rospy.Time.now()
            ss = numpy.array(s).T
            ss_xyz = ss[:3]
            ss_h = ss[3]
            ss_qxyzw = ss[4:8]
            ss_power = ss[8:]
            ss_h = ss_h.T
            ss_qxyzw = ss_qxyzw.T
            s_xyz = ss_xyz.T
            ss_power = ss_power.T
            r_xyz = land_pos[3:6]
            self.cordinate_set[number] += (rospy.Time.now()-set_time).to_sec()
            tqs = rospy.Time.now()



            
            r_xyzw = tf.transformations.quaternion_from_euler(r_xyz[0],r_xyz[1],r_xyz[2])
            r_xyzw_inv = numpy.array([-r_xyzw[0],-r_xyzw[1],-r_xyzw[2],r_xyzw[3]])
            rot_inv = tf.transformations.euler_from_quaternion(r_xyzw_inv)
            pos = -land_pos[0:3]
            pos = numpy.r_[pos,1.0]
            self.tf_q_time[number] += (rospy.Time.now()-tqs).to_sec()
            trs = rospy.Time.now()
            R_inv = tf.transformations.quaternion_matrix(r_xyzw_inv)
            pos_inv = numpy.dot(R_inv,pos)[0:3]
            self.tf_r_time[number] += (rospy.Time.now()-trs).to_sec()

            v = []

            for i, sss in enumerate(s_xyz):
                sr = calctransform(rot_inv, pos_inv, sss)
                x = sr[0]
                y = sr[1]
                z = sr[2]
                v.append([x, y, z])
            vv = v
            s_xyz = numpy.array(vv)
            q = ss_qxyzw[0]
            ls = []
            trqs = rospy.Time.now()
            for qq in ss_qxyzw:
                qt = self.quat2quat(r_xyzw, qq)
                ls.append(qt)
            self.cordinate_roq[number] += (rospy.Time.now()-trqs).to_sec()
            ls = numpy.array(ls)
            ss_h = numpy.array(ss_h)
            ss = numpy.c_[s_xyz, ss_h]           
            if self.dim > 4:
                ss = numpy.c_[ss, ls]
            if self.dim > 8:
                ss = numpy.c_[ss, ss_power]
        elif cord == self.CORD_MOV:
             ss = numpy.array(s)
        st_time = (rospy.Time.now()-s_time).to_sec()
        self.cordinate_time[number] += st_time
        self.cordinate_num[number] += 1
        return ss
        
    def forward_filtering(self, number, d, ii):
        T = len(d)
        a = numpy.ones((len(d), self.MAX_LEN, self.numclass[number])) # 前向き確率
        a[a==1.0] = None
        ll = numpy.zeros((T, self.MAX_LEN, self.numclass[number]),dtype='object')
        
#        p_bar = ProgressBar(maxval=T).start()
#            print "check"
#        print T
            
        for t in range(T):
#            p_bar.update(p_bar.currval + 1)        
            for k in range(self.MIN_LEN, self.MAX_LEN, self.SKIP_LEN):
                if t-k < 0:
                    break
                ssp = rospy.Time.now()
                lands, ob = self.get_object_from_time2(number, ii,d[t-k:t+1])
                self.land_get_time[number] += (rospy.Time.now()-ssp).to_sec()
                segm = self.get_joint_states_position(d[t-k:t+1])
                for c in range(self.numclass[number]):
                    out_prob = None
                    lll = 0
                    cord = self.cordinates[number][c]                    
                    if cord == self.CORD_LAND1 or cord== self.CORD_LAND2 or cord== self.CORD_LAND3:
                        ss = rospy.Time.now() 
                        if cord == self.CORD_LAND3:
                            near_lands = self.check_quat(segm,lands)
                        elif cord == self.CORD_LAND2:
                            near_lands = self.dist_land(segm, lands)
                        else:
                            near_lands = range(len(lands))
                        self.land_choice_time[number] += (rospy.Time.now()-ss).to_sec()
                        for iii in near_lands:                                 
                            calc_prob = self.calc_output_prob(number , c, segm, lands[iii])
                            if out_prob == None:
                                out_prob = calc_prob
                                lll = lands[iii][-1]
                            elif calc_prob >= out_prob:
                                out_prob = calc_prob
                                lll = lands[iii][-1]
                        if len(near_lands)!=0:
                            lm = self.get_dataframe2(number, ob,lll)
                        else:
#                            "no obj"
                            lm = ob[0:0]
                            ll[t,k,c] = lm
                        if len(lm)==0:
                            pass
#                            "no obj"
                        ll[t,k,c] = lm
                        
                    else:
                        out_prob = self.calc_output_prob(number, c, segm, [0.,0.,0.])
                        ll[t,k,c] = Point()

                    # 遷移確率
                    sk = rospy.Time.now()
                    tt = t-k-1
                    log_array = []
                    if out_prob==None:
                        out_prob = MINIX
                    if tt >= 0:
                        for kk in range(self.MIN_LEN,self.MAX_LEN):
                            for cc in range(self.numclass[number]):
                                if math.isnan(a[tt,kk,cc]):
                                    continue
                                log_array.append(a[tt, kk,cc] + numpy.log(self.trans_prob[number][cc, c]) + out_prob)
                        log_array = numpy.array(log_array)
                        if len(log_array)==0:
                            continue
                        min_log = numpy.min(log_array)
                        max_log = numpy.max(log_array)
                        if min_log >= MIN and max_log < MAX:
                            a[t, k, c]= numpy.log(numpy.exp(log_array).sum())
                        elif min_log < MIN and max_log < MAX:
                            min_T = MIN - min_log
                            if max_log +min_T > MAX:
                                min_T = MAX - max_log
                            log_array += min_T
                            a[t, k, c] = numpy.log(numpy.exp(log_array).sum())-min_T
                        elif max_log > MAX:
                            max_T = MAX - max_log
                            log_array += max_T
                            a[t, k, c] = numpy.log(numpy.exp(log_array).sum())-max_T

                    else:
                        # 最初の単語
                        a[t, k, c] = out_prob + numpy.log(self.trans_prob_bos[number][c])

                    if t == T - 1:
                        # 最後の単語
                        a[t, k, c] += numpy.log(self.trans_prob_eos[number][c])                        
                    self.alpha_time[number] += (rospy.Time.now()- sk).to_sec()

                    

        """
            for k in range(self.MAX_LEN):
                if t-k<0:
                    break
                print "%.2f" % a[t,k],
            print
        raw_input()
        """
        return a, ll

    def sample_idx(self, number, prob_lik):
        max_log_arg = numpy.nanargmax(prob_lik)
        max_log = prob_lik[max_log_arg]
        min_log_arg = numpy.nanargmin(prob_lik)
        min_log = prob_lik[min_log_arg]
        if min_log >= MIN and max_log < MAX:
            prob = numpy.exp(prob_lik)
        elif min_log < MIN and max_log <= MAX:
            min_T = MIN - min_log
            if max_log +min_T > MAX:
                min_T = MAX - max_log
            prob_lik += min_T
            prob = numpy.exp(prob_lik)
        elif max_log > MAX:
            max_T = MAX - max_log
            prob_lik += max_T
            prob = numpy.exp(prob_lik)
        prob = numpy.nan_to_num(prob)
        accm_prob = [0, ] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]
        accm_prob = numpy.array(accm_prob)
        accm_prob = accm_prob / accm_prob[-1] * 100.0
        r =  numpy.array(prob).reshape([-1,self.numclass[number]])
        r = r / numpy.sum(r)
        _r = []
        for i in range(self.numclass[number]):
            _r.append(numpy.sum(r[:,i]))
        if accm_prob[-1] ==0.0:
            print "non prob"
        try:
            rnd = numpy.random.uniform(0.0, accm_prob[-1])
        except:
            rospy.logwarn("error")
            rospy.loginfo(accm_prob)
            raw_input()
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i

        print "aaaaaaaaaaaaaaaaaa"

    def backward_sampling(self, number, a, d, ll):
        T = a.shape[0]
        t = T-1

        segm = []
        segm_class = []
        land = []
        print ""
        while True:
            idx = self.sample_idx(number, a[t].reshape(self.MAX_LEN * self.numclass[number]))
            l = ll[t].reshape(self.MAX_LEN * self.numclass[number])[idx]
            test = numpy.zeros(self.numclass[number])
            for kk in range(self.MAX_LEN):
                ttt = a[t][:][kk]
                for ti in range(self.numclass[number]):
                    test[ti] += ttt[ti]
            k = int(idx/(self.numclass[number]))
            c = int(idx % self.numclass[number])
            if type(l) == int:
                print "no lands"
                continue
            if type(l) == list:
                if len(l) == 0:
                    print "no lands"
                    continue
            if t - k < 0:
                print "warn"
                continue
            s = self.get_joint_states_position(d[t-k:t+1])

            # パラメータ更新
            segm.insert(0, s)
            segm_class.insert(0, c)
            land.insert(0, l)
            t = t-k-1
    
            if t <= 0:
                break
#        raw_input()
        print "class",
        return segm, segm_class, land

    def calc_start_prob(self,number):
        self.trans_prob_bos[number] = numpy.zeros(self.numclass[number])
        self.trans_prob_bos[number] += 0.1

        # 数え上げる
        for n, segm in enumerate(self.segments[number]):
            try:
                c = self.segmclass[number][id(segm[0])]
                self.trans_prob_bos[number][c] += 1.0
            except:
                pass
        self.trans_prob_bos[number] = (self.trans_prob_bos[number] /
                                   self.trans_prob_bos[number].sum())

    def calc_end_prob(self,number):        
        self.trans_prob_eos[number] = numpy.zeros(self.numclass[number])
        self.trans_prob_eos[number] += 0.1

        # 数え上げる
        for n, segm in enumerate(self.segments[number]):
            try:
                c = self.segmclass[number][id(segm[-1])]
                self.trans_prob_eos[number][c] += 1.0
            except:
                pass
        self.trans_prob_eos[number] = (self.trans_prob_eos[number] /
                                   self.trans_prob_eos[number].sum())


    def calc_trans_prob(self,number):
        self.trans_prob[number] = numpy.zeros((self.numclass[number], self.numclass[number]))
        self.trans_prob[number] += 0.1

        # 数え上げる
        for n, segm in enumerate(self.segments[number]):
            for i in range(1, len(segm)):
                try:
                    cc = self.segmclass[number][id(segm[i-1])]
                    c = self.segmclass[number][id(segm[i])]
                except KeyError, e:
                    # gibss samplingで覗かれているものは無視
                    break
                self.trans_prob[number][cc, c] += 1.0

        # 正規化
        self.trans_prob[number] = (self.trans_prob[number] /
                                   self.trans_prob[number].sum(1).reshape(self.numclass[number], 1))

    def remove_ndarray(self, lst, elem):
        l = len(elem)
        for i, e in enumerate(lst):
            if len(e) != l:
                continue
            if id(e) == id(elem):
                lst.pop(i)
                return
        raise ValueError("ndarray is not found!!")

    def learn(self,number):
        if not self.is_initialized[number]:
            # GPの学習
            st = rospy.Time.now()
            for i in range(len(self.segments[number])):
                for s in self.segments[number][i]:
                    c = self.segmclass[number][id(s)]
                    self.segm_in_class[number][c].append(s)
            rospy.loginfo((rospy.Time.now() - st).to_sec())
            # 各クラス毎に学習
            for c in range(self.numclass[number]):
                self.update_gp(number, c)

            self.is_initialized[number] = True
            rospy.loginfo((rospy.Time.now() - st).to_sec())
        self.gp_curve_old(number, self.svdir[number])
        return self.update(number,True)

    def recog(self,number):
        self.update(number,False)

    def update(self, number, learning_phase=True):
        cls = [0] * self.numclass[number]
        for i in range(len(self.segments[number])):
            d = self.joint_states[number][i]
            #  そのファイルの生データ抽出
#           そのファイルの全分節抽出
            segm = self.segments[number][i]
#           そのファイルのランドマーク位置抽出
#            print "before"
#            print "[",
            for s in segm:
                c = self.segmclass[number][id(s)]
#                print c ,",",
#               対象の分節のクラスを抽出
#               対象の分節の分類を全体から削除
                self.segmclass[number].pop(id(s))
#               対象の分節のランドマークを全体から削除
                self.segmlandmark[number].pop(id(s))
                if learning_phase:
                    # パラメータ更新
                    # 対象の分節を全体から削除
                    self.remove_ndarray(self.segm_in_class[number][c], s)
#            print "]"
            print(CATEGORY[number],self.number[number],"update1",i,len(self.segments[number]))
            if learning_phase:
                # GP更新
                start = rospy.Time.now()
                for c in range(self.numclass[number]):
                    self.update_gp(number, c)

                # 遷移確率更新
#                print (rospy.Time.now()-start).to_sec()
                self.calc_trans_prob(number)
                self.calc_start_prob(number)
                self.calc_end_prob(number)


            start = rospy.Time.now()
#            print "forward...",
            a, ll = self.forward_filtering(number ,d, i)

#            print "backward...",
            pt = "{0:d}".format(self.number[number])
            
            make_dir(self.svdir[number]+pt)
            numpy.save(self.svdir[number]+pt+"/"+"aplha{0:d}.npy".format(i),a)
            numpy.save(self.svdir[number]+pt+"/"+"lands{0:d}.npy".format(i),ll)
            segm, segm_class, land = self.backward_sampling(number, a, d, ll)
#            print (rospy.Time.now()-start).to_sec()
            self.segments[number][i] = segm
            ccc = []
            for s, c, ll in zip(segm, segm_class, land):
                p = Point()
                if type(ll) == int:
                    continue
                if type(ll) != type(p):
                    try:
                        l_test = ll[0]
                    except:
                        print s
                        print c
                        print ll
                        continue
                    
                self.segmclass[number][id(s)] = c
                ccc.append(c)
                self.object_list[number][id(s)] = ll
                if type(ll) == type(p):
                    self.segmlandmark[number][id(s)] = [ll.x, ll.y, ll.z, 0.,0.,0., -1]
                else:
                    try:
                        self.segmlandmark[number][id(s)] = self.df2land2(ll)
                    except:
                        print(ll ==[])
                        print s
                        print c
                        print ll
                        print raw_input()
                cls[c] += 1
                # パラメータ更新
                if learning_phase:
                    self.segm_in_class[number][c].append(s)

#            print " [",
            for s in self.segm_in_class[number]:
                pass
#                print len(s),
#            print "]"
            ccc.reverse()
#            print ccc
#            print(i, u"/", len(self.segments[number]))
            

            if learning_phase:
                # GP更新
                start = time.clock()
                for c in range(self.numclass[number]):
                    self.update_gp(number, c)

                # 遷移確率更新
                self.calc_trans_prob(number)
                self.calc_start_prob(number)
                self.calc_end_prob(number)
#                print cls
        ci = 0
#        co=[]
#        for s in self.segm_in_class[number]:
#            if self.cordinates[number][ci]!=0:
#                if len(s)==0:
#                    co.append(1)
#                else:
#                    co.append(0)
#            ci+=1
#        if co.count(0) == 0:
#            return False
        return True

    def calc_lik(self, number):
        lik = 0
        for segm in self.segments[number]:
            for s in segm:
                c = self.segmclass[number][id(s)]
                lik += self.gps[number][c].calc_lik(range(len(s)), s, self.MAX_LEN)

        return lik

    def learn_start(self, data):
        if GET_DATA:
            min_time = min(self.time_list)
            max_time = max(self.time_list)
            self.object.divid_df(min_time, max_time)
            self.object.save("data_frame_model.csv")
            rospy.loginfo("save")
            print raw_input()
            import sys
            sys.exit(1)
        
        s = rospy.Time.now().to_sec()
        rospy.loginfo("learn start")
        self.get_operation_object_data()
        self.extract_object()
        self.check_active_time()
        jobs = []
        self.base_dir = SAVE.format(os.times()[-1])#,self.t_n,self.h1_n,self.h_n,self.h2_n)
        rospy.loginfo("load")
        for i in range(len(CATEGORY)):
            if len(self.joint_states[i][0])==0:
                continue
            rospy.loginfo("{0:d} data in {1:s}".format(len(self.joint_states[i]),CATEGORY[i]))
            self.load_data(i)
            print "load {}".format(CATEGORY[i])
        make_dir(self.base_dir)
#        rospy.sleep(30.)
        for i in range(len(CATEGORY)):
            if len(self.joint_states[i][0])==0:
                continue
            job = multiprocessing.Process(target=self.multi_learn, args=(i,))
            jobs.append(job)
            job.start()
        [job.join() for job in jobs]
        res = TriggerResponse()
        res.message = self.base_dir
        self._set_state()
        rospy.loginfo("learn finish")
        rospy.loginfo(rospy.Time.now().to_sec() - s)
        return TriggerResponse()


    def multi_learn(self, number):
        s = rospy.Time.now()
        savedir= self.base_dir
        savedir += "{}/".format(CATEGORY[number])
        make_dir(savedir)
        self.svdir[number] = savedir
        liks = []
        rospy.loginfo((rospy.Time.now() - s).to_sec())
        self.number[number] = 0
        rospy.loginfo("{} start".format(CATEGORY[number]))
        for it in range(ITE):
            self.number[number] = it
            st = rospy.Time.now()
#            print "-----", it, "-----"
            flag = self.learn(number)
            try:
                lik = self.calc_lik(number)
            except:
                lik = 0.0
            self.gp_curve_old(number, savedir)
#            print "lik =", lik
            rospy.loginfo(rospy.Time.now().to_sec() - st.to_sec())
            if len(liks) > 3:
                if lik == liks[-1]:
                    break
            liks.append(lik)
            numpy.savetxt(savedir+"liks.txt",liks)
#            if not flag:
#                rospy.loginfo("out")
#                return False
        make_dir(savedir)
        print liks
        print("now saving")
        self.save_model(number, savedir)
        print self.calc_lik(number)
        self.generate_class(number, savedir)
        rospy.loginfo("{0:s}:all time:{1:f}".format(CATEGORY[number],(rospy.Time.now() - s).to_sec()))
     
        res = TriggerResponse()
        res.success = True
        res.message = savedir
        rospy.loginfo("{} finish".format(CATEGORY[number]))
        return True

    def recog_start(self,data):
        if GET_DATA:
            min_time = min(self.time_list)
            max_time = max(self.time_list)
            self.object.divid_df(min_time, max_time)
            self.object.save("data_frame_model.csv")
            rospy.loginfo("save")
            print raw_input()
            import sys
            sys.exit(1)
        
        s = rospy.Time.now().to_sec()
        rospy.loginfo("learn start")
        self.get_operation_object_data()
        self.extract_object()
        self.check_active_time()
        jobs = []
        self.base_dir = SAVE.format(os.times()[-1])#,self.t_n,self.h1_n,self.h_n,self.h2_n)
        rospy.loginfo("load")
        for i in range(len(CATEGORY)):
            if len(self.joint_states[i][0])==0:
                continue
            rospy.loginfo("{0:d} data in {1:s}".format(len(self.joint_states[i]),CATEGORY[i]))
            
            self.load_model(i,RECOG.format(CATEGORY[i]))
            self.load_data(i)
            print "load {}".format(CATEGORY[i])
        make_dir(self.base_dir)
#        rospy.sleep(30.)
        for i in range(len(CATEGORY)):
            if len(self.joint_states[i][0])==0:
                continue
            job = multiprocessing.Process(target=self.multi_recog, args=(i,))
            jobs.append(job)
            job.start()
        [job.join() for job in jobs]
        res = TriggerResponse()
        res.message = self.base_dir
        self._set_state()
        rospy.loginfo("learn finish")
        rospy.loginfo(rospy.Time.now().to_sec() - s)
        return TriggerResponse()
        
    def multi_recog(self,number):
        s = rospy.Time.now()
        savedir= self.base_dir
        savedir+="rec_"
        savedir += "{}/".format(CATEGORY[number])
        make_dir(savedir)
        self.svdir[number] = savedir
        liks = []
        rospy.loginfo((rospy.Time.now() - s).to_sec())
        self.number[number] = 0
        rospy.loginfo("{} recog".format(CATEGORY[number]))
        for it in range(ITE):
            self.number[number] = it
            st = rospy.Time.now()
#            print "-----", it, "-----"
            flag = self.recog(number)
            try:
                lik = self.calc_lik(number)
            except:
                lik = 0.0
            self.gp_curve_old(number, savedir)
#            print "lik =", lik
            rospy.loginfo(rospy.Time.now().to_sec() - st.to_sec())
            if len(liks) > 3:
                if lik == liks[-1]:
                    break
            liks.append(lik)
            numpy.savetxt(savedir+"liks.txt",liks)
#            if not flag:
#                rospy.loginfo("out")
#                return False
        make_dir(savedir)
        print liks
        print("now saving")
        self.save_model(number, savedir)
        print self.calc_lik(number)
        self.generate_class(number, savedir)
        rospy.loginfo("{0:s}:all time:{1:f}".format(CATEGORY[number],(rospy.Time.now() - s).to_sec()))
     
        res = TriggerResponse()
        res.success = True
        res.message = savedir
        rospy.loginfo("{} finish".format(CATEGORY[number]))
        return True

        
        rospy.loginfo("recognition start wait for result")
        for it in range(2):
            print "-----", it, "-----"
            self.number = it
            self.recog()
            print "lik =", self.calc_lik()
    #        gpsegm.save_model("test"+savedir)
    
        print("now saving")
        self.save_model(savedir)
        for n, segm in enumerate(self.segments):
            cc = []
            for s in segm:
                c = self.segmclass[id(s)]
                cc.append(int(c))
        res.recog_class = cc
        return res

    def start_recognition(self,data):        
        savedir= RECOG
        make_dir(savedir)
        self.load_model(USE)
        self.load_data()
        rospy.loginfo("recognition start wait for result")
        for it in range(3):
            print "-----", it, "-----"
            self.recog()
            print "lik =", self.calc_lik()
    #        gpsegm.save_model("test"+savedir)
    
        print("now saving")
        self.save_model(savedir)
        return TriggerResponse()
    def run(self,):
        while not rospy.is_shutdown():
            rospy.sleep(0.5)

def make_dir(name):
    try:
        os.mkdir(name)
    except:
        pass

def main():
    rospy.init_node("JointHSMM{}".format(NUMBER))
    jb = GPSegmentation()
    jb.run()
    return True

if __name__ == '__main__':
    main()
