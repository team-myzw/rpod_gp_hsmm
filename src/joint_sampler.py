#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 12:02:07 2016

@author: robocup
"""
import genpy
import os
import sys
import numpy as np
import rospy
import rosbag
from multiprocessing import Process
from rosbag_database.srv import RosbagPlay, RosbagPlayRequest
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import pickle
import importlib
import numpy as np
import os
import copy
from rosbag_database.srv import RosbagRecord, RosbagRecordResponse 
from rosbag_database.srv import RosbagStop, RosbagStopResponse
from rosbag_database.srv import RosbagPlay, RosbagPlayResponse
from progressbar import ProgressBar
#from vision_module.msg import ObjectData, ObjectInfo

USE_TOPIC = "/hsrb/joint_states"
RATE = 100
BAG = os.getenv("BAG_FOLDA")
if BAG == None:
    BAG = ""
DIS = 0.1

def check_distance(max_dis,base_pos,pos):
    dis = np.sqrt(np.power(base_pos[0]-pos[0],2)+np.power(base_pos[1]-pos[1],2)+np.power(base_pos[2]-pos[2],2))
    if dis > max_dis:
        return False
    else:
        return True
    
    
def check_pos_in_list(lis, pos):
    if lis == []:
        return -1
    for l in range(len(lis)):
        loc = lis[l]
        if check_distance(DIS,loc,pos):
            return l
    return -1

def check_pos_in_base(base_pos, poses):
    if len(poses) == 0:
        return []
    pp = np.array(poses)
    p = pp.T
    x = p[0] - base_pos[0]
    y = p[1] - base_pos[1]
    z = p[2] - base_pos[2]
    dis = x**2 + y**2 + z**2
    narray = dis < DIS**2
    if len(narray) == 0:
        return []
    return narray
    
def calc_loc(pos_list):
    n = len(pos_list)
    lands = [] 
    p_bar = ProgressBar(maxval=n).start()
    for i in range(n):
        p_bar.update(p_bar.currval + 1)
        pp = np.array(pos_list[i])
        pp = pp.reshape([-1,3])
        if pp.shape[0]==0:
            continue
        lands.append(list(np.average(pp,axis=0)))
    p_bar.finish()
    return lands

def numbring_land_pos(land_poses):
    N = len(land_poses)
    rospy.loginfo(N)
    base = land_poses
    for n in range(10):
        B = len(base)
        print B
        lsp = [[] for i in range(B)]
        p_bar = ProgressBar(maxval=B).start()
        poses = np.array(land_poses)
        for i in range(B):
            p_bar.update(p_bar.currval + 1)
            if len(poses) == 0:
                continue
            narray = check_pos_in_base(base[i], poses)
            if len(narray) != 0:
                lsp[i].append(poses[narray])
            else:
                continue
            dd = np.where(narray)
            poses = np.delete(poses,dd,0)
        p_bar.finish()
        if len(poses) != 0:
            for l in range(len(poses)):
                lsp.append(poses[l])
        base = calc_loc(lsp)
    return base
    
def typesirialize(messagetype):
    message_list = [messagetype]
    message_sirialize = pickle.dumps(message_list)
    sirialize_list= np.array(message_sirialize.split("\n"))
    message_point = [".msg" in ss for ss in sirialize_list]
    message_select = np.select([message_point],[sirialize_list])
    messages = message_select[message_select != '0']
    type_point = np.where(message_point)[0] + 1
    types = sirialize_list[type_point]
    return messages ,types


def desirialize(data, messagetype):
    messages, types = typesirialize(messagetype)
    code = data.module_code
    sirial_message = data.pickle_message
    sirial_array = np.array(sirial_message.split("\n"),dtype = '|S256')
    

    bool_type = []
    for t in range(len(types)):
        find_type = types[t]
        ff = np.array([find_type in sirial for sirial in sirial_array])
        bool_type.append(ff)

    for t in range(len(types)):
        find_type = types[t]
        for i in range(len(types)):
            check_type = types[i]
            if check_type in find_type:
                continue
            ch = np.where(bool_type[i])
            bool_type[t][ch] = False

    for t in range(len(types)):
        type_point = np.where(bool_type[t])[0]
        sirial_array[type_point] = types[t]
    sirial_code = "(c" + code
    n = len(messages)
    code_point = np.where(sirial_array == sirial_code)[0]
    code_length = np.arange(len(code_point))
    s_a = sirial_array
    for nn in range(n):
        mes = messages[nn]
        s_a[code_point[code_length[code_length % n == nn]]] = mes 
    silialize_message = "\n".join(s_a)
    topics = pickle.loads(silialize_message)

    return topics


class JointBagReader(object):
    def __init__(self):
        self.srv = rospy.ServiceProxy("rosbag_play", RosbagPlay)
        self.joint_position = []
        self.joint_effort = []
        self.joint_vel = []        
        self.joint_name = []

    def data_get(self, req, mes_type):
        req.folda_path = BAG

        data = self.srv.call(req)
        if not data.success:
            rospy.logwarn("error read bag joint")
            return None, None, None

        topics = desirialize(data, mes_type)
        return topics
        for mes in topics:
            self._joint_get(mes)


        joint = self.joint_position
        effort= self.joint_effort
        velocity = self.joint_vel

        self.joint_position = []
        self.joint_effort = []
        self.joint_vel = []
        self.land_pos = []
        
        rospy.loginfo("get joint")
        return joint, effort, velocity

    def land_pos_get(self, req):
        req.folda_path = BAG

        data = self.srv.call(req)
        if not data.success:
            rospy.logwarn("error read bag joint")
            return None, None, None

        topics = desirialize(data, Point())
        for mes in topics:
            self._land_pos_get(mes)


        land_pos = self.land_pos

        self.land_pos = []

        land_pos = numbring_land_pos(land_pos)        
        rospy.loginfo("get lands")        
        return land_pos
        


    def _joint_get(self, data):
        self.joint_effort.append(data.effort)
        self.joint_position.append(data.position)
        self.joint_vel.append(data.velocity)
        self.joint_name = data.name

    def _land_pos_get(self, data):
        if data.z <= 0.0:
            return None
        elif data.z >= 3.0 :
            return None
        elif abs(data.x) >= 10.0 :
            return None
        elif abs(data.y) >= 10.0 :
            return None
        self.land_pos.append([data.x, data.y, data.z])


    def get_name(self):
        return self.joint_name


if __name__ =="__main__":
    rospy.init_node("joint_reader")
    jb = JointBagReader()
    j , e, v = jb.joint_get("joint_bag", rospy.Time(0), rospy.Time(0), rospy.Time(0), 4)
    print len(j)
    print len(e)
    print len(v)
    print jb.get_name()