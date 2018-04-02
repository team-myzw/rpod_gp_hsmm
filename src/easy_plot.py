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
import numpy as np
import glob
import os
import pylab
import gc
import copy
import sympy as sp
from progressbar import ProgressBar
import rospy
import matplotlib.pyplot as plt
import pandas as pd

from sensor_msgs.msg import JointState
from rosbag_database.srv import RosbagPlay, RosbagPlayRequest
from rpod_gp_hsmm.srv import JointBag, JointBagRequest, JointBagResponse
from rpod_gp_hsmm.srv import JointRecog, JointRecogResponse
from sampling_joint import sampling
from sparse import sparse_dim_n
from std_srvs.srv import Empty, EmptyResponse, EmptyRequest
from joint_sampler import JointBagReader

FILE = "plot"

import sys


def mkdir(name):
    try:
        os.mkdir(FILE)
    except:
        pass
    try:
        os.mkdir(FILE+"/"+name)
    except:
        pass

if __name__=="__main__":
    rospy.init_node("plotter")
    args = sys.argv
    file_name = args[1]

    data = np.loadtxt(file_name, dtype=np.str)
    s = rospy.Time.now()
    jbr = JointBagReader()
    for i in range(len(data)):
        name = data[i][0]
        count = int(data[i][1])
        req = RosbagPlayRequest()
        req.name = name
        req.count_number = count
        req.duration = 60
        req.topics = "/hsr7/hand_states"
        j = jbr.data_get(req, JointState())
        csvs = []
        for joint in j :
            csvs.append(joint.position)
        csvs = np.array(csvs)
        nn = name.split("/")
        ns = nn[-1]
        mkdir(ns)
        df =pd.DataFrame(csvs,columns=joint.name)
        df.to_csv(FILE+"/"+ns+"/"+ns+".csv")
        ds = csvs.T
        for i in range(len(joint.name)):
            if "power" in joint.name[i]:
                continue
            if "q" in joint.name[i]:
                continue
            nl = range(len(ds[i]))
            plt.plot(nl,ds[i], label=joint.name[i])
            plt.title(joint.name[i])
            if "power" in joint.name[i]:
                plt.ylim(-40.0,40.0)
        plt.legend()
        plt.savefig(FILE+"/"+ns+"/"+ns+"_"+"Joint"+".jpg")
        plt.close()
                    
    rospy.loginfo(rospy.Time.now() - s)