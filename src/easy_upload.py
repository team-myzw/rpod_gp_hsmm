#!/usr/bin/env python
# encoding: utf8
import numpy as np
import rospy

from rpod_gp_hsmm.srv import JointBag, JointBagRequest
from std_srvs.srv import Trigger, TriggerRequest

import sys
NUMBER=1

if __name__=="__main__":
    print "wait now"
    print "start"
    rospy.init_node("work{}".format(NUMBER))
    args = sys.argv
    file_name = args[1]
    data = np.loadtxt(file_name, dtype=np.str)
    if len(data.shape) == 1:
        n = 1
    else:
        n = len(data)
    srv = rospy.ServiceProxy("joint_upload{}".format(NUMBER), JointBag)
    s = rospy.Time.now()
    for i in range(n):
        req = JointBagRequest()
        if len(data.shape) == 1:                            
            name = data[0]
        else:
            name = data[i][0]
        if len(data.shape) == 1:                            
            count = int(data[1])
        else:
            count = int(data[i][1])
        req.count_number = count
        req.duration_secs = 600
        req.name = name
        req.read_topics = ["/hsr7/hand_states", "/hsr7/object_pos_time"]
        srv.call(req)
        rospy.loginfo("call")
