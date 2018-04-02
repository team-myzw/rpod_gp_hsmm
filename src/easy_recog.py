#!/usr/bin/env python
# encoding: utf8
import numpy as np
import rospy

from rpod_gp_hsmm.srv import JointBag, JointBagRequest
from std_srvs.srv import Trigger, TriggerRequest

import sys
NUMBER=0
N = 10
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
    for _j in range(N):
        print n
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
            try:
                if len(data.shape) == 1:                            
                    start = rospy.Time.from_sec(float(data[2]))
                else:
                    start = rospy.Time.from_sec(float(data[i][2]))
            except:
                start = rospy.Time(0)

            try:
                if len(data.shape) == 1:                            
                    end = rospy.Time.from_sec(float(data[3]))
                else:
                    end = rospy.Time.from_sec(float(data[i][3]))
            except:
                end = rospy.Time(0)
            req.count_number = count
            req.duration_secs = 600
            req.name = name
            req.start_time = start
            req.end_time = end
            req.read_topics = ["/hsr7/hand_states"]
            print req
            srv.call(req)
        rospy.loginfo("call")
        sv = rospy.ServiceProxy("start_gphsmm_recog".format(NUMBER),Trigger)
        re = TriggerRequest()
        res = sv.call(re)
        rospy.loginfo(rospy.Time.now() - s)
        rospy.loginfo(res)
        print "######pause#####"
        sys.exit(1)
#        print raw_input()