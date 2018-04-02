# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 17:15:48 2016

@author: robocup
"""
import numpy as np

def sampling_all(use_joint, position, effort, velocity, joint_names):
    names = []
    
    for j in range(len(joint_names)):
        if not joint_names[j] in use_joint:
            names.append(j)
        
    pos = np.array(position)
    eff = np.array(effort)
    vel = np.array(velocity)
    
    if not names != []:
        pos = np.delete(pos, names, 1)
        eff = np.delete(eff, names, 1)
        vel = np.delete(vel, names, 1)

    data = np.c_[pos,eff,vel]
    return data


def sampling(use_joint, position,joint_names):
    names = []
    
    for j in range(len(joint_names)):
        if not joint_names[j] in use_joint:
            names.append(j)
        
    pos = np.array(position)
    
    if not names != []:
        pos = np.delete(pos, names, 1)
    return pos
