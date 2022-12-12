#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:51:40 2020

@author: miguel-asd
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
import torch
import pygad.torchga
#import matplotlib.pyplot as plt
#import csv

from src.kinematic_model import robotKinematics
from src.pybullet_debuger import pybulletDebug  
from src.gaitPlanner import trotGait
from src.sim_fb import systemStateEstimator


def rendering(render):
    """Enable/disable rendering"""
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, render)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, render)

def robot_init( dt, body_pos, fixed = True , connect = p.GUI):
    physicsClient = p.connect(connect)#p.GUI or p.DIRECT for non-graphical version
    # turn off rendering while loading the models
    rendering(0)

    p.setGravity(0,0,-10)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(
        fixedTimeStep=dt,
        numSolverIterations=100,
        enableFileCaching=0,
        numSubSteps=1,
        solverResidualThreshold=1e-10,
        erp=1e-1,
        contactERP=0.01,
        frictionERP=0.01,
    )
    # add floor
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.loadURDF("plane.urdf")
    # add robot
    body_id = p.loadURDF("4leggedRobot.urdf", body_pos, useFixedBase=fixed)
    joint_ids = []
    
    #robot properties

    maxVel = 3.703 #rad/s
    for j in range(p.getNumJoints(body_id)):
        p.changeDynamics( body_id, j, lateralFriction=1e-5, linearDamping=0, angularDamping=0)
        p.changeDynamics( body_id, j, maxJointVelocity=maxVel)
        joint_ids.append( p.getJointInfo(body_id, j))
#        info = p.getJointInfo( body_id, j )
#        joint_name = info[1].decode('UTF-8')
#        link_name = info[12].decode('UTF-8')
#        print(joint_name,link_name)

    # start record video
    #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robot.mp4")
    rendering(1)
    
    


    
    return body_id, joint_ids


def original_move_joints(body_id , angles):
    maxForce = 2 #N/m
    #move movable joints
    # print("Angles:", angles)
    for i in range(3):
        p.setJointMotorControl2(body_id, i, p.POSITION_CONTROL, 
                                targetPosition = angles[0,i] , force = maxForce)
        p.setJointMotorControl2(body_id, 4 + i, p.POSITION_CONTROL, 
                                targetPosition = angles[1,i] , force = maxForce)
        p.setJointMotorControl2(body_id, 8 + i, p.POSITION_CONTROL, 
                                targetPosition = angles[2,i] , force = maxForce) 
        p.setJointMotorControl2(body_id, 12 + i, p.POSITION_CONTROL, 
                                targetPosition = angles[3,i] , force = maxForce)

def move_joints(body_id , angles):
    maxForce = 2 #N/m
    #move movable joints
    # print("Angles:", angles)
    for i in range(3):
        p.setJointMotorControl2(body_id, i, p.POSITION_CONTROL, 
                                targetPosition = angles[0+i*4] , force = maxForce)
        p.setJointMotorControl2(body_id, 4 + i, p.POSITION_CONTROL, 
                                targetPosition = angles[1+i*4] , force = maxForce)
        p.setJointMotorControl2(body_id, 8 + i, p.POSITION_CONTROL, 
                                targetPosition = angles[2+i*4] , force = maxForce) 
        p.setJointMotorControl2(body_id, 12 + i, p.POSITION_CONTROL, 
                                targetPosition = angles[3+i*4] , force = maxForce)
        
def get_paws_poses(body_id):
    angles = list()
    for i in range(3):
        angles.append(p.getJointState(body_id, i)[0])
        angles.append(p.getJointState(body_id, i+4)[0])
        angles.append(p.getJointState(body_id, i+8)[0])
        angles.append(p.getJointState(body_id, i+12)[0])
    # print("GetPaws:", angles)
    return angles
        
        
def robot_stepsim( body_id, body_pos, body_orn, body2feet ):
    angles, body2feet_ = robotKinematics.solve( body_orn, body_pos, body2feet)
    original_move_joints(body_id, angles)



    


def robot_quit():
    p.disconnect()





##This part of code is just to save the raw telemetry data.
"""
fieldnames = ["t","FR","FL","BR","BL"]
with open('telemetry/data.csv','w') as csv_file:
    csv_writer = csv.DictWriter(csv_file,fieldnames = fieldnames)
    csv_writer.writeheader()
"""

"""
def update_data():
    #take meassurement from simulation
    t , X = meassure.states()
    U , Ui ,torque = meassure.controls()
    
    with open('telemetry/data.csv','a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        info = {"t" : t[-1],
                "FR" : Ui[2,0],
                "FL" : Ui[5,0],
                "BR" : Ui[8,0],
                "BL" : Ui[11,0]}
        csv_writer.writerow(info)
"""

#"""
def sim(solution, sol_idx, control_model, gui = False):
    dT = 0.002
    debugMode = "STATES"
    
    bodyId, jointIds = robot_init( dt = dT, body_pos = [0,0,0.13], fixed = False , connect = p.DIRECT if not gui else p.GUI)
    # printear joints para ver cual es la del cuerpo.
    meassure = systemStateEstimator(bodyId)

    #initial foot position
    #foot separation (Ydist = 0.16 -> tetta=0) and distance to floor
    Xdist, Ydist, height = 0.18, 0.15, 0.10
    #body frame to foot frame vector
    bodytoFeet0 = np.matrix([[ Xdist/2. , -Ydist/2. , height],
                            [ Xdist/2. ,  Ydist/2. , height],
                            [-Xdist/2. , -Ydist/2. , height],
                            [-Xdist/2. ,  Ydist/2. , height]])
    

    offset = np.array([0.5 , 0. , 0. , 0.5]) #defines the offset between each foot step in this order (FR,FL,BR,BL)
    footFR_index, footFL_index, footBR_index, footBL_index = 3, 7, 11, 15
    T = 0.5 #period of time (in seconds) of every step
 
    N_steps=1000 # 1000 iter each 5 secs (aprox.
    N_par = 8

    # start record video
    if gui:
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robot.mp4")

    robot_pos = list(p.getLinkState(bodyId, 0)[0]) #pos
    robot_orn = list(p.getLinkState(bodyId, 0)[1]) #orn
    paws_angs = get_paws_poses(bodyId)
    model_input = torch.tensor(robot_pos + robot_orn + paws_angs)
    score = 0
    for k_ in range(0,N_steps):
        print(k_)
        #MAIN LOOP
        #lastTime = time.time()
        

        model_out = pygad.torchga.predict(model=control_model,
                            solution=solution,
                            data=model_input)

        # entradas de la red (18): robot_pos, robot_orn, paws_angs
        # salidas de la red (12): paws_angs_out
        # print(model_out.tolist())
        move_joints(bodyId, model_out.tolist())

        robot_pos = list(p.getLinkState(bodyId, 0)[0]) #pos
        robot_orn = list(p.getLinkState(bodyId, 0)[1]) #orn
        paws_angs = get_paws_poses(bodyId)
        model_input = torch.tensor(robot_pos + robot_orn + paws_angs)
        #update_data() # debug data
        p.stepSimulation()
        #print(time.time() - lastTime)

        vel_vec = list(p.getLinkState(bodyId, 0, computeLinkVelocity = True)[6])


        den = (2 * abs(3 - robot_pos[0]) + abs(robot_pos[1]) - min(0, 0.3 - robot_pos[2]) - 3 * min(0, 0.2 - vel_vec[0])) # 0.115 original Z value when standing still
        if den == 0:
            score += 999999999
        else:
            score += 1 / den
        
    robot_quit()

    return score

#"""


if __name__ == '__main__':
    dT = 0.002
    debugMode = "STATES"
    kneeConfig = "><"
    robotKinematics = robotKinematics(kneeConfig) # "><" or "<<"
    trot = trotGait()
    
    bodyId, jointIds = robot_init( dt = dT, body_pos = [0,0,0], fixed = False , connect = p.GUI)
    # printear joints para ver cual es la del cuerpo.
    pybulletDebug = pybulletDebug(debugMode)
    meassure = systemStateEstimator(bodyId)

    """initial foot position"""
    #foot separation (Ydist = 0.16 -> tetta=0) and distance to floor
    Xdist, Ydist, height = 0.18, 0.15, 0.10
    #body frame to foot frame vector
    bodytoFeet0 = np.matrix([[ Xdist/2. , -Ydist/2. , height],
                            [ Xdist/2. ,  Ydist/2. , height],
                            [-Xdist/2. , -Ydist/2. , height],
                            [-Xdist/2. ,  Ydist/2. , height]])
    

    offset = np.array([0.5 , 0. , 0. , 0.5]) #defines the offset between each foot step in this order (FR,FL,BR,BL)
    footFR_index, footFL_index, footBR_index, footBL_index = 3, 7, 11, 15
    T = 0.5 #period of time (in seconds) of every step
 
    N_steps=1000000 # 1000 iter each 5 secs (aprox.)
    N_par = 8

    # start record video
    #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "robot.mp4")
    for k_ in range(0,N_steps):
        #MAIN LOOP
        lastTime = time.time()
        
        pos , orn , L , Lrot , angle , T , sda = pybulletDebug.cam_and_robotstates(bodyId)
        


        robot_pos = list(p.getLinkState(bodyId, 0)[0]) #pos
        robot_orn = p.getLinkState(bodyId, 0)[1] #orn
        #print("pos,\torn,\tL,\tLrot,\tangle,\tT,\tsda")
        #print(pos, "\t", orn, "\t")
        #print(robot_pos[2])
        print(list(p.getLinkState(bodyId, 0, computeLinkVelocity = True)[6]))
        bodytoFeet = trot.loop( L , angle , Lrot , T , offset , bodytoFeet0 , sda)
        robot_stepsim( bodyId, pos, orn, bodytoFeet )
        
        #robot_stepsim_net(bodyId, pos , orn , angle) #our function
            
        #update_data() # debug data
        
        p.stepSimulation()
        #print(time.time() - lastTime)
    robot_quit()



