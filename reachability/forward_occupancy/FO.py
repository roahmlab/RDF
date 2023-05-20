"""
Forward Occupancy Algorithm
Author: Yongseok Kwon
"""

import sys
sys.path.append('../..')
import torch
from reachability.conSet import polyZonotope, matPolyZonotope

def forward_occupancy(rotatos,link_zonos,robot_params,zono_order = 2):
    '''
    P: <list>
    '''    
    dtype, device = rotatos[0].dtype, rotatos[0].device
    n_joints = robot_params['n_joints']
    P = robot_params['P']
    R = robot_params['R']
    P_motor = [polyZonotope(torch.zeros(3,dtype=dtype,device=device).unsqueeze(0))]
    R_motor = [matPolyZonotope(torch.eye(3,dtype=dtype,device=device).unsqueeze(0))]
    FO_link = []
    for i in range(n_joints):
        P_motor_temp = R_motor[-1]@P[i] + P_motor[-1]
        P_motor.append(P_motor_temp.reduce_indep(zono_order))
        R_motor_temp = R_motor[-1]@R[i]@rotatos[i]
        R_motor.append(R_motor_temp.reduce_indep(zono_order))
        FO_link_temp = R_motor[-1]@link_zonos[i] + P_motor[-1]
        FO_link.append(FO_link_temp.reduce_indep(zono_order))
    return FO_link, R_motor[1:], P_motor[1:]

