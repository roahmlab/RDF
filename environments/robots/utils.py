"""
Utilities for parsing URDF data
Author: Yongseok Kwon
Reference:
"""
from torch import tensor, cos, sin

def Euler_to_Rot(rpy):
    '''
    roll-pitch-yaw
    '''
    C = [cos(tensor(th)) for th in rpy]
    S = [sin(tensor(th)) for th in rpy]
    Rot = [[0,0,0],[0,0,0],[0,0,0]]
    Rot[0][0] = C[1]*C[2]
    Rot[0][1] = S[0]*S[1]*C[2] - C[0]*S[2] 
    Rot[0][2] = C[0]*S[1]*C[2] + S[0]*S[2] 
    Rot[1][0] = C[1]*S[2]
    Rot[1][1] = S[0]*S[1]*S[2] + C[0]*C[2] 
    Rot[1][2] = C[0]*S[1]*S[2] - S[0]*C[2] 
    Rot[2][0] = -S[1]
    Rot[2][1] = S[0]*C[1]
    Rot[2][2] = C[0]*C[1]
    return Rot
def Rp_to_Trans(R,p):
    return [R[i]+[p[i]] if i<3 else [0,0,0,1] for i in range(4)]


def parellel_axis(Ii, mass, R, p):
    skew = tensor([[0.0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]])
    Io = R@Ii@R.T + mass*(skew@skew.T)
    return Io
