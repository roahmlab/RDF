"""
Load parameters from URDF file and parse data to continuous set to represent uncertainties
Author: Yongseok Kwon
Reference: MATLAB robotics toolbox
"""
import sys
sys.path.append('..')
from environments.robots.urdf_parser_py.urdf import URDF
from environments.robots.tree import rigidBodyTree
from environments.robots.utils import parellel_axis
import torch
import os
import copy
from mat4py import loadmat
from reachability.conSet import zonotope, polyZonotope, matPolyZonotope

dirname = os.path.dirname(__file__)
ROBOTS_PATH = os.path.join(dirname,'assets/robots')
LINK_ZONO_PATH = os.path.join(dirname,'link_zonotopes')
ROBOT_ARM_PATH = {'Fetch': 'fetch_arm/fetch_arm_reduced.urdf',
                  'Kinova3': 'kinova_arm/gen3.urdf',
                  'Kuka': 'kuka_arm/kuka_iiwa_arm.urdf',
                  'UR5': 'ur5_robot.urdf'
                  }

JOINT_TYPE_MAP = {'revolute': 'revolute',
                  'continuous': 'revolute',
                  'prismatic': 'prismatic',
                  'fixed': 'fixed'}

def import_robot(urdf_file,gravity=True):
    if urdf_file in ROBOT_ARM_PATH.keys():
        urdf_file = ROBOT_ARM_PATH[urdf_file]
        urdf_path = os.path.join(ROBOTS_PATH,urdf_file)
    else:
        urdf_path = urdf_file
    # parse URDF file
    robot_urdf = URDF.from_xml_string(open(urdf_path,'rb').read())
    #robot_urdf = URDF.from_xml_file(urdf_path)
    # order link and joints
    link_map = robot_urdf.link_map
    joints = robot_urdf.joints
    n_joints = len(joints)
    has_root = [True for _ in range(n_joints)]
    for i in range(n_joints):
        for j in range(i+1,n_joints):
            if joints[i].parent == joints[j].child:
                has_root[i] = False
            elif joints[j].parent == joints[i].child:
                has_root[j] = False
    
    for i in range(n_joints):
        if has_root[i]: 
            base = link_map[joints[i].parent]
    robot = rigidBodyTree(robot_urdf.links,joints,base)
    return robot

def load_sinlge_robot_arm_params(urdf_file,gravity=True):
    '''
    Assumed all the active joint is revolute.
    '''
    
    link_path = os.path.join(LINK_ZONO_PATH,urdf_file)
    is_link_file = os.path.isfile(os.path.join(link_path,'link_zonotope_0.mat'))
    robot = import_robot(urdf_file,gravity)

    params = {}
    mass = [] # mass
    I = [] # momnet of inertia
    G = [] # spatial inertia
    com = [] # CoM position
    com_rot = [] # CoM orientation
    joint_axes = [] # joint axes
    H = [] # transform of ith joint in pre. joint in home config.
    R = [] # rotation of    "   "
    P = [] # translation of    "   "  
    M = [] # transform of ith CoM in prev. CoM in home config.
    screw = [] # screw axes in base
    pos_lim = [] # joint position limit
    vel_lim = [] # joint velocity limit
    tor_lim = [] # joint torque limit
    lim_flag = [] # False for continuous, True for everything else

    link_zonos = [] # link zonotopes
    
    Tj = torch.eye(4,dtype=torch.float) # transform of ith joint in base
    K = torch.eye(4,dtype=torch.float) # transform of ith CoM in ith joint
    
    body = robot[robot.base.children_id[0]]
    for i in range(robot.n_bodies):
        if is_link_file:
            Z = loadmat(os.path.join(link_path,f'link_zonotope_{i+1}.mat'))
            link_zonos.append(zonotope(Z['Z']))
        
        mass.append(body.mass)
        I.append(body.inertia)
        #import pdb; pdb.set_trace()
        G.append(torch.block_diag(body.inertia,body.mass*torch.eye(3)))
        com.append(body.com)
        com_rot.append(body.com_rot)
        joint_axes.append(torch.Tensor(body.joint.axis))
        H.append(body.transform)
        R.append(body.transform[:3,:3])
        P.append(body.transform[:3,3])

        Tj = Tj @ body.transform # transform of ith joint in base
        K_prev = K
        K = torch.eye(4)
        K[:3,:3],K[:3,3] = body.com_rot, body.com 
        M.append(torch.inverse(K_prev)@body.transform@K)

        w = Tj[:3,:3] @ body.joint.axis
        v = torch.cross(-w,Tj[:3,3])
        screw.append(torch.hstack((w,v)))
        
        pos_lim.append(body.joint.pos_lim.tolist())
        vel_lim.append(body.joint.vel_lim)
        tor_lim.append(body.joint.f_lim)
        lim_flag.append(body.joint.type!='continuous')
        if len(body.children_id)!=1 or robot[body.children_id[0]].joint.type == 'fixed':
            n_joints = i+1
            break
        else:
            body = robot[body.children_id[0]]

    params = {'mass':mass, 'I':I, 'G':G, 'com':com, 'com_rot':com_rot, 'joint_axes':joint_axes,
    'H':H, 'R':R, 'P':P, 'M':M, 'screw':screw,
    'pos_lim':pos_lim, 'vel_lim':vel_lim, 'tor_lim':tor_lim, 'lim_flag':lim_flag,
    'n_bodies': robot.n_bodies, 'n_joints': n_joints,
    'gravity': robot.gravity, 'use_interval': False, 'link_zonos':link_zonos
    }
    return params, robot