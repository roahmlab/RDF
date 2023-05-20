"""
Define class for rigid bodies to represent URDF data
Author: Yongseok Kwon
Reference: MATLAB robotics toolbox
"""
import sys
sys.path.append('..')
from environments.robots.utils import Euler_to_Rot,Rp_to_Trans
from torch import tensor, pi
from torch import float as torch_float
class rigidBodyTree:
    def __init__(self,links,joints,base,gravity=True):
        self.n_bodies = len(links)-1
        self.base = rigidBody(base)        
        self.base_name = base.name
        self.bodies = [rigidBody(body,joints) for body in links if body.name != base.name]
        self.body_name = [body.name for body in self.bodies]
        if gravity:
            self.gravity = tensor([0.0, 0.0, -9.81],dtype=torch_float)
        else:
            self.gravity = tensor([0.0, 0.0, 0.0],dtype=torch_float)
        self.__match_tree_id()

    def __repr__(self):
        gravity_list = [round(el,4) for el in self.gravity.tolist()]
        return f"""rigidBodyTree
 - base: {self.base_name} 
 - bodies: {self.n_bodies} rigidBodies
 - gravity: {gravity_list}"""
    __str__ = __repr__      
    
    def __getitem__(self,idx):
        if isinstance(idx,list):
            return [self.bodies[i-1] for i in idx]
        assert idx is None or idx >=0 
        if idx == 0:
            return self.base
        elif idx is None:
            return None
        else:
            return self.bodies[idx-1]

    def __match_tree_id(self):
        for i in range(self.n_bodies):
            for j in range(i+1,self.n_bodies):
                body1, body2 = [self.bodies[ind] for ind in [i,j]]
                
                if body1.name == body2.parent_name:
                    self.bodies[i].add_child(j+1)
                    self.bodies[j].add_parent(i+1)
                elif body1.parent_name == body2.name:
                    self.bodies[i].add_parent(j+1)
                    self.bodies[j].add_child(i+1)
                if body1.parent_name == self.base_name:
                    self.base.add_child(i+1)
                    self.bodies[i].add_parent(0)

class rigidBody:
    def __init__(self,body,joints=None):
        self.name = body.name
        self.__set_inertial(body.inertial)
        self.__add_joint(joints)
        self.parent_id = None
        self.children_id = []

    def __repr__(self):
        return f"rigidBody: {self.name}"
    def __str__(self):
        inertia_list = [[round(el,4) for el in row] for row in self.inertia.tolist()]
        com_list = [round(el,4) for el in self.com.tolist()]
        com_rot_list = [[round(el,4) for el in row] for row in self.com_rot.tolist()]
        str1 = f'rigidBody: {self.name}'
        str2 = f""" - mass: {self.mass}
 - inertia: {inertia_list}
 - com: {com_list}
 - orientation of com: {com_rot_list}
 - parent id: {self.parent_id}
 - children id: {self.children_id}"""
        if self.joint is None:
            joint_str = '\n - joint: None\n'
        else: 
            joint_str = f'\n - joint: {self.joint.name} ({self.joint.type})\n'
        return str1+joint_str+str2
    
    def __set_inertial(self,inertial):
        if inertial is None:
            self.mass = None
            self.inertia = tensor([],dtype=torch_float)
            self.com = tensor([],dtype=torch_float)
            self.com_rot = tensor([],dtype=torch_float)
        else:
            self.mass = inertial.mass
            self.inertia = tensor(inertial.inertia.to_matrix(),dtype=torch_float)
            if inertial.origin is not None:
                self.com = tensor(inertial.origin.xyz,dtype=torch_float)
                self.com_rot = tensor(Euler_to_Rot(inertial.origin.rpy),dtype=torch_float)
            else:
                self.com = tensor([0.0,0.0,0.0])
                self.com_rot = tensor([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    def __add_joint(self,joints):
        if joints is None:
            self.joint = None
            self.transform= tensor([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
            return
        for joint in joints:
            if joint.child == self.name:
                self.joint = rigidBodyJoint(joint)
                self.parent_name = joint.parent
                self.__set_transform(joint)
                break
    def __set_transform(self,joint):
        R = Euler_to_Rot(joint.origin.rpy)
        self.transform = tensor(Rp_to_Trans(R,joint.origin.xyz),dtype=torch_float)

    def add_parent(self,id):
        self.parent_id =  id
    def add_child(self,id):
        if id not in self.children_id:
            self.children_id.append(id)
    
class rigidBodyJoint:
    def __init__(self,joint):
        self.type = joint.type
        self.name = joint.name
        
        if self.type == 'fixed':
            self.axis = tensor([0.0,0.0,0.0])
            self.pos_lim = tensor([],dtype=torch_float)
            self.vel_lim = None
            self.f_lim = None
        else: 
            self.axis = tensor(joint.axis,dtype=torch_float)
            if self.type == 'continuous':
                self.pos_lim = tensor([pi, -pi])
            else:
                self.pos_lim = tensor([joint.limit.upper, joint.limit.lower],dtype=torch_float)
            self.vel_lim = joint.limit.velocity
            self.f_lim = joint.limit.effort
    def __repr__(self):
        return f"rigidBodyJoint: {self.name} ({self.type})"
    def __str__(self):
        axis_list = [round(el,4) for el in self.axis.tolist()]
        pos_lim_list = [round(el,4) for el in self.pos_lim.tolist()]        
        return f"rigidBodyJoint: {self.name} ({self.type}) \n - axis: {axis_list} \n - pos_lim: {pos_lim_list} \n - vel_lim: {self.vel_lim} \n - f_lim: {self.f_lim}"
        