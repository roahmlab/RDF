"""
Load trigonometry version of precomtuted joint reacheable set (precomputed by CORA)
Author: Yongseok Kwon
"""

import sys
sys.path.append('../..')
import torch
from reachability.transformations.rotation import gen_rotatotope_from_jrs_trig, gen_batch_rotatotope_from_jrs_trig
from reachability.conSet import zonotope, polyZonotope, batchZonotope
from scipy.io import loadmat
import os


T_fail_safe = 0.5

dirname = os.path.dirname(__file__)
jrs_path = os.path.join(dirname,'jrs_trig_mat_saved/')
jrs_tensor_path = os.path.join(dirname,'jrs_trig_tensor_saved/')

JRS_KEY = loadmat(jrs_path+'c_kvi.mat')
#JRS_KEY = torch.tensor(JRS_KEY['c_kvi'],dtype=torch.float)

'''
qjrs_path = os.path.join(dirname,'qjrs_mat_saved/')
qjrs_key = loadmat(qjrs_path+'c_kvi.mat')
qjrs_key = torch.tensor(qjrs_key['c_kvi'])
'''
cos_dim = 0 
sin_dim = 1
vel_dim = 2
ka_dim = 3
acc_dim = 3 
kv_dim = 4
time_dim = 5

def preload_batch_JRS_trig(dtype=torch.float,device='cpu'):
    jrs_tensor = []
    for c_kv in JRS_KEY['c_kvi'][0]:
        jrs_filename = jrs_tensor_path+'jrs_trig_tensor_mat_'+format(c_kv,'.3f')+'.mat'
        jrs_tensor_load = loadmat(jrs_filename)
        jrs_tensor.append(jrs_tensor_load['JRS_tensor'].tolist()) 
    return torch.tensor(jrs_tensor,dtype=dtype,device=device)


def load_batch_JRS_trig_ic(q_0,qd_0,joint_axes=None,dtype=torch.float,device='cpu'):
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    jrs_key = torch.tensor(JRS_KEY['c_kvi'],dtype=dtype,device=device)
    n_batches, n_joints = qd_0.shape
    PZ_JRS_batch = []
    R_batch = []
    if joint_axes is None:
        joint_axes = [torch.tensor([0.0,0.0,1.0],dtype=dtype,device=device) for _ in range(n_joints)]
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[:,i:i+1]-jrs_key),dim=-1)
        jrs_tensor = []
        for b in range(n_batches):
            jrs_filename = jrs_tensor_path+'jrs_trig_tensor_mat_'+format(jrs_key[0,closest_idx[b]],'.3f')+'.mat'
            jrs_tensor_load = loadmat(jrs_filename)
            jrs_tensor.append(torch.tensor(jrs_tensor_load['JRS_tensor'],dtype=dtype,device=device).unsqueeze(0))
        
        JRS_batch_zono = batchZonotope(torch.cat(jrs_tensor,0))
        c_qpos = torch.cos(q_0[:,i:i+1]).unsqueeze(-1)
        s_qpos = torch.sin(q_0[:,i:i+1]).unsqueeze(-1)
        A = (c_qpos*torch.tensor([[1.0]+[0]*5,[0,1]+[0]*4]+[[0]*6]*4,dtype=dtype,device=device) 
            + s_qpos*torch.tensor([[0,-1]+[0]*4,[1]+[0]*5]+[[0]*6]*4,dtype=dtype,device=device)
            + torch.tensor([[0.0]*6]*2+[[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],dtype=dtype,device=device))
        JRS_batch_zono = A.unsqueeze(1)@JRS_batch_zono.slice(kv_dim,qd_0[:,i:i+1].unsqueeze(1).repeat(1,100,1))
        PZ_JRS = JRS_batch_zono.deleteZerosGenerators(sorted=True).to_polyZonotope(ka_dim,prop='k_trig')

        '''
        delta_k = PZ_JRS.G[:,0,0,ka_dim]
        c_breaking = - qd_0[:,i]/T_fail_safe
        delta_breaking = - delta_k/T_fail_safe
        PZ_JRS.c[:,50:,acc_dim] = c_breaking.unsqueeze(-1)
        PZ_JRS.G[:,50:,0,acc_dim] = delta_breaking.unsqueeze(-1)
        '''
        R_temp= gen_batch_rotatotope_from_jrs_trig(PZ_JRS,joint_axes[i])
        PZ_JRS_batch.append(PZ_JRS)
        R_batch.append(R_temp)

    return PZ_JRS_batch, R_batch


def load_batch_JRS_trig(q_0,qd_0,joint_axes=None,dtype=torch.float,device='cpu'):
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    jrs_key = torch.tensor(JRS_KEY['c_kvi'],dtype=dtype,device=device)
    n_joints = qd_0.shape[-1]
    PZ_JRS_batch = []
    R_batch = []
    if joint_axes is None:
        joint_axes = [torch.tensor([0.0,0.0,1.0],dtype=dtype,device=device) for _ in range(n_joints)]
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[i]-jrs_key))
        jrs_filename = jrs_tensor_path+'jrs_trig_tensor_mat_'+format(jrs_key[0,closest_idx],'.3f')+'.mat'            
        jrs_tensor_load = loadmat(jrs_filename)
        jrs_tensor_load = torch.tensor(jrs_tensor_load['JRS_tensor'],dtype=dtype,device=device)
        JRS_batch_zono = batchZonotope(jrs_tensor_load)
        c_qpos = torch.cos(q_0[i])
        s_qpos = torch.sin(q_0[i])
        Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]],dtype=dtype,device=device)
        A = torch.block_diag(Rot_qpos,torch.eye(4,dtype=dtype,device=device))
        JRS_batch_zono = A@JRS_batch_zono.slice(kv_dim,qd_0[i])
        PZ_JRS = JRS_batch_zono.deleteZerosGenerators(sorted=True).to_polyZonotope(ka_dim,prop='k_trig')
        '''
        delta_k = PZ_JRS.G[0,0,ka_dim]
        c_breaking = - qd_0[i]/T_fail_safe
        delta_breaking = - delta_k/T_fail_safe
        PZ_JRS.c[50:,acc_dim] = c_breaking
        PZ_JRS.G[50:,0,acc_dim] = delta_breaking
        '''
        R_temp= gen_batch_rotatotope_from_jrs_trig(PZ_JRS,joint_axes[i])

        PZ_JRS_batch.append(PZ_JRS)
        R_batch.append(R_temp)
    return PZ_JRS_batch, R_batch


def load_JRS_trig(q_0,qd_0,joint_axes=None,dtype=torch.float,device='cpu'):
    '''
    load joint reachable set precomputed by MATLAB CORA (look gen_jrs).
    Then, operate loaded JRS zonotope into JRS polyzonotope w/ k-sliceable dep. gen. 
    for initial joint pos. and intial joint vel.

    qpos: <torch.Tensor> initial joint position
    , size [N]
    qvel: <torch.Tensor> initial joint velocity
    , size [N]

    return <dict>, <polyZonotope> dictionary of polynomical zonotope
    JRS_poly[t][i]
    - t: t-th timestep \in [0,99] -> 0 ~ 1 sec
    - i: i-th joint
    

    ** dimension index
    0: cos(qpos_i)
    1: sin(qpos_i)
    2: qvel_i
    3: ka_i
    4: kv_i
    5: t

    '''
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    jrs_key = torch.tensor(JRS_KEY['c_kvi'],dtype=dtype,device=device)
    if isinstance(q_0,list):
        q_0 = torch.tensor(q_0,dtype=dtype,device=device)
    if isinstance(qd_0,list):
        qd_0 = torch.tensor(qd_0,dtype=dtype,device=device)
    assert len(q_0.shape) == len(qd_0.shape) == 1 
    n_joints = len(qd_0)
    assert len(q_0) == n_joints
    if joint_axes is None:
        joint_axes = [torch.tensor([0.0,0.0,1.0],dtype=dtype,device=device) for _ in range(n_joints)]
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[i]-jrs_key))
        jrs_filename = jrs_path+'jrs_trig_mat_'+format(jrs_key[0,closest_idx],'.3f')+'.mat'
        
        jrs_mats_load = loadmat(jrs_filename)
        jrs_mats_load = jrs_mats_load['JRS_mat']
        n_time_steps = len(jrs_mats_load) # 100
        if i == 0:
            PZ_JRS = [[] for _ in range(n_time_steps)]
            R = [[] for _ in range(n_time_steps)]            
        for t in range(n_time_steps):
            c_qpos = torch.cos(q_0[i])
            s_qpos = torch.sin(q_0[i])
            Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]],dtype=dtype,device=device)
            A = torch.block_diag(Rot_qpos,torch.eye(4,dtype=dtype,device=device))
            JRS_zono_i = zonotope(torch.tensor(jrs_mats_load[t,0],dtype=dtype,device=device).squeeze(0))
            JRS_zono_i = A @ JRS_zono_i.slice(kv_dim,qd_0[i])
            PZ_JRS[t].append(JRS_zono_i.deleteZerosGenerators().to_polyZonotope(ka_dim,prop='k_trig'))
            '''
            # fail safe
            if t == 0:
                delta_k = PZ_JRS[0][i].G[0,ka_dim]
                c_breaking = - qd_0[i]/T_fail_safe
                delta_breaking = - delta_k/T_fail_safe
            elif t >= int(n_time_steps/2):
                PZ_JRS[t][i].c[acc_dim] = c_breaking
                PZ_JRS[t][i].G[0,acc_dim] = delta_breaking
            '''
            R_temp= gen_rotatotope_from_jrs_trig(PZ_JRS[t][i],joint_axes[i])
            R[t].append(R_temp)
    return PZ_JRS, R

def load_traj_JRS_trig(q_0, qd_0, uniform_bound, Kr, joint_axes = None,dtype=torch.float,device='cpu'):
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    n_joints = len(q_0)
    if joint_axes is None:
        joint_axes = [torch.tensor([0,0,1],dtype=dtype,device=device) for _ in range(n_joints)]
    PZ_JRS, R = load_JRS_trig(q_0,qd_0,joint_axes)
    n_time_steps = len(PZ_JRS)
    jrs_dim = PZ_JRS[0][0].dimension
    # Error Zonotope
    gain = Kr[0,0].reshape(1,1) # controller gain
    e = uniform_bound/ gain # max pos error
    d = torch.tensor([2*uniform_bound],dtype=dtype,device=device).reshape(1,1) # max vel error

    G_pos_err = torch.zeros(jrs_dim,2,dtype=dtype,device=device)
    G_pos_err[0,0] = (torch.cos(torch.zeros(1,dtype=dtype,device=device))-torch.cos(e,dtype=dtype,device=device))/2
    G_pos_err[1,1] = (torch.sin(e,dtype=dtype,device=device)-torch.sin(-e,dtype=dtype,device=device))/2

    G_vel_err = d

    # create trajectories
    q, qd, qd_a, qdd_a, r, R_t = [[[] for _ in range(n_time_steps)] for _ in range(6)]
    for t in range(n_time_steps):
        for i in range(n_joints):
            JRS = PZ_JRS[t][i]
            Z = JRS.Z
            C, G = Z[:,0], Z[:,1:]

            # desired traj.
            vel_C, vel_G = C[vel_dim].reshape(1), G[vel_dim].reshape(1,-1)
            vel_G = vel_G[:,torch.any(vel_G,axis=0)]

            acc_C, acc_G = C[acc_dim].reshape(1), G[acc_dim].reshape(1,-1)
            acc_G = acc_G[:,torch.any(acc_G,axis=0)]

            # actual traj.
            q[t].append(polyZonotope(C,G[:,0],torch.hstack((G[:,1:],G_pos_err))))
            qd[t].append(polyZonotope(vel_C,torch.hstack((vel_G,G_vel_err))))
            
            # modified trajectories
            qd_a[t].append(polyZonotope(vel_C, torch.hstack((vel_G,gain*e))))
            qdd_a[t].append(polyZonotope(acc_C, torch.hstack((acc_G,gain*d))))
            r[t].append(polyZonotope(torch.zeros(1,dtype=torch.float),torch.hstack((d,gain*e))))
            
            R_t[t].append(R[t][i].T)
    # return q_des, qd_des, qdd_des, q, qd, qd_a, qdd_a, r, c_k, delta_k, id, id_names
    return q, qd, qd_a, qdd_a, R, R_t #, r, c_k, delta_k
if __name__ == '__main__':
    import time 
    ts = time.time()
    Z = preload_batch_JRS_trig()
    print(time.time()-ts) 
    import pdb;pdb.set_trace()