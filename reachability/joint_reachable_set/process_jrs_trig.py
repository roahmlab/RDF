import torch
from reachability.conSet import batchZonotope
from reachability.joint_reachable_set.load_jrs_trig import JRS_KEY
from reachability.transformations.rotation import  gen_batch_rotatotope_from_jrs_trig

T_fail_safe = 0.5

cos_dim = 0 
sin_dim = 1
vel_dim = 2
ka_dim = 3
acc_dim = 3 
kv_dim = 4
time_dim = 5

def process_batch_JRS_trig(jrs_tensor, q_0,qd_0,joint_axes):
    dtype, device = jrs_tensor.dtype, jrs_tensor.device 
    q_0 = q_0.to(dtype=dtype,device=device)
    qd_0 = qd_0.to(dtype=dtype,device=device)
    jrs_key = torch.tensor(JRS_KEY['c_kvi'],dtype=dtype,device=device)
    n_joints = qd_0.shape[-1]
    PZ_JRS_batch = []
    R_batch = []
    for i in range(n_joints):
        closest_idx = torch.argmin(abs(qd_0[i]-jrs_key))
        JRS_batch_zono = batchZonotope(jrs_tensor[closest_idx])
        c_qpos = torch.cos(q_0[i])
        s_qpos = torch.sin(q_0[i])
        Rot_qpos = torch.tensor([[c_qpos,-s_qpos],[s_qpos,c_qpos]],dtype=dtype,device=device)
        A = torch.block_diag(Rot_qpos,torch.eye(4,dtype=dtype,device=device))
        JRS_batch_zono = A@JRS_batch_zono.slice(kv_dim,qd_0[i:i+1])
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