"""
Define rotatotope
Author: Yongseok Kwon
"""

import sys
sys.path.append('../..')
import torch
from reachability.conSet import matPolyZonotope, batchMatPolyZonotope

cos_dim = 0
sin_dim = 1
vel_dim = 2
acc_dim = 3
k_dim = 3


def gen_batch_rotatotope_from_jrs_trig(bPZ,rot_axis):
    dtype, device = bPZ.dtype, bPZ.device
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],dtype=dtype,device=device)
    cosq = bPZ.c[bPZ.batch_idx_all+(slice(cos_dim,cos_dim+1),)].unsqueeze(-1)
    sinq = bPZ.c[bPZ.batch_idx_all+(slice(sin_dim,sin_dim+1),)].unsqueeze(-1)
    C = torch.eye(3,dtype=dtype,device=device) + sinq*w_hat + (1-cosq)*w_hat@w_hat
    cosq = bPZ.Z[bPZ.batch_idx_all+(slice(1,None),slice(cos_dim,cos_dim+1))].unsqueeze(-1)
    sinq = bPZ.Z[bPZ.batch_idx_all+(slice(1,None),slice(sin_dim,sin_dim+1))].unsqueeze(-1)
    G = sinq*w_hat - cosq*(w_hat@w_hat)
    return batchMatPolyZonotope(torch.cat((C.unsqueeze(-3),G),-3),bPZ.n_dep_gens,bPZ.expMat,bPZ.id,compress=0)

def gen_rotatotope_from_jrs_trig(polyZono,rot_axis):
    '''
    polyZono: <polyZonotope>
    rot_axis: <torch.Tensor>
    '''
    dtype, device = polyZono.dtype, polyZono.device
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],dtype=dtype,device=device)

    cosq = polyZono.c[cos_dim]
    sinq = polyZono.c[sin_dim]
    # Rodrigues' rotation formula
    C = (torch.eye(3,dtype=dtype,device=device) + sinq*w_hat + (1-cosq)*w_hat@w_hat).unsqueeze(0)
    cosq = polyZono.Z[1:,cos_dim:cos_dim+1].unsqueeze(-1)
    sinq = polyZono.Z[1:,sin_dim:sin_dim+1].unsqueeze(-1)
    G = sinq*w_hat - cosq*(w_hat@w_hat)
    return matPolyZonotope(torch.vstack((C,G)),polyZono.n_dep_gens,polyZono.expMat,polyZono.id,compress=0)

def gen_rot_from_q(q,rot_axis):
    if isinstance(q,(int,float)):
        q = torch.tensor(q,dtype=torch.float)
    dtype, device = q.dtype, q.device
    cosq = torch.cos(q,dtype=dtype,device=device)
    sinq = torch.sin(q,dtype=dtype,device=device)
    # normalize
    w = rot_axis/torch.norm(rot_axis)
    # skew-sym. mat for cross prod 
    w_hat = torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],dtype=dtype,device=device)
    # Rodrigues' rotation formula
    Rot = torch.eye(3,dtype=dtype,device=device) + sinq*w_hat + (1-cosq)*w_hat@w_hat
    return Rot