"""
Utilities for polynomial zonotope and matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA
"""
import torch
def removeRedundantExponentsBatch(ExpMat,G,batch_idx_all,dim_N=2):
    '''
    add up all generators that belong to terms with identical exponents
    
    ExpMat: <torch.tensor> matrix containing the exponent vectors
    G: <torch.tensor> generator matrix
    
    return,
    ExpMat: <torch.tensor> modified exponent matrix
    G: <torch.tensor> modified generator matrix
    '''
    # True for non-zero generators
    # NOTE: need to fix or maybe G should be zero tensor for empty

    batch_shape = G.shape[:-dim_N]
    idxD = torch.sum(G!=0,tuple(range(len(batch_shape)))+tuple(range(-1,-dim_N,-1)))!=0 # non-zero generator index
    # skip if all non-zero OR G is non-empty 
    if not idxD.all() or G.shape[-dim_N] == 0:
        # if all generators are zero
        if (~idxD).all():
            Gnew = torch.tensor([],dtype=G.dtype,device=G.device).reshape(batch_shape+(0,)+G.shape[1-dim_N:])
            ExpMatNew = torch.tensor([],dtype=ExpMat.dtype,device=G.device).reshape((0,)+ExpMat.shape[1:])
            return ExpMatNew, Gnew
        else:
            # keep non-zero genertors
            G = G[batch_idx_all+(idxD,)]
            ExpMat = ExpMat[idxD]
    # add hash value of the exponent vector to the exponent matrix
    temp = torch.arange(ExpMat.shape[1],device=G.device).reshape(-1,1) + 1
    rankMat = torch.hstack((ExpMat.to(dtype=torch.float)@temp.to(dtype=torch.float),ExpMat))
    # sort the exponents vectors according to the hash value
    ind = torch.unique(rankMat,dim=0,sorted=True,return_inverse=True)[1].argsort()

    ExpMatTemp = ExpMat[ind]
    Gtemp = G[batch_idx_all+(ind,)]
    
    # vectorized 
    ExpMatNew, ind_red = torch.unique_consecutive(ExpMatTemp,dim=0,return_inverse=True)
    if ind_red.max()+1 == ind_red.numel():
        return ExpMatTemp,Gtemp

    n_rem = ind_red.max()+1
    ind = torch.arange(n_rem,device=G.device).unsqueeze(1) == ind_red 
    num_rep = ind.sum(1)
    Gtemp2 = Gtemp.repeat((1,)*len(batch_shape)+(n_rem,)+(1,)*(dim_N-1))[batch_idx_all + (ind.reshape(-1),)].cumsum(-dim_N)    
    Gtemp2 = torch.cat((torch.zeros(batch_shape+(1,)+Gtemp2.shape[1-dim_N:],device=G.device),Gtemp2),-dim_N)
    
    num_rep2 = torch.hstack((torch.zeros(1,dtype=torch.long,device=G.device),num_rep.cumsum(0)))
    Gnew = (Gtemp2[batch_idx_all +(num_rep2[1:],)] - Gtemp2[batch_idx_all +(num_rep2[:-1],)])
    return ExpMatNew, Gnew


def removeRedundantExponents(ExpMat,G):
    '''
    add up all generators that belong to terms with identical exponents
    
    ExpMat: <torch.tensor> matrix containing the exponent vectors
    G: <torch.tensor> generator matrix
    
    return,
    ExpMat: <torch.tensor> modified exponent matrix
    G: <torch.tensor> modified generator matrix
    '''
    # True for non-zero generators
    # NOTE: need to fix or maybe G should be zero tensor for empty

    dim_G = len(G.shape)
    idxD = torch.sum(G!=0,tuple(range(-1,-dim_G,-1)))!=0 # non-zero generator index

    '''
    idxD = torch.any(abs(G)>eps,)
    for _ in range(dim_G-2):
        idxD = torch.any(idxD,-1)
    '''
    # skip if all non-zero OR G is non-empty 
    if not all(idxD) or G.shape[0] == 0:
        # if all generators are zero
        if all(~idxD):
            Gnew = torch.tensor([],dtype=G.dtype,device=G.device).reshape((0,)+G.shape[1:])
            ExpMatNew = torch.tensor([],dtype=ExpMat.dtype,device=G.device).reshape((0,)+ExpMat.shape[1:])
            return ExpMatNew, Gnew
        else:
            # keep non-zero genertors
            G = G[idxD]
            ExpMat = ExpMat[idxD]
    '''
    hash = torch.hstack((torch.tensor(1),(ExpMat.max(0).values+1).cumprod(0)))[:-1]*ExpMat
    hash = hash.sum(1)
    ind = torch.unique(hash,dim=0,sorted=True,return_inverse=True)[1].argsort()

    '''
    # add hash value of the exponent vector to the exponent matrix
    temp = torch.arange(ExpMat.shape[1],device=G.device).reshape(-1,1) + 1
    rankMat = torch.hstack((ExpMat.to(dtype=torch.float)@temp.to(dtype=torch.float),ExpMat))
    # sort the exponents vectors according to the hash value
    ind = torch.unique(rankMat,dim=0,sorted=True,return_inverse=True)[1].argsort()

    ExpMatTemp = ExpMat[ind]
    Gtemp = G[ind]
    
    # vectorized 
    ExpMatNew, ind_red = torch.unique_consecutive(ExpMatTemp,dim=0,return_inverse=True)
    if ind_red.max()+1 == ind_red.numel():
        return ExpMatTemp,Gtemp

    n_rem = ind_red.max()+1
    ind = torch.arange(n_rem,device=G.device).unsqueeze(1) == ind_red 
    num_rep = ind.sum(1)

    Gtemp2 = Gtemp.repeat((n_rem,)+(1,)*(dim_G-1))[ind.reshape(-1)].cumsum(0)
    Gtemp2 = torch.cat((torch.zeros((1,)+Gtemp2.shape[1:],dtype=G.dtype,device=G.device),Gtemp2),0)
    
    num_rep2 = torch.hstack((torch.zeros(1,dtype=torch.long,device=G.device),num_rep.cumsum(0)))
    Gnew = (Gtemp2[num_rep2[1:]] - Gtemp2[num_rep2[:-1]])
    return ExpMatNew, Gnew

def mergeExpMatrix(id1, id2, expMat1, expMat2):
    '''
    Merge the ID-vectors of two polyZonotope and adapt the  matrice accordingly
    id1: <>,
    id2:
    expMat1: <>
    expMat1: <>
    
    return,
    id: <>, merged ID-vector
    expMat1: <>, adapted exponent matric of the first polynomial zonotope
    expMat2: <>, adapted exponent matric of the second polynomial zonotope
    '''
    L1 = len(id1)
    L2 = len(id2)

    # ID vectors are identical
    if L1 == L2 and all(id1==id2):
        id = id1

    # ID vectors not identical -> MERGE
    else:
        ind2 =torch.zeros_like(id2)

        Ind_rep = id2.reshape(-1,1) == id1
        ind = torch.any(Ind_rep,axis=1)
        non_ind = ~ind
        ind2[ind] = Ind_rep.nonzero()[:,1]
        ind2[non_ind] = torch.arange(non_ind.sum(),device=non_ind.device) + len(id1)
        id = torch.hstack((id1,id2[non_ind]))
        # construct the new exponent matrices
        L = len(id)
        expMat1 = torch.hstack((expMat1,torch.zeros(len(expMat1),L-L1,dtype=expMat1.dtype,device=expMat1.device)))
        temp = torch.zeros(len(expMat2),L,dtype=expMat1.dtype,device=expMat1.device)
        temp[:,ind2] = expMat2
        expMat2 = temp

    return id, expMat1, expMat2
