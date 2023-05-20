"""
Define class for batch matrix zonotope
Author: Yongseok Kwon
"""

import sys
sys.path.append('../..')
from reachability.conSet.zonotope.mat_zono import matZonotope
from reachability.conSet.zonotope.batch_zono import batchZonotope
from reachability.conSet.zonotope.utils import pickedBatchGenerators
import torch

class batchMatZonotope():
    '''
    matZono: <matZonotope>, <torch.float64>

    Z: <torch.Tensor> center vector and generator matrix Z = [c,G]
    , shape [N+1, nx, ny]
    center: <torch.Tensor> center matrix
    , shape [nx,ny] 
    generators: <torch.Tensor> generator tensor
    , shape [N, nx, ny]
    
    
    Eq. (coeff. a1,a2,...,aN \in [0,1])
    G = [[G1],[G2],...,[GN]]
    zono = C + a1*G1 + a2*G2 + ... + aN*GN
    '''
    def __init__(self,Z):
        if isinstance(Z,list):
            Z = torch.tensor(Z)
        assert isinstance(Z,torch.Tensor), f'The input matrix should be torch tensor, but {type(Z)}.'
        assert len(Z.shape) > 3, f'The dimension of Z input should be either 2 or 3, but {len(Z.shape)}.'
        self.Z = Z
        self.batch_dim = len(Z.shape) - 3
        self.batch_idx_all = tuple([slice(None) for _ in range(self.batch_dim)])
    def __getitem__(self,idx):
        Z = self.Z[idx]
        if len(Z.shape) > 3:
            return batchMatZonotope(Z)
        else:
            return matZonotope(Z)
    @property 
    def batch_shape(self):
        return self.Z.shape[:self.batch_dim]
    @property
    def dtype(self):
        return self.Z.dtype
    @property
    def device(self):
        return self.Z.device   
    @property
    def center(self):
        return self.Z[self.batch_idx_all+(0,)]
    @center.setter
    def center(self,value):
        self.Z[self.batch_idx_all+(0,)] = value
    @property
    def generators(self):
        return self.Z[self.batch_idx_all+(slice(1,None),)]
    @generators.setter
    def generators(self,value):
        self.Z[self.batch_idx_all+(slice(1,None),)] = value
    @property 
    def shape(self):
        return self.Z.shape[-2:]
    @property
    def n_rows(self):
        return self.Z.shape[-2]
    @property
    def n_cols(self):
        return self.Z.shape[-1]
    @property
    def n_generators(self):
        return self.Z.shape[-3]-1
    @property
    def T(self):
        return batchMatZonotope(self.Z.transpose(-2,-1))

    def to(self,dtype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device)
        return batchMatZonotope(Z)
    def cpu(self):
        Z = self.Z.cpu()
        return batchMatZonotope(Z)
        
    def __matmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matZonotope
        self: <matZonotope>
        other: <torch.tensor> OR <zonotope>
        return <zonotope>
        
        other: <matZonotope>
        return <matZonotope>
        '''
        if isinstance(other, torch.Tensor):
            z = self.Z@other
            return batchZonotope(z) 
        elif isinstance(other,batchZonotope):
            assert self.n_cols == other.dimension
            z = self.Z.unsqueeze(-3)@other.Z.unsqueeze(-1).unsqueeze(-4)
            return batchZonotope(z.reshape(z.shape[:-4]+(-1,self.n_rows)))
        elif isinstance(other,batchMatZonotope):
            assert self.n_cols == other.n_rows
            Z = self.Z.unsqueeze(-3)@other.Z.unsqueeze(-4)
            return batchMatZonotope(Z.reshape(Z.shape[:-4]+(-1,self.n_rows,other.n_cols)))
        else:
            assert False, 'Invalid object for matrix multiplication with matrix zonotope.'

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matZonotope
        self: <matZonotope>
        other: <torch.tensor> OR <zonotope>
        return <zonotope>
        
        other: <matZonotope>
        return <matZonotope>
        '''
        if isinstance(other, torch.Tensor):
            assert other.shape[-1] == self.n_rows
            assert len(other.shape) > 2
            Z = other @ self.Z
            return batchMatZonotope(Z)
        else:
            assert False, 'Invalid object for reversed matrix multiplication with matrix zonotope.'


    def deleteZerosGenerators(self,sorted=False,sort=False):
        '''
        delete zero vector generators
        self: <zonotope>

        return <zonotope>
        '''
        if sorted:
            non_zero_idxs = torch.sum(torch.sum(self.generators==0,(-2,-1))==0,tuple(range(self.batch_dim))) != 0
            g_red = self.generators[self.batch_idx_all+(non_zero_idxs,)]
        else: 
            zero_idxs = torch.sum(self.generators!=0,(-2,-1))==0     
            ind = zero_idxs.to(dtype=torch.float).sort(-1)[1].reshape(self.batch_shape+(self.n_generators,1,1)).repeat((1,)*(self.batch_dim+1)+self.shape)
            max_non_zero_len = (~zero_idxs).sum(-1).max()
            g_red = self.generators.gather(-3,ind)[self.batch_idx_all+(slice(None,max_non_zero_len),)]
        Z = torch.cat((self.center.unsqueeze(self.batch_dim),g_red),self.batch_dim)
        return batchMatZonotope(Z)

    def reduce(self,order,option='girard'):
        if option == 'girard':
            Z = self.deleteZerosGenerators()
            if order == 1:
                center, G = Z.center, Z.generators
                d = torch.sum(abs(G),-3).reshape(self.batch_shape+(-1,))
                Gbox = torch.diag_embed(d).reshape(self.batch_shape+(-1,3,3))
                ZRed= torch.cat((center.unsqueeze(self.batch_dim),Gbox),-3)          
            else:
                center, Gunred, Gred = pickedBatchGenerators(Z,order)
                d = torch.sum(abs(Gred),-3).reshape(self.batch_shape+(-1,))
                Gbox = torch.diag_embed(d).reshape(self.batch_shape+(-1,3,3))
                ZRed= torch.cat((center.unsqueeze(self.batch_dim),Gunred,Gbox),-3)
            #import pdb;pdb.set_trace()
            return batchMatZonotope(ZRed)
        else:
            assert False, 'Invalid reduction option'