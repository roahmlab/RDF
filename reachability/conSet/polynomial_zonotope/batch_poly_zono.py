"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: CORA
"""

import sys
sys.path.append('../..')
from reachability.conSet.polynomial_zonotope.utils import removeRedundantExponentsBatch, mergeExpMatrix
from reachability.conSet.polynomial_zonotope.poly_zono import polyZonotope
from reachability import conSet 
import torch

class batchPolyZonotope:
    '''
    pZ: <polyZonotope>
    
    c: <torch.Tensor> center of the polyonmial zonotope
    , shape: [B1, B2, .. , Bb, nx]
    G: <torch.Tensor> generator matrix containing the dependent generators
    , shape: [B1, B2, .. , Bb, N, nx]
    Grest: <torch.Tensor> generator matrix containing the independent generators
    , shape: [B1, B2, .. , Bb, M, nx]
    expMat: <troch.Tensor> matrix containing the exponents for the dependent generators
    , shape: [N, p]
    id: <torch.Tensor> vector containing the integer identifiers for the dependent factors
    , shape: [p]
    compress: <int> level for compressing dependent generators with expodent
    0: no compress, 1: compress zero dependent generators, 2: compress zero dependent generators and remove redundant expodent

    Eq. (coeff. a1,a2,...,aN; b1,b2,...,bp \in [0,1])
    G = [gd1,gd2,...,gdN]
    Grest = [gi1,gi2,...,giM]
    expMat = [[i11,i12,...,i1N],[i21,i22,...,i2N],...,[ip1,ip2,...,ipN]]
    id = [0,1,2,...,p-1]

    pZ = c + a1*gi1 + a2*gi2 + ... + aN*giN + b1^i11*b2^i21*...*bp^ip1*gd1 + b1^i12*b2^i22*...*bp^ip2*gd2 + ... 
    + b1^i1M*b2^i2M*...*bp^ipM*gdM
     
    '''
    # NOTE: property for mat pz
    def __init__(self,Z,n_dep_gens=0,expMat=None,id=None,prop='None',compress=2):
        assert isinstance(prop,str), 'Property should be string.'
        assert isinstance(Z, torch.Tensor), 'The input matrix should be either torch tensor or list.'
        assert len(Z.shape) > 2, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'
        self.batch_dim = len(Z.shape) - 2
        self.batch_idx_all = tuple([slice(None) for _ in range(self.batch_dim)])        

        c = Z[self.batch_idx_all+(0,)]
        G = Z[self.batch_idx_all+(slice(1,1+n_dep_gens),)]
        Grest = Z[self.batch_idx_all+(slice(1+n_dep_gens,None),)]
        if expMat == None and id == None:
            nonzero_g = torch.sum(G!=0,tuple(range(self.batch_dim))+(-1,))!=0 # non-zero generator index
            G = G[self.batch_idx_all+(nonzero_g,)]
            self.expMat = torch.eye(G.shape[self.batch_dim],dtype=torch.long,device=Z.device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)
            self.id = conSet.PROPERTY_ID.update(self.expMat.shape[1],prop).to(device=Z.device) # if G is EMPTY_TENSOR, if will be EMPTY_TENSOR
        elif expMat != None:
            #check correctness of user input 
            if isinstance(expMat, list):
                expMat = torch.tensor(expMat,dtype=torch.long,device=Z.device)
            assert isinstance(expMat,torch.Tensor), 'The exponent matrix should be either torch tensor or list.'
            assert expMat.dtype in (torch.int, torch.long,torch.short), 'Exponent should have integer elements.'
            assert torch.all(expMat >= 0) and expMat.shape[0] == n_dep_gens, 'Invalid exponent matrix.' 
            if compress == 2: 
                self.expMat,G = removeRedundantExponentsBatch(expMat,G,self.batch_idx_all)
            elif compress == 1:
                nonzero_g = torch.sum(G!=0,tuple(range(self.batch_dim))+(-1,))!=0 # non-zero generator index
                G = G[self.batch_idx_all+(nonzero_g,)]
                self.expMat = expMat[nonzero_g]
            else:
                self.expMat =expMat            
            

            #self.expMat =expMat
            if id != None:
                if isinstance(id, list):
                    id = torch.tensor(id,dtype=torch.long,device=Z.device)
                if id.numel() !=0:
                    assert prop == 'None', 'Either ID or property should not be defined.'
                    assert max(id) < conSet.PROPERTY_ID.offset, 'Non existing ID is defined'
                assert isinstance(id, torch.Tensor), 'The identifier vector should be either torch tensor or list.'
                assert id.shape[0] == expMat.shape[1], f'Invalid vector of identifiers. The number of exponents is {expMat.shape[1]}, but the number of identifiers is {id.shape[0]}.'
                self.id = id
            else:
                self.id = conSet.PROPERTY_ID.update(self.expMat.shape[1],prop).to(device=Z.device) 
        else:
            assert False, 'Identifiers can only be defined as long as the exponent matrix is defined.'

        self.Z = torch.cat((c.unsqueeze(-2),G,Grest),dim=-2)
        self.n_dep_gens = G.shape[-2]

    def __getitem__(self,idx):
        Z = self.Z[idx]
        if len(Z.shape) > 2:
            return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=0)
        else:
            return polyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=0)
    @property 
    def batch_shape(self):
        return self.Z.shape[:-2]
    @property
    def itype(self):
        return self.expMat.dtype
    @property 
    def dtype(self):
        return self.Z.dtype 
    @property
    def device(self):
        return self.Z.device
    @property 
    def c(self):
        return self.Z[self.batch_idx_all+(0,)]
    @property 
    def G(self):
        return self.Z[self.batch_idx_all+(slice(1,self.n_dep_gens+1),)]
    @property 
    def Grest(self):
        return self.Z[self.batch_idx_all+(slice(self.n_dep_gens+1,None),)]
    @property
    def n_generators(self):
        return self.Z.shape[-2]-1
    @property
    def n_indep_gens(self):
        return self.Z.shape[-2]-1-self.n_dep_gens
    @property 
    def dimension(self):
        return self.Z.shape[-1]
    @property 
    def shape(self):
        return self.Z.shape[-1:]
    @property 
    def input_pairs(self):
        id_sorted, order = torch.sort(self.id)
        expMat_sorted = self.expMat[:,order] 
        return self.Z, self.n_dep_gens, expMat_sorted, id_sorted

    def to(self,dtype=None,itype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device)
        expMat = self.expMat.to(dtype=itype,device=device)
        id = self.id.to(device=device)
        return batchPolyZonotope(Z,self.n_dep_gens,expMat,id,compress=0)

    def cpu(self):
        Z = self.Z.cpu()
        expMat = self.expMat.cpu()
        id = self.id.cpu()
        return batchPolyZonotope(Z,self.n_dep_gens,expMat,id,compress=0)

    def  __add__(self,other):
        '''
        Overloaded '+' operator for Minkowski sum
        self: <polyZonotope>
        other: <torch.tensor> OR <zonotope> OR <polyZonotope>
        return <polyZonotope>
        '''
        # if other is a vector
        if  isinstance(other,(torch.Tensor,float,int)):
            assert isinstance(other,(float,int)) or other.shape == self.shape, f'array dimension does not match: should be {self.shape}, not {other.shape}.'
            Z = torch.clone(self.Z)
            Z[self.batch_idx_all+(0,)] += other
            n_dep_gens, expMat, id = self.n_dep_gens, self.expMat, self.id
            return batchPolyZonotope(Z,n_dep_gens,expMat,id,compress=0)

        # if other is a polynomial zonotope
        elif isinstance(other,polyZonotope): # exact Plus
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            Z = torch.cat(((self.c+other.c).unsqueeze(-2),self.G,other.G.repeat(self.batch_shape+(1,1,)),self.Grest,other.Grest.repeat(self.batch_shape+(1,1,))),-2)
            expMat = torch.vstack((expMat1,expMat2))
            n_dep_gens = self.n_dep_gens + other.n_dep_gens
        elif isinstance(other,batchPolyZonotope): # exact Plus
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            Z = torch.cat(((self.c+other.c).unsqueeze(-2),self.G,other.G,self.Grest,other.Grest),-2) 
            expMat = torch.vstack((expMat1,expMat2))
            n_dep_gens = self.n_dep_gens + other.n_dep_gens
        elif isinstance(other,conSet.zonotope):
            n_dep_gens, expMat, id = self.n_dep_gens, self.expMat, self.id 
            Z = torch.cat(((self.c+other.center).unsqueeze(-2),self.G,self.Grest,other.generators.repeat(self.batch_shape+(1,1,))),-2)
        elif isinstance(other,conSet.batchZonotope):
            n_dep_gens, expMat, id = self.n_dep_gens, self.expMat, self.id 
            Z = torch.cat(((self.c+other.center).unsqueeze(-2),self.G,self.Grest,other.generators),-2)

        return batchPolyZonotope(Z,n_dep_gens,expMat,id)
    __radd__ = __add__
    def __sub__(self,other):
        return self.__add__(-other)
    def __rsub__(self,other):
        return -self.__sub__(other)
    def __pos__(self):
        return self
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation
        self: <polyZonotope>
        return <polyZonotope>
        '''
        return batchPolyZonotope(torch.cat((-self.Z[:1+self.n_dep_gens],self.Grest)),self.n_dep_gens,self.expMat, self.id,compress=0)

    def __iadd__(self,other): 
        return self+other
    def __isub__(self,other):
        return self-other
    def __mul__(self,other):
        if isinstance(other,(torch.Tensor,int,float)):
            #assert len(other.shape) == 1
            assert isinstance(other,(int,float)) or len(other.shape) == 0 or self.dimension == other.shape[0] or self.dimension == 1 or len(other.shape) == 1, 'Invalid dimension.'
            Z = self.Z*other
            n_dep_gens = self.n_dep_gens
            expMat = self.expMat
            id = self.id
            return batchPolyZonotope(Z,n_dep_gens,expMat,id,compress=0)

        elif isinstance(other,(polyZonotope,batchPolyZonotope)):
            assert self.dimension == other.dimension, 'Both polynomial zonotope must have same dimension.'
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            
            _Z = self.Z.unsqueeze(-2)*other.Z.unsqueeze(-3)
            Z_shape = _Z.shape[:-2]+(-1,self.dimension)
            z1 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),0)]
            z2 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),slice(1,other.n_dep_gens+1))].reshape(Z_shape)
            z3 = _Z[self.batch_idx_all +(slice(self.n_dep_gens+1,None),)].reshape(Z_shape)
            z4 = _Z[self.batch_idx_all +(slice(None,self.n_dep_gens+1),slice(other.n_dep_gens+1,None))].reshape(Z_shape)            
            Z = torch.cat((z1,z2,z3,z4),dim=-2)
            expMat = torch.vstack((expMat1,expMat2,expMat2.repeat(self.n_dep_gens,1)+expMat1.repeat_interleave(other.n_dep_gens,dim=0)))
            n_dep_gens = (self.n_dep_gens+1) * (other.n_dep_gens+1)-1 
            return batchPolyZonotope(Z,n_dep_gens,expMat,id)


    __rmul__ = __mul__

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a matrix or an interval matrix with a polyZonotope
        self: <polyZonotope>
        other: <torch.tensor> OR <intervals>
        return <polyZonotope>
        '''
        
        # if other is a matrix
        if isinstance(other, torch.Tensor):            
            Z = self.Z@other.transpose(-2,-1)
        return polyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=1)

    def reduce_indep(self,order,option='girard'):
        # extract dimensions
        N = self.dimension
        Q = self.n_indep_gens
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if Q > N*order and K >=0:
            G = self.Grest
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,-1) # NOTE -1
            # determine the smallest gens to remove            
            ind = torch.argsort(len,dim=-1,descending=True).unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape)
            ind_rem, ind_red = ind[self.batch_idx_all+(slice(K),)], ind[self.batch_idx_all+(slice(K,None),)]
            # reduce the generators with the reducetion techniques for linear zonotopes
            d = torch.sum(abs(G.gather(-2,ind_red)),-2)
            Gbox = torch.diag_embed(d)
            # add the reduced gens as new indep gens
            ZRed = torch.cat((self.c.unsqueeze(-2),self.G,G.gather(-2,ind_rem),Gbox),dim=-2)
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.dimension == 1 and n_dg_red != 1:            
            ZRed = torch.cat((ZRed[self.batch_idx_all+(0,)],ZRed[self.batch_idx_all+(slice(1,n_dg_red+1),)].sum(-2).unsqueeze(-2),ZRed[self.batch_idx_all+(slice(n_dg_red+1,None),)]),dim=-2)
            n_dg_red = 1
        return batchPolyZonotope(ZRed,n_dg_red,self.expMat,self.id,compress=0)


    def exactCartProd(self,other):
        '''
        self: <polyZonotope>
        other: <polyZonotope>
        return <polyZonotope>
        '''    
        if isinstance(other,polyZonotope):
            c = torch.hstack((self.c,other.c))
            id,expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            g1 = torch.cat((self.G,torch.zeros(self.batch_shape+(self.n_indep_gens,other.dimension),dtype=self.dtype,device=self.device)),dim=-1)
            g2 = torch.cat((torch.zeros(self.batch_shape+(other.n_indep_gens,self.dimension),dtype=self.dtype,device=self.device),other.G.repeat(self.batch_shape+(1,1))),dim=-1)
            G = torch.cat((g1,g2),dim=-2)
            expMat = torch.vstack((expMat1,expMat2))
            g1 = torch.cat((self.Grest,torch.zeros(self.batch_shape+(self.n_dep_gens,other.dimension),dtype=self.dtype,device=self.device)),dim=-1)
            g2 = torch.cat((torch.zeros(self.batch_shape+(other.n_dep_gens,self.dimension),dtype=self.dtype,device=self.device),other.Grest.repeat(self.batch_shape+(1,1))),dim=-1)
            Grest = torch.cat((g1,g2),dim=-2)
        if isinstance(other,batchPolyZonotope):
            c = torch.hstack((self.c,other.c))
            id,expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            g1 = torch.cat((self.G,torch.zeros(self.batch_shape+(self.n_indep_gens,other.dimension),dtype=self.dtype,device=self.device)),dim=-1)
            g2 = torch.cat((torch.zeros(self.batch_shape+(other.n_indep_gens,self.dimension),dtype=self.dtype,device=self.device),other.G),dim=-1)
            G = torch.cat((g1,g2),dim=-2)
            expMat = torch.vstack((expMat1,expMat2))
            g1 = torch.cat((self.Grest,torch.zeros(self.batch_shape+(self.n_dep_gens,other.dimension),dtype=self.dtype,device=self.device)),dim=-1)
            g2 = torch.cat((torch.zeros(self.batch_shape+(other.n_dep_gens,self.dimension),dtype=self.dtype,device=self.device),other.Grest),dim=-1)
            Grest = torch.cat((g1,g2),dim=-2)
        Z = torch.cat((c,G,Grest),dim=-2)
        n_dep_gens = self.n_dep_gens+other.n_dep_gens
        return batchPolyZonotope(Z,n_dep_gens,expMat,id)

    def to_batchZonotope(self):
        if self.n_dep_gens != 0:
            ind = torch.any(self.expMat%2,1)
            Gquad = self.G[self.batch_idx_all+(~ind,)]
            c = self.c + 0.5*torch.sum(Gquad,-2)
            Z = torch.cat((c.unsqueeze(-2), self.G[self.batch_idx_all+(ind,)],0.5*Gquad,self.Grest),-2)
        else: 
            Z = self.Z
        return conSet.batchZonotope(Z)

    def to_interval(self,method='interval'):
        if method == 'interval':
            return self.to_batchZonotope().to_interval()
        else:
            assert False, 'Not implemented'

    def slice_dep(self,id_slc,val_slc):
        '''
        Slice polynomial zonotpe in depdent generators
        id_slc: id to dlice
        val_slc: indeterminant to slice
        '''
        if isinstance(id_slc,(int,list)):
            if isinstance(id_slc,int):
                id_slc = [id_slc]
            id_slc = torch.tensor(id_slc,dtype=self.dtype,device=self.device)
        if isinstance(val_slc,(int,float,list)):
            if isinstance(val_slc,(int,float)):
                val_slc = [val_slc]
            val_slc = torch.tensor(val_slc,dtype=self.dtype,device=self.device)
        
        if any(abs(val_slc)>1):
            import pdb; pdb.set_trace()
        #assert all(val_slc<=1) and all(val_slc>=-1), 'Indereminant should be in [-1,1].'
        
        id_slc, val_slc = id_slc.reshape(-1,1), val_slc.reshape(1,-1)
        order = torch.argsort(id_slc.reshape(-1))
        id_slc, val_slc  = id_slc[order], val_slc[:,order]
        ind = torch.any(self.id==id_slc,dim=0) # corresponding id for self.id  
        ind2 = torch.any(self.id==id_slc,dim=1)# corresponding id for id_slc
        #assert ind.numel()==len(id_slc), 'Some specidied IDs do not exist!'
        if ind.shape[0] != 0:
            G = self.G*torch.prod(val_slc[:,ind2]**self.expMat[:,ind],dim=1)
            expMat = self.expMat[:,~ind]
            id = self.id[:,~ind]
        else:
            G = self.G
            expMat = self.expMat
            id = self.id

        #expMat, G = removeRedundantExponents(expMat,G)
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[ind],0)
            G = G[~ind]
            expMat = expMat[~ind]
        else:
            c = self.c
        '''
        id = self.id
        ind = torch.sum(expMat,0) == 0
        if torch.any(ind):
            expMat = expMat[:,~ind]
            id = id[:,~ind]
        '''
        
        if G.shape[0] == 0 and self.Grest.shape[0] == 0:
            return polyZonotope(c,0,expMat,id)
        else:
            return polyZonotope(torch.vstack((c,G,self.Grest)), G.shape[0],expMat,id)
    

    def center_slice_all_dep(self,val_slc):
        n_ids = self.id.shape[0] # n_ids
        val_slc = val_slc[self.batch_idx_all + (slice(n_ids),)].unsqueeze(-2) # b1, b2,..., 1, n_ids
        expMat = self.expMat[:,torch.argsort(self.id)] # n_dep_gens, n_ids
        #val_slc**expMat: b1, b2, ..., n_dep_gens, n_ids
        #torch.prod(val_slc**expMat,dim=-1).unsqueeze(-2): b1, b2, ..., 1, n_dep_gens
        #(torch.prod(val_slc**expMat,dim=-1).unsqueeze(-2)@self.G).squeeze(-2):b1, b2, ...,dim
        return self.c + (torch.prod(val_slc**expMat,dim=-1).unsqueeze(-2)@self.G).squeeze(-2) 

    def grad_center_slice_all_dep(self,val_slc):        
        n_ids = self.id.shape[0] 
        n_short = val_slc.shape[-1] - n_ids
        val_slc = val_slc[self.batch_idx_all + (slice(n_ids),)].reshape(self.batch_shape+(1,1,n_ids)) # b1, b2,..., 1, 1, n_ids
        expMat = self.expMat[:,torch.argsort(self.id)] # n_dep_gens, n_ids
        expMat_red = expMat.unsqueeze(0).repeat(n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) # a tensor of reduced order expMat for each column, n_ids,  n_dep_gens, n_ids
        #torch.prod(val_slc**expMat_red,dim=-1), b1, b2,..., n_ids,  n_dep_gens
        #(expMat.T*torch.prod(val_slc**expMat_red,dim=-1)), b1, b2,..., n_ids,  n_dep_gens
        #self.G, b1, b2,..., n_dep_gens, dim
        #(self.G.unsqueeze(-2)*(expMat.T*torch.prod(val_slc**expMat_red,dim=-1)).unsqueeze(-3)), 
        #res, b1, b2,..., 1, n_ids,  n_dep_gens
        grad = ((expMat.T*torch.prod(val_slc**expMat_red,dim=-1).nan_to_num())@self.G).transpose(-1,-2) # b1, b2,..., dim, n_ids
        grad = torch.cat((grad,torch.zeros(self.batch_shape+self.shape+(n_short,),dtype=self.dtype,device=self.device)),-1)
        return grad

    def hess_center_slice_all_dep(self,val_slc):
        n_ids= self.id.shape[0]
        n_vals = val_slc.shape[-1]
        val_slc = val_slc[self.batch_idx_all + (slice(n_ids),)].reshape(self.batch_shape+(1,1,1,n_ids)) # b1, b2,..., 1, 1, 1, n_ids
        expMat = self.expMat[:,torch.argsort(self.id)] # n_dep_gens, n_ids
        expMat_red = expMat.unsqueeze(0).repeat(n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) # a tensor of reduced order expMat for each column
        expMat_twice_red = expMat.reshape((1,1)+expMat.shape).repeat(n_ids,n_ids,1,1) - torch.eye(n_ids,dtype=int).unsqueeze(-2) - torch.eye(n_ids,dtype=int).reshape(n_ids,1,1,n_ids)
        expMat_first = expMat.T.unsqueeze(1).repeat(1,n_ids,1)
        hess = torch.zeros(self.batch_shape+(self.dimension,n_vals,n_vals),dtype=self.dtype,device=self.device)
        hess[self.batch_idx_all+(slice(None),slice(n_ids),slice(n_ids))] = ((expMat_first*expMat_red.transpose(-1,-2)*torch.prod(val_slc**expMat_twice_red,dim=-1).nan_to_num())@self.G.unsqueeze(-3)).transpose(-3,-1)
        return hess

    
    def slice_all_dep(self,val_slc):
        '''
        Slice polynomial zonotpe in depdent generators
        id_slc: id to slice
        val_slc: indeterminant to slice
        c: <torch.Tensor>, shape [b1,b2,...,nx]
        grad_c: <torch.Tensor>, d c/d val_slc
        shape [b1,b2,...,nx,n_ids]
        '''

        ##################################
 
        n_ids = self.id.shape[0] # n_ids
        val_slc = val_slc[self.batch_idx_all + (slice(n_ids),)].unsqueeze(-2) # b1, b2,..., 1, n_ids
        expMat = self.expMat[:,torch.argsort(self.id)] # n_dep_gens, n_ids
        #val_slc**expMat: b1, b2, ..., n_dep_gens, n_ids
        #torch.prod(val_slc**expMat,dim=-1).unsqueeze(-2): b1, b2, ..., 1, n_dep_gens
        #(torch.prod(val_slc**expMat,dim=-1).unsqueeze(-2)@self.G).squeeze(-2):b1, b2, ...,dim
        c = self.c + (torch.prod(val_slc**expMat,dim=-1).unsqueeze(-2)@self.G).squeeze(-2) 
        return conSet.batchZonotope(torch.cat((c.unsqueeze(-2),self.Grest),-2))


    def deleteZerosGenerators(self,eps=0):
        expMat, G = removeRedundantExponentsBatch(expMat,G,self.batch_idx_all)
        ind = torch.sum(expMat,1) == 0
        if torch.any(ind):
            c = self.c + torch.sum(G[ind],0)
            G = G[~ind]
            expMat = expMat[~ind]
        else:
            c = self.c
        
        id = self.id
        ind = torch.sum(expMat,0) == 0
        if torch.any(ind):
            expMat = expMat[:,~ind]
            id = id[:,~ind]
        return polyZonotope(torch.vstack((c,G,self.Grest)),G.shape[0],expMat,id,compress=0)

    def project(self,dim=[0,1]):
        Z = self.Z[self.batch_idx_all+(slice(None),dim)]
        return batchPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=1)