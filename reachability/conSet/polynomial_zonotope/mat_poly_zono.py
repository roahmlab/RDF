"""
Define class for matrix polynomial zonotope
Author: Yongseok Kwon
Reference: Patrick Holme's implementation
"""

import sys
sys.path.append('../..')
from reachability.conSet.polynomial_zonotope.utils import removeRedundantExponents, mergeExpMatrix
from reachability import conSet 
import torch

class matPolyZonotope():
    '''
    <matPolyZonotope>

    c: <torch.Tensor> center maxtrix of the matrix polyonmial zonotope
    , shape: [nx,ny] 
    G: <torch.tensor> generator tensor containing the dependent generators 
    , shape: [N, nx, ny] 
    Grest: <torch.Tensor> generator tensor containing the independent generators
    , shape: [M, nx, ny]
    expMat: <troch.Tensor> matrix containing the exponents for the dependent generators
    , shape: [N, p]
    id: <torch.Tensor> vector containing the integer identifiers for the dependent factors
    , shape: [p]
    compress: <int> level for compressing dependent generators with expodent
    0: no compress, 1: compress zero dependent generators, 2: compress zero dependent generators and remove redundant expodent

    Eq. (coeff. a1,a2,...,aN; b1,b2,...,bp \in [0,1])
    G = [[Gd1],[Gd2],...,[GdN]]
    Grest = [[Gi1],[Gi2],...,[GiM]]
    (Gd1,Gd2,...,GdN,Gi1,Gi2,...,GiM \in R^(nx,ny), matrix)
    expMat = [[i11,i12,...,i1p],[i21,i22,...,i2p],...,[iN1,iN2,...,iNp]]
    id = [0,1,2,...,p-1]

    pZ = c + a1*Gi1 + a2*Gi2 + ... + aN*GiN + b1^i11*b2^i21*...*bp^ip1*Gd1 + b1^i12*b2^i22*...*bp^ip2*Gd2 + ... 
    + b1^i1M*b2^i2M*...*bp^ipM*GdM
    '''
    def __init__(self,Z,n_dep_gens=0,expMat=None,id=None,prop='None',compress=2):
        if isinstance(Z,list):
            Z = torch.tensor(Z,dtype=torch.float)
        assert isinstance(prop,str), 'Property should be string.'
        assert isinstance(Z, torch.Tensor), 'The input matrix should be either torch tensor or list.'
        C = Z[0]
        G = Z[1:1+n_dep_gens]
        Grest = Z[1+n_dep_gens:]
        if expMat == None and id == None:
            nonzero_g = torch.sum(G!=0,(-1,-2))!=0 # non-zero generator index
            G = G[nonzero_g]
            self.expMat = torch.eye(G.shape[0],dtype=torch.long,device=Z.device) # if G is EMPTY_TENSOR, it will be EMPTY_TENSOR, size = (0,0)            
            self.id = conSet.PROPERTY_ID.update(self.expMat.shape[1],prop).to(device=Z.device) # if G is EMPTY_TENSOR, if will be EMPTY_TENSOR
        elif expMat != None:
            #check correctness of user input 
            if isinstance(expMat, list):
                expMat = torch.tensor(expMat,dtype=torch.long,device=Z.device)
            assert isinstance(expMat,torch.Tensor), 'The exponent matrix should be either torch tensor or list.'
            assert expMat.dtype in (torch.int, torch.long,torch.short), 'Exponent should have integer elements.'
            assert torch.all(expMat >= 0) and expMat.shape[0] == n_dep_gens, 'Invalid exponent matrix.' 
            if compress == 2: 
                self.expMat,G = removeRedundantExponents(expMat,G)
            elif compress == 1:
                nonzero_g = torch.sum(G!=0,(-1,-2))!=0 # non-zero generator index
                G = G[nonzero_g]    
                self.expMat =expMat[nonzero_g]
            else:
                self.expMat =expMat
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
        self.Z = torch.vstack((C.unsqueeze(0),G,Grest))
        self.n_dep_gens = G.shape[0]
    @property
    def dtype(self):
        return self.Z.dtype
    @property
    def itype(self):
        return self.expMat.dtype
    @property
    def device(self):
        return self.Z.device
    @property 
    def C(self):
        return self.Z[0]
    @property 
    def G(self):
        return self.Z[1:self.n_dep_gens+1]
    @property 
    def Grest(self):
        return self.Z[self.n_dep_gens+1:]
    @property
    def n_generators(self):
        return len(self.Z)-1
    @property
    def n_indep_gens(self):
        return len(self.Z)-self.n_dep_gens-1
    @property 
    def n_rows(self):
        return self.Z.shape[1]
    @property 
    def n_cols(self):
        return self.Z.shape[2]
    @property
    def T(self):        
        return matPolyZonotope(self.Z.transpose(1,2),self.n_dep_gens,self.expMat,self.id,compress=0)
    @property 
    def input_pairs(self):
        id_sorted, order = torch.sort(self.id)
        expMat_sorted = self.expMat[:,order] 
        return self.Z, self.n_dep_gens, expMat_sorted, id_sorted
        
    def to(self,dtype=None,itype=None,device=None):
        Z = self.Z.to(dtype=dtype,device=device)
        expMat = self.expMat.to(dtype=itype,device=device)
        id = self.id.to(device=device)
        return matPolyZonotope(Z,self.n_dep_gens,expMat,id,compress=0)
    def cpu(self):
        Z = self.Z.cpu()
        expMat = self.expMat.cpu()
        id = self.id.cpu()
        return matPolyZonotope(Z,self.n_dep_gens,expMat,id,compress=0)
        
    def __matmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matPolyZonotope
        self: <matPolyZonotope>
        other: <torch.tensor> OR <polyZonotope>
        return <polyZonotope>
        
        other: <matPolyZonotope>
        return <matPolyZonotope>
        '''
        if isinstance(other, torch.Tensor):
            assert other.shape[0] == self.n_cols or other.shape[-2] == self.n_cols
            
            if len(other.shape) == 1:
                Z = self.Z @ other
                return conSet.polyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=1)
            elif len(other.shape) == 2:
                Z = self.Z @ other
                return matPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=1)
            else:
                Z = self.Z @ other.unsqueeze(-3)
                return conSet.batchMatPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=1)

        # NOTE: this is 'OVERAPPROXIMATED' multiplication for keeping 'fully-k-sliceables'
        # The actual multiplication should take
        # dep. gnes.: C_G, G_c, G_G, Grest_Grest, G_Grest, Grest_G
        # indep. gens.: C_Grest, Grest_c
        #
        # But, the sliceable multiplication takes
        # dep. gnes.: C_G, G_c, G_G (fully-k-sliceable)
        # indep. gnes.: C_Grest, Grest_c, Grest_Grest
        #               G_Grest, Grest_G (partially-k-sliceable)
        
        elif isinstance(other,conSet.polyZonotope):
           
            assert self.n_cols == other.dimension
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            _Z = (self.Z.unsqueeze(1) @ other.Z.unsqueeze(-1)).squeeze(-1)
            z1 = _Z[:self.n_dep_gens+1,0]
            z2 = _Z[:self.n_dep_gens+1,1:other.n_dep_gens+1].reshape(-1,self.n_rows)
            z3 = _Z[self.n_dep_gens+1:].reshape(-1,self.n_rows)
            z4 = _Z[:self.n_dep_gens+1,other.n_dep_gens+1:].reshape(-1,self.n_rows)
            Z = torch.vstack((z1,z2,z3,z4))
            expMat = torch.vstack((expMat1,expMat2,expMat2.repeat(self.n_dep_gens,1)+expMat1.repeat_interleave(other.n_dep_gens,dim=0)))
            n_dep_gens = (self.n_dep_gens+1) * (other.n_dep_gens+1)-1 
            return conSet.polyZonotope(Z,n_dep_gens,expMat,id)

        elif isinstance(other,matPolyZonotope):
            assert self.n_cols == other.n_rows
            id, expMat1, expMat2 = mergeExpMatrix(self.id,other.id,self.expMat,other.expMat)
            _Z = (self.Z.unsqueeze(1) @ other.Z)
            z1 = _Z[:self.n_dep_gens+1,0]
            z2 = _Z[:self.n_dep_gens+1,1:other.n_dep_gens+1].reshape(-1,self.n_rows,other.n_cols)
            z3 = _Z[self.n_dep_gens+1:].reshape(-1,self.n_rows,other.n_cols)
            z4 = _Z[:self.n_dep_gens+1,other.n_dep_gens+1:].reshape(-1,self.n_rows,other.n_cols)
            Z = torch.vstack((z1,z2,z3,z4))
            expMat = torch.vstack((expMat1,expMat2,expMat2.repeat(self.n_dep_gens,1)+expMat1.repeat_interleave(other.n_dep_gens,dim=0)))
            n_dep_gens = (self.n_dep_gens+1) * (other.n_dep_gens+1)-1 
            return matPolyZonotope(Z,n_dep_gens,expMat,id)
        elif isinstance(other,conSet.batchMatPolyZonotope):
            return other.__rmatmul__(self)
        else:
            raise ValueError('the other object should be torch tensor or polynomial zonotope.')

    def __rmatmul__(self,other):
        '''
        Overloaded '@' operator for the multiplication of a __ with a matPolyZonotope
        self: <matPolyZonotope>
        other: <torch.tensor> OR <polyZonotope>
        return <polyZonotope>
        
        other: <matPolyZonotope>
        return <matPolyZonotope>
        '''
        if isinstance(other,torch.Tensor):
            assert len(other.shape) == 2, 'The other object should be 2-D tensor.'  
            assert other.shape[1] == self.n_rows
            Z = other @ self.Z
            return matPolyZonotope(Z,self.n_dep_gens,self.expMat,self.id,compress=1)

    def to_matZonotope(self):
        if self.n_dep_gens != 0:
            ind = torch.any(self.expMat%2,1)
            Gquad = self.G[~ind]
            c = self.C + 0.5*torch.sum(Gquad,0)
            Z = torch.vstack((c.unsqueeze(0), self.G[ind],0.5*Gquad,self.Grest))
        else:
            Z = self.Z
        return conSet.matZonotope(Z)


    def reduce(self,order,option='girard'):
        # extract dimensions
        N = self.n_rows * self.n_cols
        P = self.n_dep_gens 
        Q = self.n_indep_gens
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if P+Q > N*order and K >=0:
            G = self.Z[1:]
            # half the generators length for exponents that are all even
            temp = torch.any(self.expMat%2,1)
            ind = (~temp).nonzero().squeeze()
            G[ind] *= 0.5
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,(1,2))
            # determine the smallest gens to remove            
            ind = torch.argsort(len,descending=True)
            ind_red,ind_rem = ind[:K], ind[K:]
            # split the indices into the ones for dependent and independent
            indDep = ind_rem[ind_rem < P]
            ind_REM = torch.hstack((indDep, ind_rem[ind_rem >= P]))
            indDep_red = ind_red[ind_red < P]
            ind_RED = torch.hstack((indDep_red,ind_red[ind_red >= P]))
            # construct a zonotope from the gens that are removed
            n_dg_rem = indDep.shape[0]
            Erem = self.expMat[indDep]
            Ztemp = torch.vstack((torch.zeros(1,self.n_rows,self.n_cols,dtype=self.dtype,device=self.device),G[ind_REM]))
            pZtemp = matPolyZonotope(Ztemp,n_dg_rem,Erem,self.id,compress=1) # NOTE: ID???
            zono = pZtemp.to_matZonotope() # zonotope over-approximation
            # reduce the constructed zonotope with the reducetion techniques for linear zonotopes
            zonoRed = zono.reduce(1,option)
            
            # remove the gens that got reduce from the gen matrices
            expMatRed = self.expMat[indDep_red]    
            n_dg_red = indDep_red.shape[0]
            # add the reduced gens as new indep gens
            ZRed = torch.vstack(((self.C + zonoRed.center).unsqueeze(0),G[ind_RED],zonoRed.generators))
        else:
            ZRed = self.Z
            n_dg_red = self.n_dep_gens
            expMatRed = self.expMat
        # remove all exponent vector dimensions that have no entries
        ind = torch.sum(expMatRed,0)>0
        #ind = temp.nonzero().reshape(-1)
        expMatRed = expMatRed[:,ind]
        idRed = self.id[ind]
        if self.n_rows == 1 and self.n_cols == 1:
            ZRed = torch.vstack((ZRed[:1],ZRed[1:n_dg_red+1].sum(0).unsqueeze(0),ZRed[n_dg_red+1:]))
        return matPolyZonotope(ZRed,n_dg_red,expMatRed,idRed,compress=1)

    def reduce_indep(self,order,option='girard'):
        # extract dimensions
        N = self.n_rows * self.n_cols
        P = self.n_dep_gens 
        Q = self.n_indep_gens
            
        # number of gens kept (N gens will be added back after reudction)
        K = int(N*order-N)
        # check if the order need to be reduced
        if Q > N*order and K >=0:
            G = self.Grest
            # caculate the length of the gens with a special metric
            len = torch.sum(G**2,(1,2))
            # determine the smallest gens to remove            
            ind = torch.argsort(len,descending=True)
            ind_red,ind_rem = ind[:K], ind[K:]
            # reduce the generators with the reducetion techniques for linear zonotopes            
            d = torch.sum(abs(G[ind_rem]),0).reshape(-1)
            Gbox = torch.diag(d).reshape(-1,self.n_rows,self.n_cols)
            # add the reduced gens as new indep gens
            ZRed = torch.vstack((self.C .unsqueeze(0),self.G,G[ind_red],Gbox))
        else:
            ZRed = self.Z
        n_dg_red = self.n_dep_gens
        if self.n_rows == 1 == self.n_cols and n_dg_red != 1:
            ZRed = torch.vstack((ZRed[:1],ZRed[1:n_dg_red+1].sum(0).unsqueeze(0),ZRed[n_dg_red+1:]))
            n_dg_red = 1
        return matPolyZonotope(ZRed,n_dg_red,self.expMat,self.id,compress=1)

