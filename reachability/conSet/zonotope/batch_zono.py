"""
Define class for batch zonotope
Author: Yongseok Kwon
Reference: CORA
"""

import sys
sys.path.append('../..')
import torch
from reachability.conSet.polynomial_zonotope.batch_poly_zono import batchPolyZonotope 
from reachability.conSet.zonotope.zono import zonotope
from reachability.conSet.zonotope.utils import pickedBatchGenerators

class batchZonotope:
    '''
    b-zono: <batchZonotope>

    Z: <torch.Tensor> batch of center vector and generator matrix Z = [[C],[G]]
    , shape [B1, B2, .. , Bb, N+1, nx]
    center: <torch.Tensor> batch of center vector
    , shape [B1, B2, .. , Bb, nx] 
    generators: <torch.Tensor> batch of generator matrix
    , shape [B1, B2, .. , Bb, N, nx]
    
    Eq.
    G = [[g1],[g2],...,[gN]]
    b-zono = {c + a1*g1 + a2*g2 + ... + aN*gN | coeff. a1,a2,...,aN \in [-1,1] }
    '''
    def __init__(self,Z):
        if isinstance(Z,list):
            Z = torch.tensor(Z,dtype=torch.float)
        assert isinstance(Z,torch.Tensor), f'The input matrix should be either torch tensor or list, not {type(Z)}.'
        assert Z.dtype == torch.float or Z.dtype == torch.double, f'dtype should be either torch.float (torch.float32) or torch.double (torch.float64), but {Z.dtype}.'
        assert len(Z.shape) > 2, f'The dimension of Z input should be either 1 or 2, not {len(Z.shape)}.'
        self.Z = Z
        self.batch_dim = len(Z.shape) - 2
        self.batch_idx_all = tuple([slice(None) for _ in range(self.batch_dim)])

    def __getitem__(self,idx):
        Z = self.Z[idx]
        if len(Z.shape) > 2:
            return batchZonotope(Z)
        else:
            return zonotope(Z)
    @property 
    def batch_shape(self):
        return self.Z.shape[:self.batch_dim]
    @property
    def dtype(self):
        '''
        The data type of a batch zonotope properties
        return torch.float or torch.double        
        '''
        return self.Z.dtype
    @property
    def device(self):
        '''
        The device of a batch zonotope properties
        return 'cpu', 'cuda:0', or ...
        '''
        return self.Z.device
    @property
    def center(self):
        '''
        The center of a batch zonotope
        return <torch.Tensor>
        , shape [B1, B2, .. , Bb, nx] 
        '''
        return self.Z[self.batch_idx_all+(0,)]
    @center.setter
    def center(self,value):
        '''
        Set value of the center
        '''
        self.Z[self.batch_idx_all+(0,)] = value
    @property
    def generators(self):
        '''
        Generators of a batch zonotope
        return <torch.Tensor>
        , shape [B1, B2, .. , Bb, N, nx] 
        '''
        return self.Z[self.batch_idx_all+(slice(1,None),)]
    @generators.setter
    def generators(self,value):
        '''
        Set value of generators
        '''
        self.Z[self.batch_idx_all+(slice(1,None),)] = value
    @property 
    def shape(self):
        '''
        The shape of vector elements (ex. center) of a layer of a batch zonotope
        return <tuple>, (nx,)
        '''
        return (self.Z.shape[-1],)
    @property
    def dimension(self):
        '''
        The dimension of a batch zonotope
        return <int>, nx
        '''        
        return self.Z.shape[-1]
    @property
    def n_generators(self):
        '''
        The number of generators of a batch zonotope
        return <int>, N
        '''
        return self.Z.shape[-2]-1
    def to(self,dtype=None,device=None):
        '''
        Change the device and data type of a batch zonotope
        dtype: torch.float or torch.double
        device: 'cpu', 'gpu', 'cuda:0', ...
        '''
        Z = self.Z.to(dtype=dtype, device=device)
        return batchZonotope(Z)
    def cpu(self):
        '''
        Change the device of a batch zonotope to CPU
        '''
        Z = self.Z.cpu()
        return batchZonotope(Z)

    def __repr__(self):
        '''
        Representation of a batch zonotope as a text
        return <str>, 
        ex. batchZonotope([[[0., 0., 0.],[1., 0., 0.]],[[0., 0., 0.],[2., 0., 0.]]])
        '''
        return str(self.Z).replace('tensor','batchZonotope')
    def  __add__(self,other):
        '''
        Overloaded '+' operator for addition or Minkowski sum
        self: <batchZonotope>
        other: <torch.Tensor> OR <zonotope> or <batchZonotope>
        return <batchZonotope>
        '''   
        if isinstance(other, torch.Tensor):
            #assert other.shape == self.shape, f'array dimension does not match: should be {self.shape}, not {other.shape}.'
            Z = torch.clone(self.Z)
            Z[self.batch_idx_all+(0,)] += other
        elif isinstance(other, zonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.cat(((self.center + other.center).unsqueeze(-2),self.generators,other.generators.repeat(self.batch_shape+(1,1,))),-2)
        elif isinstance(other, batchZonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.cat(((self.center + other.center).unsqueeze(-2),self.generators,other.generators),-2)
        else:
            assert False, f'the other object is neither a zonotope nor a torch tensor, not {type(other)}.'
        return batchZonotope(Z)

    __radd__ = __add__ # '+' operator is commutative.

    def __sub__(self,other):
        '''
        Overloaded '-' operator for substraction or Minkowski difference
        self: <batchZonotope>
        other: <torch.Tensor> OR <zonotope> or <batchZonotope>
        return <batchZonotope>
        '''   
        if isinstance(other, torch.Tensor):
            Z = torch.clone(self.Z)
            #assert other.shape == self.shape, f'array dimension does not match: should be {self.shape}, not {other.shape}.'
            Z[self.batch_idx_all+(0,)] -= other
        elif isinstance(other, zonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.cat(((self.center - other.center).unsqueeze(self.batch_dim),self.generators,other.generators.repeat(self.batch_shape+(1,1,))),self.batch_dim)
        elif isinstance(other, batchZonotope): 
            assert self.dimension == other.dimension, f'zonotope dimension does not match: {self.dimension} and {other.dimension}.'
            Z = torch.cat(((self.center - other.center).unsqueeze(self.batch_dim),self.generators,other.generators),self.batch_dim)
        else:
            assert False, f'the other object is neither a zonotope nor a torch tensor, not {type(other)}.'
        return batchZonotope(Z)
    def __rsub__(self,other):
        '''
        Overloaded reverted '-' operator for substraction or Minkowski difference
        self: <batchZonotope>
        other: <torch.Tensor> OR <zonotope> or <batchZonotope>
        return <batchZonotope>
        '''   
        return -self.__sub__(other)
    def __iadd__(self,other):
        '''
        Overloaded '+=' operator for addition or Minkowski sum
        self: <batchZonotope>
        other: <torch.Tensor> OR <zonotope> or <batchZonotope>
        return <batchZonotope>
        '''   
        return self+other
    def __isub__(self,other):
        '''
        Overloaded '-=' operator for substraction or Minkowski difference
        self: <batchZonotope>
        other: <torch.Tensor> OR <zonotope> or <batchZonotope>
        return <batchZonotope>
        '''   
        return self-other

    def __pos__(self):
        '''
        Overloaded unary '+' operator for a batch zonotope ifself
        self: <zonotope>
        return <zonotope>
        '''   
        return self    
    
    def __neg__(self):
        '''
        Overloaded unary '-' operator for negation of a batch zonotope
        self: <batchZonotope>
        return <batchZonotope>
        '''   
        Z = -self.Z
        Z[self.batch_idx_all+(slice(1,None),)] = self.Z[self.batch_idx_all+(slice(1,None),)]
        return batchZonotope(Z)    
    
    def __rmatmul__(self,other):
        '''
        Overloaded reverted '@' operator for matrix multiplication on vector elements of a batch zonotope
        self: <batchZonotope>
        other: <torch.Tensor>
        return <batchZonotope>
        '''   
        assert isinstance(other, torch.Tensor), f'The other object should be torch tensor, but {type(other)}.'
        Z = self.Z@other.transpose(-2,-1)
        return batchZonotope(Z) 


    def __mul__(self,other):
        '''
        Overloaded reverted '*' operator for scaling a batch zonotope
        self: <batchZonotope>
        other: <int> or <float>
        return <batchZonotope>
        '''   
        if isinstance(other,(float,int)):
            Z = other*self.Z
        return batchZonotope(Z)

    __rmul__ = __mul__ # '*' operator is commutative.

    def slice(self,slice_dim,slice_pt):
        '''
        slice zonotope on specified point in a certain dimension
        self: <zonotope>
        slice_dim: <torch.Tensor> or <list> or <int>
        , shape  []
        slice_pt: <torch.Tensor> or <list> or <float> or <int>
        , shape  []
        return <zonotope>
        '''
        if isinstance(slice_dim, list):
            slice_dim = torch.tensor(slice_dim,dtype=torch.long,device=self.device)
        elif isinstance(slice_dim, int) or (isinstance(slice_dim, torch.Tensor) and len(slice_dim.shape)==0):
            slice_dim = torch.tensor([slice_dim],dtype=torch.long,device=self.device)

        if isinstance(slice_pt, list):
            slice_pt = torch.tensor(slice_pt,dtype=self.dtype,device=self.device)
        elif isinstance(slice_pt, int) or isinstance(slice_pt, float) or (isinstance(slice_pt, torch.Tensor) and len(slice_pt.shape)==0):
            slice_pt = torch.tensor([slice_pt],dtype=self.dtype,device=self.device)

        assert isinstance(slice_dim, torch.Tensor) and isinstance(slice_pt, torch.Tensor), 'Invalid type of input'
        assert len(slice_dim.shape) ==1, 'slicing dimension should be 1-dim component.'
        #assert slice_pt.shape[:-1] ==self.batch_shape, 'slicing point should be (batch+1)-dim component.'
        assert len(slice_dim) == slice_pt.shape[-1], f'The number of slicing dimension ({len(slice_dim)}) and the number of slicing point ({slice_pt.shape[-1]}) should be the same.'

        N = len(slice_dim)
        slice_dim, ind = torch.sort(slice_dim)
        slice_pt = slice_pt[(slice(None),)*(len(slice_pt.shape)-1)+(ind,)]

        c = self.center
        G = self.generators
        G_dim = G[self.batch_idx_all+(slice(None),slice_dim)]
        non_zero_idx = G_dim != 0
        assert torch.all(torch.sum(non_zero_idx,-2)==1), 'There should be one generator for each slice index.'
        slice_idx = non_zero_idx.transpose(-2,-1).nonzero()


        #slice_idx = torch.any(non_zero_idx,-1)        
        slice_c = c[self.batch_idx_all+(slice_dim,)]
        ind = tuple(slice_idx[:,:-2].T)
        slice_g = G_dim[ind+(slice_idx[:,-1],slice_idx[:,-2])].reshape(self.batch_shape+(N,))
        slice_lambda = (slice_pt-slice_c)/slice_g
        assert not (abs(slice_lambda)>1).any(), 'slice point is ouside bounds of reach set, and therefore is not verified'        
        Z = torch.cat((c.unsqueeze(-2) + slice_lambda.unsqueeze(-2)@G[ind+(slice_idx[:,-1],)].reshape(self.batch_shape+(N,self.dimension)),G[~non_zero_idx.any(-1)].reshape(self.batch_shape+(-1,self.dimension))),-2)
        return batchZonotope(Z)
    def project(self,dim=[0,1]):
        '''
        The projection of a batch zonotope onto the specified dimensions
        self: <batchZonotope>
        dim: <int> or <list> or <torch.Tensor> dimensions for prjection 
        
        return <batchZonotope>
        '''
        Z = self.Z[self.batch_idx_all+(slice(None),dim)]
        return batchZonotope(Z)

    def polygon(self,nan=True):
        '''
        NOTE: this is unstable for zero generators
        converts a 2-d zonotope into a polygon as vertices
        self: <zonotope>

        return <torch.Tensor>, <torch.float64>
        '''
        dim = 2
        z = self.deleteZerosGenerators()
        c = z.center[self.batch_idx_all+(slice(2),)].unsqueeze(-2)#.repeat((1,)*(self.batch_dim+2))
        G = torch.clone(z.generators[self.batch_idx_all+(slice(None),slice(2))])
        x_idx = self.batch_idx_all+(slice(None),0)
        y_idx = self.batch_idx_all+(slice(None),1)
        G_y = G[y_idx]
        x_max = torch.sum(abs(G[x_idx]),-1)
        y_max = torch.sum(abs(G_y),-1)
        
        G[G_y<0] = - G[G_y<0]
        if nan:
            G[torch.linalg.norm(G,dim=-1)==0] = torch.nan
        angles = torch.atan2(G[y_idx],G[x_idx])    
        ang_idx = torch.argsort(angles,dim=-1).unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+(2,))
        vertices_half = torch.cat((torch.zeros(self.batch_shape+(1,)+(2,),dtype=self.dtype,device=self.device),2*G.gather(-2,ang_idx).cumsum(axis=self.batch_dim)),-2)
        vertices_half[x_idx] += (x_max - torch.max(vertices_half[x_idx].nan_to_num(-torch.inf),dim=-1)[0]).unsqueeze(-1)
        vertices_half[y_idx] -= y_max.unsqueeze(-1)
        if nan:
            last_idx = (z.n_generators-angles.isnan().sum(-1)).reshape(self.batch_shape+(1,1)).repeat((1,)*self.batch_dim+(1,2))
            temp = (vertices_half[self.batch_idx_all+(0,)].unsqueeze(-2)+ vertices_half.gather(-2,last_idx))
        else:
            temp = (vertices_half[self.batch_idx_all+(0,)]+ vertices_half[self.batch_idx_all+(-1,)]).unsqueeze(-2)


        full_vertices = torch.cat((vertices_half,-vertices_half[self.batch_idx_all+(slice(1,None),)] + temp),dim=self.batch_dim) + c
        return full_vertices        

    def polytope(self,combs=None):
        '''
        converts a zonotope from a G- to a H- representation
        P
        comb
        isDeg
        NOTE: there is a possibility with having nan value on the output, so you might wanna use nan_to_num()
        OR, just use python built-in max function instead of torch.max or np.max.
        '''
        c = self.center
        G = torch.clone(self.generators)
        h = torch.linalg.vector_norm(G,dim=-1)
        h_sort, indicies = torch.sort(h,dim=-1,descending=True)


        h_nonzero = h_sort > 1e-6
        h_zero_all = ((h_nonzero).sum(tuple(range(self.batch_dim))) ==0)
        #G[~h_nonzero] = 0 # make sure everything less than 1e-6 to be actual zero, so that non-removable zero padding can be converged into nan value on the output value
        # NOTE: for some reason the above one didnt work out
        if torch.any(h_zero_all): 
            first_reduce_idx = torch.nonzero(h_zero_all).squeeze(-1)[0]
            G=G.gather(self.batch_dim,indicies.unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape))[self.batch_idx_all+(slice(None,first_reduce_idx),)]
        
        n_gens, dim = G.shape[-2:] 
        if dim == 1:
            C = G/torch.linalg.vector_norm(G,dim=-1).unsqueeze(-1)
        elif dim == 2:
            x_idx = self.batch_idx_all+(slice(None),slice(0,1))
            y_idx = self.batch_idx_all+(slice(None),slice(1,2))
            C = torch.cat((-G[y_idx],G[x_idx]),-1)
            C = C/torch.linalg.vector_norm(C,dim=-1).unsqueeze(-1)
        elif dim == 3:
            # not complete for example when n_gens < dim-1; n_gens =0 or n_gens =1 
            if combs is None or n_gens >= len(combs):
                comb = torch.combinations(torch.arange(n_gens),r=dim-1)
            else:
                comb = combs[n_gens]
            Q = torch.cat((G[self.batch_idx_all+(comb[:,0],)],G[self.batch_idx_all+(comb[:,1],)]),dim=-1)
            temp1 = (Q[self.batch_idx_all+(slice(None),1)]*Q[self.batch_idx_all+(slice(None),5)]-Q[self.batch_idx_all+(slice(None),2)]*Q[self.batch_idx_all+(slice(None),4)]).unsqueeze(-1)
            temp2 = (-Q[self.batch_idx_all+(slice(None),0)]*Q[self.batch_idx_all+(slice(None),5)]+Q[self.batch_idx_all+(slice(None),2)]*Q[self.batch_idx_all+(slice(None),3)]).unsqueeze(-1)
            temp3 = (Q[self.batch_idx_all+(slice(None),0)]*Q[self.batch_idx_all+(slice(None),4)]-Q[self.batch_idx_all+(slice(None),1)]*Q[self.batch_idx_all+(slice(None),3)]).unsqueeze(-1)
            C = torch.cat((temp1,temp2,temp3),dim=-1)
            C = C/torch.norm(C,dim=-1,keepdim=True)
        elif dim >=4 and dim<=7:
            assert False
        else:
            assert False
        
        deltaD = torch.sum(abs(C@self.generators.transpose(-2,-1)),dim=-1)
        
        d = (C@c.unsqueeze(-1)).squeeze(-1)
        PA = torch.cat((C,-C),dim=-2)
        Pb = torch.cat((d+deltaD,-d+deltaD),dim=-1)
        # NOTE: torch.nan_to_num()
        return PA, Pb

    def deleteZerosGenerators(self,sorted=False,sort=False):
        '''
        delete zero vector generators
        self: <zonotope>

        return <zonotope>
        '''
        if sorted:
            non_zero_idxs = torch.sum(torch.any(self.generators!=0,-1),tuple(range(self.batch_dim))) != 0
            g_red = self.generators[self.batch_idx_all+(non_zero_idxs,)]
        else:
            zero_idxs = torch.all(self.generators==0,axis=-1)
            ind = zero_idxs.to(dtype=torch.float).sort(-1)[1].unsqueeze(-1).repeat((1,)*(self.batch_dim+1)+self.shape)
            max_non_zero_len = (~zero_idxs).sum(-1).max()
            g_red = self.generators.gather(-2,ind)[self.batch_idx_all+(slice(None,max_non_zero_len),)]
        Z = torch.cat((self.center.unsqueeze(self.batch_dim),g_red),self.batch_dim)
        return batchZonotope(Z)

    def reduce(self,order,option='girard'):
        if option == 'girard':
            Z = self.deleteZerosGenerators()
            if order == 1:
                center, G = Z.center, Z.generators
                d = torch.sum(abs(G),-2)
                Gbox = torch.diag_embed(d)
                ZRed= torch.cat((center.unsqueeze(self.batch_dim),Gbox),-2)
            else:
                center, Gunred, Gred = pickedBatchGenerators(Z,order)
                d = torch.sum(abs(Gred),-2)
                Gbox = torch.diag_embed(d)
                ZRed= torch.cat((center.unsqueeze(self.batch_dim),Gunred,Gbox),-2)
            return batchZonotope(ZRed)
        else:
            assert False, 'Invalid reduction option'

    def to_polyZonotope(self,dim=None,prop='None'):
        '''
        convert zonotope to polynomial zonotope
        self: <zonotope>
        dim: <int>, dimension to take as sliceable
        return <polyZonotope>
        '''
        if dim is None:
            return batchPolyZonotope(self.Z,0)
        assert isinstance(dim,int) and dim <= self.dimension
        idx = self.generators[self.batch_idx_all+(slice(None),dim)] == 0
        assert ((~idx).sum(-1)==1).all(), 'sliceable generator should be one for the dimension.'
        Z = torch.cat((self.center.unsqueeze(-2),self.generators[~idx].reshape(self.batch_shape+(-1,self.dimension)),self.generators[idx].reshape(self.batch_shape+(-1,self.dimension))),-2)
        return batchPolyZonotope(Z,1,prop=prop)
