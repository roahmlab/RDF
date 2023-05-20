"""
Continuous set representation
Author: Yongseok Kwon
"""
import torch 

from .polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope 
from .polynomial_zonotope.mat_poly_zono import matPolyZonotope 
from .polynomial_zonotope.batch_poly_zono import batchPolyZonotope 
from .polynomial_zonotope.poly_zono import polyZonotope
from .zonotope.batch_mat_zono import batchMatZonotope 
from .zonotope.mat_zono import matZonotope 
from .zonotope.batch_zono import batchZonotope 
from .zonotope.zono import zonotope 


class __Property_ID(object):
    __dict: dict = {'None':[]}
    __properties: list = []
    __ids: torch.Tensor = torch.tensor([],dtype=torch.long)
    __device: str = 'cpu' 

    def _reset(self,device='cpu'):
        self.__dict = {'None':torch.tensor([],dtype=torch.long,device=device)}
        self.__properties = []
        self.__ids = torch.tensor([],dtype=torch.long,device=device)
        self.__device = device

    def __getitem__(self,key): 
        assert isinstance(key,str)       
        return torch.tensor(self.__dict[key],dtype=torch.long)
    @property 
    def dictionary(self):
        return self.__dict
    @property 
    def properties(self):
        return self.__properties 
    @property 
    def ids(self):
        return self.__ids
    @property 
    def device(self):
        return self.__device    
    @property
    def offset(self):
        return self.__ids.numel()
    def __repr__(self):
        return str(self.__dict)
    def __str__(self):
        return self.__repr__()
    def to(self,device):
        self.__ids = self.__ids.to(device=device)
        for key, item in self.__dict.items():
            self.__dict[key] = item.to(deivce=device)

    def update(self,num,prop='None'):
        if isinstance(prop,str):
            new_id = torch.arange(num,dtype=torch.long) + self.offset
            self.__ids = torch.hstack((self.__ids,new_id))
            if prop in self.__properties:
                self.__dict[prop] = torch.hstack((self.__dict[prop],new_id))
            else:
                self.__properties.append(prop)
                self.__dict[prop] = new_id.clone()
        '''
        elif isinstance(prop,dict):
            # prop = prop names : # gens
            ct, ct_prev = 0, 0
            for pro in prop:
                if pro in self.__properties:
                    ct_prev = ct
                    ct = ct_prev + prop[pro]
                    self.__dict[pro].extend(new_id[ct:ct_prev].tolist())
                else:
                    ct_prev = ct
                    ct = ct_prev + prop[pro]
                    self.__properties.append(pro)
                    self.__dict[pro] = new_id[ct:ct_prev].tolist()
        '''


        return new_id

PROPERTY_ID = __Property_ID()


def pz_reset(n_ids = 0, device = 'cpu'):
    PROPERTY_ID._reset(device)
    if n_ids > 0:
        PROPERTY_ID.update(n_ids)


