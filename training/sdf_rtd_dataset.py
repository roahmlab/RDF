import os
import torch
import pickle
from torch.utils.data import Dataset

class RdfArmDataset(Dataset):
    def __init__(self, dataset_file, dataset_dir, signed=False, fix_size=False, n_links=2, normalize=False) -> Dataset:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.data_file = os.path.join(self.dataset_dir, dataset_file)
        self.is_signed = signed
        self.fix_size = fix_size
        self.n_links = n_links
        self.normalize = normalize
        with open(self.data_file, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return self.data['qpos'].shape[0]
    
    def __getitem__(self, index):
        qpos = torch.from_numpy(self.data['qpos'][index]).to(torch.float)
        qvel = torch.from_numpy(self.data['qvel'][index]).to(torch.float)
        obstacle = torch.from_numpy(self.data['obstacle'][index]).to(torch.float)
        k = torch.from_numpy(self.data['k'][index]).to(torch.float)
        s = torch.from_numpy(self.data['s'][index][:self.n_links]).to(torch.float).flatten()
        
        if self.normalize:
            dim = obstacle.shape[0] // 2
            if dim == 2:
                qpos = qpos / torch.pi
                qvel = qvel / (torch.pi / 2)
            elif dim == 3:
                qpos_limit = torch.tensor([3.1416, 2.4100, 3.1416, 2.6600, 3.1416, 2.2300, 3.1416])
                qvel_limit = torch.tensor([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
                qpos = qpos / qpos_limit
                qvel = qvel / qvel_limit
                
        # Only keep obstacle center position if obstacle size is supposed to be fixed
        if self.fix_size:
            dim = obstacle.shape[0]
            obstacle = obstacle[:dim // 2].requires_grad_()
        
        return qpos, qvel, obstacle, k, s
