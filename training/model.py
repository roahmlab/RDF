import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_links=2, n_dims=2, num_hidden_layers=3, hidden_size=256, fix_size=True, all_timesteps=False, softplus_beta=1, sdf=False, trig=True):
        super().__init__()
        self.input_size = n_links + (2 - fix_size) * n_dims if sdf else n_links * 3 + (2 - fix_size) * n_dims
        self.trig = trig
        if sdf and trig:
            self.input_size += n_links * 2
        
        self.num_layers = num_hidden_layers + 2
        self.sdf = sdf
        if softplus_beta > 0:
            self.activation_layer = nn.Softplus(beta=softplus_beta)
        else:
            self.activation_layer = nn.ReLU()

        self.linear0 = nn.Linear(self.input_size, hidden_size)
        for layer_id in range(1, num_hidden_layers + 1):
            if layer_id == num_hidden_layers // 2:
                output_dim = hidden_size - self.input_size
            else:
                output_dim = hidden_size
            setattr(self, f'linear{layer_id}', nn.Linear(hidden_size, output_dim))
        if all_timesteps:
            setattr(self, f'linear{num_hidden_layers+1}', nn.Linear(hidden_size, n_links * 100))
        else:
            setattr(self, f'linear{num_hidden_layers+1}', nn.Linear(hidden_size, n_links))
    
    def forward(self, qpos, qvel, obstacle, k):
        if self.sdf and self.trig:
            inputs = torch.cat([qpos, torch.cos(qpos), torch.sin(qpos), obstacle], dim=1)
        elif self.sdf and not self.trig:
            inputs = torch.cat([qpos, obstacle], dim=1)
        else:
            inputs = torch.cat([qpos, qvel, obstacle, k], dim=1) 
        outputs = inputs
        for i in range(self.num_layers):
            layer = getattr(self, f"linear{i}")
            if i == self.num_layers // 2:
                outputs = torch.cat([outputs, inputs], dim=-1)
            outputs = layer(outputs)
            if i != self.num_layers - 1:
                outputs = self.activation_layer(outputs)
        return outputs
    