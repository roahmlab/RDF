import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from training.sdf_rtd_dataset import RdfArmDataset
from training.model import MLP
from torch.autograd import grad
import argparse

def read_params():
    parser = argparse.ArgumentParser(description="SDF RTD Training")
    # general env setting
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--n_links', type=int, default=2)
    parser.add_argument('--n_dims', type=int, default=2)
    parser.add_argument('--robot', type=str, default='arm')
    parser.add_argument('--wandb_group_name', type=str, default="")

    # dataset setting
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--data', type=str, default='2d_2link_8obs_80size_signed_1-80seed.pkl')
    parser.add_argument('--signed', action='store_true') # whether to use negative distances in model inputs
    parser.add_argument('--fix_size', action='store_true') # whether the size of obstacle is fixed
    parser.add_argument('--all_timesteps', action='store_true') # whether to predict distance at all timesteps
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--sdf', action='store_true') # whether the model is learning sdf instead of rdf; the difference is whether there are qvel and k as input
    parser.add_argument('--trig', action='store_true') # whether the model is putting cos and sin as input

    # model setting
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=256)

    # learning setting
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weighted", action='store_true') # whether to use a weighted loss
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument('--use_sqrt', action='store_true') # whether to sqrt the loss when training the model
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--softplus_beta', type=float, default=1)
    parser.add_argument("--loss", type=str, default='MSE') #options: L1, MSE
    parser.add_argument("--device", type=str, default='0')
    parser.add_argument('--eikonal', type=float, default=0.0) # whether to apply eikonal regularization

    return parser.parse_args()

def make_model(params):
    if params.model == 'MLP' and params.robot == 'arm':
        model = MLP(
            n_links=params.n_links, 
            n_dims=params.n_dims,
            num_hidden_layers=params.num_hidden_layers, 
            hidden_size=params.hidden_size,
            fix_size=params.fix_size,
            all_timesteps=params.all_timesteps,
            softplus_beta=params.softplus_beta,
            sdf=params.sdf,
            trig=params.trig,
        )
        model_name = f"armMlp{params.num_hidden_layers}x{params.hidden_size}"    
    else:
        raise NotImplementedError("The specified model type has not been implemented.")

    return model, model_name


def make_datasets(params):
    if params.robot == 'arm':
        dataset = RdfArmDataset(
            dataset_dir=params.dataset_dir, 
            dataset_file=params.data,
            signed=params.signed,
            fix_size=params.fix_size,
            n_links=params.n_links,
            normalize=params.normalize
        )
    else:
        raise NotImplementedError
    
    return dataset
    

def make_dataloaders(params, training_proportion=0.8):
    dataset = make_datasets(params)
        
    training_size = int(len(dataset) * training_proportion)
    training_dataset, val_dataset = torch.utils.data.random_split(dataset, [training_size, len(dataset)-training_size])
    training_dataloader = DataLoader(training_dataset, batch_size=params.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True)
    dataloaders_dict = {'train':training_dataloader, 'val':val_dataloader}
    print(f"Data size: training {len(training_dataset)}, validation {len(val_dataset)}")

    return dataloaders_dict

def make_loss(params):
    if params.loss == 'L1':
        criterion = nn.L1Loss()
    elif params.loss == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("Such type of loss has not been implemented yet")

    return criterion


def compute_eikonal_loss(inputs, outputs):
    n_links = outputs.shape[1]
    n_dims = inputs.shape[1]
    gradients = torch.zeros(inputs.shape[0], n_links, n_dims).to(inputs.device, inputs.dtype)
    for i in range(n_links):
        gradients[:,i,:] = grad(outputs=outputs[:,i], 
                 inputs=inputs,
                 grad_outputs=torch.ones_like(outputs[:,i]),
                 retain_graph=True,
                 create_graph=True,
                 )[0]
    eikonal_loss = torch.square(torch.linalg.norm(gradients, dim=2) - 1).mean(dim=1).mean()
    return eikonal_loss


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
