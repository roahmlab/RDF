from tqdm import tqdm
import torch
import argparse
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
from training.utils import set_random_seed, make_model, make_datasets


def eval_model(model, dataloader, params):
    device = params.device
    model = model.to(device)
    model.eval()

    # Iterate over data.
    with torch.no_grad(): 
        differences = torch.zeros(0, params.n_links).to(device)
        for data_tuple in tqdm(dataloader):
            if params.robot == 'arm':
                qpos, qvel, obstacle, k, labels = data_tuple
                qpos, qvel, obstacle, k = qpos.to(device), qvel.to(
                    device), obstacle.to(device), k.to(device)
                labels = labels.to(device)
                
                outputs_data = model(qpos=qpos, qvel=qvel, obstacle=obstacle, k=k).detach()
                differences = torch.vstack((differences, torch.abs(outputs_data - labels)))
            
    stats = {
        'mean_difference': differences.mean().item(),
        'std_difference': differences.std().item()
    }
    return stats

def read_params():
    parser = argparse.ArgumentParser(description="Model evaluation on testset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--n_links', type=int, default=2)
    parser.add_argument('--n_dims', type=int, default=2)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--softplus_beta', type=float, default=1)
    parser.add_argument('--all_timesteps', action='store_true')

    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--robot", type=str, default='arm')
    parser.add_argument("--normalize", action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    params = read_params()
    n_dims = params.n_dims
    n_links = params.n_links
    
    params.__setattr__('fix_size', True)
    params.__setattr__('signed', True)
    params.__setattr__("normalize", True)
    params.__setattr__("sdf", False)
    params.__setattr__("trig", False)
    params.__setattr__('dataset_dir', '../test_dataset')
    params.__setattr__('data', f'{n_dims}d_{n_links}link_16obs_20size_signed_81-100seed.pkl')
    if n_dims == 3:
        params.__setattr__('data', f'{n_dims}d_{n_links}link_16obs_40size_signed_161-200seed.pkl')
    
    set_random_seed(params.seed)   
    
    dataset = make_datasets(params=params)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)
    model, _ = make_model(params)
    model_name = f'../trained_models/RDF{n_dims}D/{n_links}links/{n_dims}d-signed-convexhull.pth'
    model.load_state_dict(torch.load(model_name, map_location=params.device))
    
    stats = eval_model(model, dataloader, params)
    print(f"Stats with model={model_name} on dataset={params.data} is {stats}")
