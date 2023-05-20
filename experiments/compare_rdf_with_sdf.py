import os
import sys
import torch
import time
import json
import argparse
import numpy as np
import gurobipy as gp
from tqdm import tqdm
from gurobipy import GRB
import sys
sys.path.append('../')
from reachability.conSet.zonotope.zono import zonotope
from reachability.forward_occupancy.FO import forward_occupancy
from environments.robots.load_robot import load_sinlge_robot_arm_params
from reachability.joint_reachable_set.load_jrs_trig import load_batch_JRS_trig, preload_batch_JRS_trig
from reachability.joint_reachable_set.process_jrs_trig import process_batch_JRS_trig
from training.model import MLP

T_PLAN, T_FULL = 0.5, 1.0

def wrap_cont_joint_to_pi_traj(phases):
    lim_flag = torch.tensor([False,  True, False,  True, False,  True, False])
    phases_new = torch.clone(phases)
    phases_new[:,~lim_flag] = (phases[:,~lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
    return phases_new

def compute_distance(qpos, qvel, k, obstacle, n_links, n_dims, link_zonos, params):
    n_timesteps = 100

    # compute forward occupancy
    if n_dims == 2:
        _, R_trig1 = load_batch_JRS_trig(qpos, qvel)
        FO_link1, _, _ = forward_occupancy(R_trig1, link_zonos, params)
    elif n_dims == 3:
        max_combs = 50
        jrs_tensor = preload_batch_JRS_trig()
        combs = [torch.tensor([0])]
        for i in range(max_combs):
            combs.append(torch.combinations(torch.arange(i+1),2))
        _, R_trig1 = process_batch_JRS_trig(jrs_tensor, qpos, qvel, joint_axes=params['joint_axes'])
        FO_link1, _, _ = forward_occupancy(R_trig1, link_zonos, params)
    
    # turn forward occupancy into convex hull
    obs_generators = obstacle.generators
    obs_center = obstacle.center.cpu().numpy()
    obstacle_distances = torch.ones(n_links, n_timesteps) * torch.inf   
    for i_joint in range(n_links):
        if n_dims == 2:
            FO_link1[i_joint] = FO_link1[i_joint].project([0,1])
        batched_link_zonotope = FO_link1[i_joint].slice_all_dep(k.view(1,n_links).repeat(100,1))

        # for every obstalce, buffer the link zono with obs zono and get a convex hull
        for t in range(n_timesteps):
            # buffer the link zonotope
            link_zono = batched_link_zonotope[t]
            buffered_link_zono_Z = torch.vstack((link_zono.Z, obs_generators))
            buffered_link_zonotope = zonotope(buffered_link_zono_Z) 
            # get vertices from buffered link zonotope
            A, b = buffered_link_zonotope.polytope(combs)
            A = A.cpu().numpy()
            b = b.cpu().numpy()

            # compute shortest distances
            constraints_evaluation = A @ obs_center - b
            if np.all(constraints_evaluation <= 0):
                max_index = np.argmax(constraints_evaluation)
                distance = constraints_evaluation[max_index]
            else:
                distance, y = optimize_for_closest_distance(A, b, obs_center, n_dims=n_dims)
                distance = np.sqrt(distance)
            obstacle_distances[i_joint, t] = distance.item()
        
    return torch.min(obstacle_distances, dim=1).values


def optimize_for_closest_distance(A, b, y, n_dims=2):
    qp = gp.Model("qp")
    x = qp.addMVar(shape=(n_dims,), name="x", vtype=GRB.CONTINUOUS, ub=np.inf, lb=-np.inf)
    x_list = x.tolist()

    if n_dims == 2:
        obj = (x_list[0] - y[0]) ** 2 + (x_list[1] - y[1]) ** 2
    elif n_dims == 3:
        obj = (x_list[0] - y[0]) ** 2 + (x_list[1] - y[1]) ** 2 + (x_list[2] - y[2]) ** 2

    qp.setObjective(obj, GRB.MINIMIZE)

    qp.addConstr(A @ x - b <= 0, "buffered_zono")
    qp.params.OutputFlag = 0
    qp.params.LogToConsole = 0
    qp.optimize()
    
    result = np.zeros(n_dims)
    for i, v in enumerate(qp.getVars()):
        result[i] = v.X
    
    return obj.getValue(), result[:n_dims]


def read_params():
    parser = argparse.ArgumentParser(description="Compare Runtime of RDF and QP")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument('--n_links', type=int, default=7)
    parser.add_argument('--n_dims', type=int, default=3)
    parser.add_argument('--n_data', type=int, default=1000)
    parser.add_argument('--n_interpolate', nargs='+', type=int)
    parser.add_argument('--stationary', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    params = read_params()
    n_links = params.n_links
    n_dims = params.n_dims

    seed = params.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # env and arm setting
    link_params, _ = load_sinlge_robot_arm_params('Kinova3')
    link_zonos = [l.to_polyZonotope() for l in link_params['link_zonos']]
    obstacle_center_range = 1.0
    obstacle_sidelength = 0.1
    
    # load trained MLP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    rdf_model_path = f'../trained_models/RDF{n_dims}D/{n_links}links/{n_dims}d-signed-convexhull.pth'
    rdf_model = MLP(n_links=n_links, n_dims=n_dims, num_hidden_layers=8, hidden_size=1024, fix_size=True)
    rdf_model.load_state_dict(torch.load(rdf_model_path, map_location=device))
    rdf_model.to(device)
    rdf_model.eval()
    
    sdf_model_path = f'../trained_models/SDF{n_dims}D/{n_links}links/{n_dims}d-signed-convexhull.pth'
    sdf_model = MLP(n_links=n_links, n_dims=n_dims, num_hidden_layers=8, hidden_size=1024, fix_size=True, sdf=True, trig=True)
    sdf_model.load_state_dict(torch.load(sdf_model_path, map_location=device))
    sdf_model.to(device)
    sdf_model.eval()
    
    rdf_time_list = []
    sdf_time_list_dict = {n_interpolate: [] for n_interpolate in params.n_interpolate}
    loop_sdf_time_list_dict = {n_interpolate: [] for n_interpolate in params.n_interpolate}
    sdf_distances_dict = {n_interpolate: torch.zeros(0, params.n_links).to(device) for n_interpolate in params.n_interpolate}
    
    ground_truth_distances = torch.zeros(0, params.n_links).to(device)
    rdf_distances = torch.zeros(0, params.n_links).to(device)

    # compute distances using QP or MLP model
    for i_data in tqdm(range(params.n_data + 1)):
        qpos_limit = torch.tensor([3.1416, 2.4100, 3.1416, 2.6600, 3.1416, 2.2300, 3.1416])
        qvel_limit = torch.tensor([1.3963, 1.3963, 1.3963, 1.3963, 1.2218, 1.2218, 1.2218])
        qpos = (torch.rand(n_links) * 2 - 1) * qpos_limit
        qvel = (torch.rand(n_links) * 2 - 1) * qvel_limit
        k = torch.rand(n_links) * 2 - 1
        obstacle_center = (torch.rand(n_dims) * 2 - 1) * obstacle_center_range
        obstacle_generators = torch.zeros(n_dims, n_dims)
        obstacle_generators[range(n_dims), range(n_dims)] = obstacle_sidelength
        obstacle_zonotope = zonotope(torch.vstack((obstacle_center, obstacle_generators)))
        
        if params.stationary:
            qvel *= 0
            k *= 0
        
        # compute ground-truth distances
        ground_truth_distance = compute_distance(qpos, qvel, k, obstacle_zonotope, n_links, n_dims, link_zonos, link_params)
        
        qpos = qpos.to(device)
        qvel = qvel.to(device)
        k = k.to(device)
        obstacle = obstacle_center.to(device)
        
        with torch.no_grad():
            ### RDF starts ###
            rdf_start = time.time()
            rdf_qpos = (qpos / qpos_limit.to(device)).repeat(1,1)
            rdf_qvel = (qvel / qvel_limit.to(device)).repeat(1,1)
            rdf_k = k.repeat(1,1)
            obstacle = obstacle.repeat(1,1)
            rdf_distance = rdf_model(qpos=rdf_qpos, qvel=rdf_qvel, obstacle=obstacle, k=rdf_k)
            rdf_time = time.time() - rdf_start
            rdf_time_list.append(rdf_time)
            ### RDF ends ###
            
            for n_interpolate in params.n_interpolate:
                sdf_time_list = sdf_time_list_dict[n_interpolate]
                loop_sdf_time_list = loop_sdf_time_list_dict[n_interpolate]
            
                ### batched SDF starts ###
                sdf_start = time.time()
                
                k *= torch.pi / 24
                T_traj = torch.linspace(0, T_FULL, n_interpolate+1).to(device)
                T_to_peak = T_traj[:int(T_PLAN / T_FULL * n_interpolate) + 1]
                T_to_brake = T_traj[int(T_PLAN / T_FULL * n_interpolate) + 1:] - T_PLAN

                qpos_to_peak = qpos + torch.outer(T_to_peak, qvel) + 0.5 * torch.outer(T_to_peak ** 2, k)
                qpos_peak, qvel_peak = qpos + qvel * T_PLAN + 0.5 * k * T_PLAN ** 2, qvel + k * T_PLAN
                bracking_accel = (0 - qvel_peak) / (T_FULL - T_PLAN)
                qpos_to_brake = qpos_peak + torch.outer(T_to_brake, qvel_peak) + 0.5 * torch.outer(T_to_brake ** 2, bracking_accel)
                qpos_traj = wrap_cont_joint_to_pi_traj(torch.vstack((qpos_to_peak,qpos_to_brake)))
                
                sdf_obstacle = obstacle.repeat(n_interpolate+1, 1)
                sdf_distance = sdf_model(qpos=qpos_traj, qvel=None, obstacle=sdf_obstacle, k=None)
                
                sdf_distance = torch.min(sdf_distance, dim=0).values
                sdf_time = time.time() - sdf_start
                sdf_time_list.append(sdf_time)
                
                sdf_distances_dict[n_interpolate] = torch.vstack((sdf_distances_dict[n_interpolate], sdf_distance.view(1,-1)))
                ### batched SDF ends ###
                
                ### loop SDF starts ###
                if n_interpolate > 0:
                    loop_sdf_start = time.time()

                    for i in range(n_interpolate + 1):     
                        loop_sdf_distance = sdf_model(qpos=qpos_traj[i:i+1], qvel=None, obstacle=sdf_obstacle[i:i+1], k=None)
                    
                    loop_sdf_time = time.time() - loop_sdf_start
                    loop_sdf_time_list.append(loop_sdf_time)
                
                ### loop SDF ends ###
            
        ground_truth_distances = torch.vstack((ground_truth_distances, ground_truth_distance.to(device)))
        rdf_distances = torch.vstack((rdf_distances, rdf_distance))
        
    
    rdf_time_list = rdf_time_list[1:]
    stats = {
        'n_dims': n_dims,
        'n_links': n_links,
        'n_data': params.n_data,
        'n_interpolate': n_interpolate,
        'time step': T_FULL / (n_interpolate+1), 
        
        'RDF time': {
            'mean': np.mean(rdf_time_list),
            'std': np.std(rdf_time_list),
        },
            
        'RDF accuracy': {
            'mean error': torch.mean(torch.abs(ground_truth_distances - rdf_distances)).cpu().item(),
            'std': torch.std(torch.abs(ground_truth_distances - rdf_distances)).cpu().item(),
            'max error': torch.max(torch.abs(ground_truth_distances - rdf_distances)).cpu().item(),
        },
    }
    
    for n_interpolate in params.n_interpolate:
        sdf_time_list = sdf_time_list_dict[n_interpolate]
        sdf_time_list = sdf_time_list[1:]
        sdf_distances = sdf_distances_dict[n_interpolate]

        loop_sdf_time_list = loop_sdf_time_list_dict[n_interpolate] 
        loop_sdf_time_list = loop_sdf_time_list[1:]
        interpolate_stats = {                
            'Batched SDF time': {
                'mean': np.mean(sdf_time_list),
                'std': np.std(sdf_time_list),
            },

            'SDF accuracy': {
                'mean error': torch.mean(torch.abs(ground_truth_distances - sdf_distances)).cpu().item(),
                'std': torch.std(torch.abs(ground_truth_distances - sdf_distances)).cpu().item(),
                'max error': torch.max(torch.abs(ground_truth_distances - sdf_distances)).cpu().item(),
            },
        }
        if n_interpolate > 0:
            interpolate_stats['Loop SDF time'] = {
                'mean': np.mean(loop_sdf_time_list),
                'std': np.std(loop_sdf_time_list),
            }, 
        
        stats[f'{n_interpolate}-interpolate'] = interpolate_stats
    
    if not os.path.exists('sdf_comparison_results'):
        os.mkdir('sdf_comparison_results')
    
    if params.stationary and not os.path.exists('sdf_comparison_results/stationary'):
        os.mkdir('sdf_comparison_results/stationary')

    with open(f"sdf_comparison_results/{'stationary/' if params.stationary else ''}results.json", 'w') as f:
        json.dump(stats, f, indent=2)
        
                    
            