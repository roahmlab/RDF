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
from reachability.joint_reachable_set.load_jrs_trig import load_batch_JRS_trig
from scipy.spatial import ConvexHull
from training.model import MLP
from torch.autograd import grad

def compute_distance(qpos, qvel, k, obstacle, n_links, n_dims, link_zonos, params):
    n_timesteps = 100

    # compute forward occupancy
    _, R_trig1 = load_batch_JRS_trig(qpos, qvel)
    FO_link1, _, _ = forward_occupancy(R_trig1, link_zonos, params)

    # turn forward occupancy into convex hull
    obs_generators = obstacle.generators
    obs_center = obstacle.center.cpu().numpy()
    obstacle_distances = torch.ones(n_links) * torch.inf
    t_optimization = 0.0 
    for i_joint in range(n_links):
        if n_dims == 2:
            FO_link1[i_joint] = FO_link1[i_joint].project([0,1])
        batched_link_zonotope = FO_link1[i_joint].slice_all_dep(k.view(1,n_links).repeat(100,1))

        vertices_over_timesteps = torch.zeros(0, n_dims)
        # for every obstalce, buffer the link zono with obs zono and get a convex hull
        for t in range(n_timesteps):
            # buffer the link zonotope
            link_zono = batched_link_zonotope[t]
            buffered_link_zono_Z = torch.vstack((link_zono.Z, obs_generators))
            buffered_link_zonotope = zonotope(buffered_link_zono_Z) 
            # get vertices from buffered link zonotope
            if n_dims == 2:     
                new_vertices = buffered_link_zonotope.polygon()
            elif n_dims == 3:
                new_vertices = buffered_link_zonotope.polyhedron()
            vertices_over_timesteps = torch.cat([vertices_over_timesteps, new_vertices])
            
        hull = ConvexHull(vertices_over_timesteps.cpu().numpy())
        A_b = hull.equations
        A, b = A_b[:,:-1], -A_b[:,-1]

        # compute shortest distances
        distance_time_start = time.time()
        constraints_evaluation = A @ obs_center - b       
        if np.all(constraints_evaluation <= 0):
            max_index = np.argmax(constraints_evaluation)
            distance = constraints_evaluation[max_index]
        else:
            distance, y = optimize_for_closest_distance(A, b, obs_center, n_dims=n_dims)
            distance = np.sqrt(distance)
        t_optimization += time.time() - distance_time_start
        obstacle_distances[i_joint] = distance
        
    return obstacle_distances, t_optimization
    

def compute_jacobian(qpos, qvel, k, obstacle, n_links, n_dims, link_zonos, params, s):
    epsilon = 1e-3
    jacobian = torch.zeros((n_links, n_links))
    t_distance_total = 0
    for i_k_joint in range(n_links):
        delta = torch.zeros(n_links)
        delta[i_k_joint] = epsilon
        k_plus = k + delta
        s_plus, t_distance = compute_distance(qpos, qvel, k_plus, obstacle, n_links, n_dims, link_zonos, params)
        t_distance_total += t_distance
        jacobian[:, i_k_joint] = (s_plus - s) / epsilon
    return jacobian, t_distance_total


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
    qp.params.Threads = 16
    qp.optimize()
    
    result = np.zeros(n_dims)
    for i, v in enumerate(qp.getVars()):
        result[i] = v.X
    
    return obj.getValue(), result[:n_dims]


def read_params():
    parser = argparse.ArgumentParser(description="Compare Runtime of RDF and QP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--n_links', type=int, default=2)
    parser.add_argument('--n_dims', type=int, default=2)
    parser.add_argument('--n_data', type=int, default=1000)
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    params = read_params()
    n_links = params.n_links
    n_dims = params.n_dims

    seed = params.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # env and arm setting
    if n_dims == 2:
        scale_down = 1.2 * n_links
        link_params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*n_links, 
            'R': [torch.eye(3)]*n_links,
            'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0]) / scale_down]*(n_links-1),
            'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(n_links-1),
            'n_joints':n_links}
        link_zonos = [zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T / scale_down).to_polyZonotope()] * n_links    
        obstacle_center_range = 1.0
        obstacle_sidelength = 0.1 / (1.2 * n_links)
    elif n_dims == 3:
        link_params, _ = load_sinlge_robot_arm_params('Kinova3')
        link_zonos = [l.to_polyZonotope() for l in link_params['link_zonos']]
        obstacle_center_range = 1.0
        obstacle_sidelength = 0.1
    
    # load trained MLP model
    model_path = f'../trained_models/RDF{n_dims}D/{n_links}links/{n_dims}d-signed-convexhull.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(n_links=n_links, n_dims=n_dims, num_hidden_layers=8, hidden_size=1024, fix_size=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    qp_time_list = []
    qp_time_optimization_list = []
    qp_jacobian_time_list = []
    qp_jacobian_optimization_time_list = []
    model_time_list = []
    model_jacobian_time_list = []
    # compute distances using QP or MLP model
    for i_data in tqdm(range(params.n_data + 1)):
        if n_dims == 2:
            qpos = (torch.rand(n_links) * 2 - 1) * torch.pi
            qvel = (torch.rand(n_links) * 2 - 1) * torch.pi / 2.0
        elif n_dims == 3:
            pos_coefficient = torch.tensor([limit[0] for limit in link_params['pos_lim']])
            vel_coefficient = torch.tensor(link_params['vel_lim'])
            qpos = (torch.rand(n_links) * 2 - 1) * pos_coefficient
            qvel = (torch.rand(n_links) * 2 - 1) * vel_coefficient
            
        k = torch.rand(n_links) * 2 - 1
        obstacle_center = (torch.rand(n_dims) * 2 - 1) * obstacle_center_range
        obstacle_generators = torch.zeros(n_dims, n_dims)
        obstacle_generators[range(n_dims), range(n_dims)] = obstacle_sidelength
        obstacle_zonotope = zonotope(torch.vstack((obstacle_center, obstacle_generators)))
        
        # compare time
        qp_start = time.time()
        qp_distance, qp_time_optimization = compute_distance(qpos, qvel, k, obstacle_zonotope, n_links, n_dims, link_zonos, link_params)
        qp_time = time.time() - qp_start
        
        qp_jacobian_start = time.time()
        qp_jacobian, qp_jacobian_optimization_time = compute_jacobian(qpos, qvel, k, obstacle_zonotope, n_links, n_dims, link_zonos, link_params, qp_distance)
        qp_jacobian_time = time.time() - qp_jacobian_start
        
        
        model_start = time.time()
        k_with_grad = k.clone().requires_grad_()
        model_distance = model(qpos=qpos.repeat(1,1).to(device) / torch.pi, qvel=qvel.repeat(1,1).to(device) / (torch.pi / 2), obstacle=obstacle_center.repeat(1,1).to(device), k=k_with_grad.repeat(1,1).to(device))
        model_time = time.time() - model_start
        
        model_jacobian = torch.zeros(n_links, n_links)
        for i_link in range(n_links):
            model_jacobian[i_link] = grad(outputs=model_distance[0, i_link], 
                                inputs=k_with_grad, 
                                retain_graph=True)[0][0]
        model_jacobian_time = time.time() - model_start
        
        qp_time_list.append(qp_time)
        qp_time_optimization_list.append(qp_time_optimization)
        qp_jacobian_time_list.append(qp_jacobian_time)
        qp_jacobian_optimization_time_list.append(qp_jacobian_optimization_time)
        model_time_list.append(model_time)
        model_jacobian_time_list.append(model_jacobian_time)
    
    qp_time_list = qp_time_list[1:]
    qp_time_optimization_list = qp_time_optimization_list[1:]
    qp_jacobian_time_list = qp_jacobian_time_list[1:]
    qp_jacobian_optimization_time_list = qp_jacobian_optimization_time_list[1:]
    model_time_list = model_time_list[1:]
    model_jacobian_time_list = model_jacobian_time_list[1:]
    stats = {
        'n_dims': n_dims,
        'n_links': n_links, 
        'n_data': params.n_data,
        
        'QP distance total time': {
                'mean': np.mean(qp_time_list),
                'std': np.std(qp_time_list),
        },
        
        'QP distance optimization time': {
                'mean': np.mean(qp_time_optimization_list),
                'std': np.std(qp_time_optimization_list),
        },
            
        'QP jacobian total time': {
                'mean': np.mean(qp_jacobian_time_list),
                'std': np.std(qp_jacobian_time_list),
        },
        
        'QP jacobian optimization total time': {
                'mean': np.mean(qp_jacobian_optimization_time_list),
                'std': np.std(qp_jacobian_optimization_time_list),
        },
            
        'model distance time': {
                'mean': np.mean(model_time_list),
                'std': np.std(model_time_list),
        },
            
        'model jacobian time': {
                'mean': np.mean(model_jacobian_time_list),
                'std': np.std(model_jacobian_time_list),
        },
    }
    print(stats)
    if not os.path.exists('qp_comparison_results'):
        os.mkdir('qp_comparison_results')

    with open(f"qp_comparison_results/{n_dims}d{n_links}links.json", 'w') as f:
        json.dump(stats, f, indent=2)
        
                    
            