"""
the script to generate dataset to learn a signed distance function
dataset format:
{
    'qpos'    :  N by n_links, configurations (rad),
    'qvel'    :  N by n_links, angular velocity of each joint (rad/s),
    'k'       :  N by n_links, trajectory parameter (rad/s^2),
    'obstacle':  N by (2 * n_dims), center and side length of an obstacle x,y,a,b (m),
    's'       :  N by n_links, the shortest distance from the arm to the obstacle (m)
}
"""
import sys
sys.path.append('../')
import os
import torch
import pickle
import argparse
import numpy as np
import gurobipy as gp
from tqdm import tqdm
from gurobipy import GRB
import matplotlib.pyplot as plt
from reachability.conSet import zonotope
from reachability.joint_reachable_set.load_jrs_trig import load_batch_JRS_trig
from reachability.forward_occupancy.FO import forward_occupancy
from scipy.spatial import ConvexHull


def generate_dataset(params, link_zonos,n_links=2, n_dims=2, N=1e6, signed=False, num_obstacles_each_initial_condition = 5, output_dataset_filename="link.pkl", verbose=False, plot=False, save=False):
    N_joints = n_links
    n_timesteps = 100
    
    obstacle_center_range = 1.0
    # obstacle_sidelength_range = 0.4 #0.6
    min_obstacle_sidelength = 0.1 / (1.2 * N_joints) #0.2 

    qpos_dataset = []
    qvel_dataset = []
    k_dataset = []
    obstacle_dataset = []
    s_dataset = []

    for i_data in tqdm(range(N // num_obstacles_each_initial_condition)):
        # random configurations
        qpos = (torch.rand(n_links) * 2 - 1) * torch.pi
        qvel = (torch.rand(n_links) * 2 - 1) * torch.pi / 2.0
        k = torch.rand(n_links) * 2 - 1
        
        if verbose:
            print(f"{i_data}-th data:Random initial condition:\nqpos={qpos},\nqvel={qvel},\nk={k}")

        # randomly generate obstacles
        obstacles = []
        obstacle_tensors = []
        for i_obs in range(num_obstacles_each_initial_condition):
            obstacle_center = (torch.rand(n_dims) * 2 - 1) * obstacle_center_range
            obstacle_sidelength = torch.ones(n_dims) * min_obstacle_sidelength #torch.rand(n_dims) * obstacle_sidelength_range + min_obstacle_sidelength
            obstacle_generators = torch.zeros(n_dims, n_dims)
            obstacle_generators[range(n_dims), range(n_dims)] = obstacle_sidelength
            obstacle_zonotope = zonotope(torch.vstack((obstacle_center, obstacle_generators)))
            obstacles.append(obstacle_zonotope)
            obstacle_tensors.append(torch.cat([obstacle_center, obstacle_sidelength]))
            if verbose:
                print(f"Obstacle {i_obs}: center={obstacle_center}, sidelength={obstacle_sidelength}")

        # compute forward occupancy
        _, R_trig1 = load_batch_JRS_trig(qpos, qvel)
        FO_link1, _, _ = forward_occupancy(R_trig1, link_zonos, params)

        # turn forward occupancy into convex hull
        obs_generators = obstacles[0].generators
        convex_hulls = [] # n_links by n_obs
        batched_links_zonotopes = []
        A_b_pairs_list = []
        for i_joint in range(N_joints):
            if n_dims == 2:
                FO_link1[i_joint] = FO_link1[i_joint].project([0,1])
            batched_link_zonotope = FO_link1[i_joint].slice_all_dep(k.view(1,n_links).repeat(100,1))
            batched_links_zonotopes.append(batched_link_zonotope)

            vertices_over_timesteps = torch.zeros(0, n_dims)
            # for every obstalce, buffer the link zono with obs zono and get a convex hull
            for t in range(n_timesteps):

                # buffer the link zonotope
                link_zono = batched_links_zonotopes[i_joint][t]
                buffered_link_zono_Z = torch.vstack((link_zono.Z, obs_generators))
                buffered_link_zonotope = zonotope(buffered_link_zono_Z) 

                # get vertices from buffered link zonotope       
                vertices_over_timesteps = torch.cat([vertices_over_timesteps, buffered_link_zonotope.polygon()])
            
            hull = ConvexHull(vertices_over_timesteps.cpu().numpy())
            convex_hulls.append(hull)
            A_b = hull.equations
            A, b = A_b[:,:-1], -A_b[:,-1]
            A_b_pairs_list.append((A,b))
        
        # if plot, plot the convex hull for the first obstacle
        if plot:
            fig, (ax, ax2) = plt.subplots(ncols=2)
            for i_joint in range(n_links):
                hull = convex_hulls[i_joint]
                points = hull.points[hull.vertices]
                points = np.vstack((points, points[0]))
                ax2.plot(points[:,0], points[:,1])
            for i_obs, obs in enumerate(obstacles):
                c = obs.center
                ax2.scatter(c[0], c[1], c='red', s=3)

        # compute shortest distances
        obstacle_distances = np.ones((num_obstacles_each_initial_condition, n_links)) * np.inf   
        optimal_ys = np.zeros((num_obstacles_each_initial_condition, n_links, n_dims))
        for i_joint in range(N_joints):
            if verbose > 1:
                print(f"{i_joint}-th link")
            if plot:
                batched_link_zonotope = batched_links_zonotopes[i_joint]
                for _ in range(100):
                    if _ % 10 == 0:
                        batched_link_zonotope[_].plot(ax)
            A, b = A_b_pairs_list[i_joint]
            for i_obs, obs in enumerate(obstacles):
                if plot:
                    obs.plot(ax, linewidth=1, edgecolor='orange')
                    
                try:
                    # in case solver error breaks the whole data generation process
                    obs_center = obs.center.cpu().numpy()
                    
                    constraints_evaluation = A @ obs_center - b       
                    if np.all(constraints_evaluation <= 0):
                        max_index = np.argmax(constraints_evaluation)
                        distance = constraints_evaluation[max_index]
                        kk = distance / np.sum(np.square(A[max_index]))
                        y = obs_center - kk * A[max_index]
                    else:
                        distance, y = optimize_for_closest_distance(A, b, obs_center, n_dims=n_dims)
                        distance = np.sqrt(distance)
                    obstacle_distances[i_obs, i_joint] = distance
                    optimal_ys[i_obs][i_joint] = np.copy(y)
                except Exception as e:
                    if verbose:
                        print(e)
                        print("Solver error; leaving out this part of data...")
                        
        if verbose:
            for i_obs in range(num_obstacles_each_initial_condition):
                print(f"The solution is found on y={optimal_ys[i_obs]}, d={obstacle_distances[i_obs]}")
        
        if plot:
            for i_obs, obs in enumerate(obstacles):
                c = obs.center.cpu().numpy()
                for i_joint in range(n_links):
                    closest_point = optimal_ys[i_obs][i_joint]
                    ax2.plot([c[0], closest_point[0]], [c[1], closest_point[1]], c='grey', linewidth=0.4)

            if not os.path.exists("vizs_convexhull"):
                os.mkdir("vizs_convexhull")
            if not os.path.exists(f"vizs_convexhull/2d{n_links}links_viz{num_obstacles_each_initial_condition}"):
                os.mkdir(f'vizs_convexhull/2d{n_links}links_viz{num_obstacles_each_initial_condition}')
            r = 1.2
            ax.axis('square')
            ax.set_xlim([-r,r])
            ax.set_ylim([-r,r])
            ax2.axis('square')
            ax2.set_xlim([-r,r])
            ax2.set_ylim([-r,r])
            plt.savefig(f"vizs_convexhull/2d{n_links}links_viz{num_obstacles_each_initial_condition}/{i_data}.jpg", dpi=600)
            plt.close()

        if save:
            for i_obs in range(num_obstacles_each_initial_condition):
                if np.any(obstacle_distances[i_obs] == np.inf):
                    continue
                qpos_dataset.append(qpos.numpy())
                qvel_dataset.append(qvel.numpy())
                k_dataset.append(k.numpy())
                obstacle_dataset.append(obstacle_tensors[i_obs].numpy())
                s_dataset.append(obstacle_distances[i_obs])
    if save:
        dataset = {
            'qpos'    : np.array(qpos_dataset),
            'qvel'    : np.array(qvel_dataset),
            'k'       : np.array(k_dataset),
            'obstacle': np.array(obstacle_dataset),
            's'       : np.array(s_dataset)
        }
        if signed:
            prefix = "signed"
        else:
            prefix = "unsigned"
        if not os.path.exists("2d_signed_rdf_dataset"):
                os.mkdir("2d_signed_rdf_dataset")
        with open(f"2d_signed_rdf_dataset/{prefix}{n_dims}d{n_links}link{num_obstacles_each_initial_condition}obs{len(qpos_dataset)}size{output_dataset_filename}", "wb") as f:
            pickle.dump(dataset, f)
        print(f"Generated {len(qpos_dataset)} data.")

def optimize_for_closest_distance(A, b, y, n_dims=2):
    qp = gp.Model("qp")
    x = qp.addMVar(shape=(2,), name="x", vtype=GRB.CONTINUOUS, ub=np.inf, lb=-np.inf)
    x_list = x.tolist()

    obj = (x_list[0] - y[0]) ** 2 + (x_list[1] - y[1]) ** 2
    qp.setObjective(obj, GRB.MINIMIZE)

    qp.addConstr(A @ x - b <= 0, "buffered_zono")
    qp.params.OutputFlag = 0
    qp.optimize()
    
    result = np.zeros(n_dims)
    for i, v in enumerate(qp.getVars()):
        result[i] = v.X
    
    return obj.getValue(), result[:2]


def read_params():
    parser = argparse.ArgumentParser(description="SDF RTD Dataset Generation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--n_links', type=int, default=10)
    parser.add_argument('--n_dims', type=int, default=2)
    parser.add_argument('--n_obs', type=int, default=8)
    parser.add_argument('--n_data', type=int, default=64)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--output_path', type=str, default='.pkl')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--signed', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    params = read_params()
    N_joints = params.n_links
    n_dims = params.n_dims
    scale_down = 1.2 * N_joints

    seed = params.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_filename = f"{seed}seed{params.output_path}"

    link_params = {'joint_axes':[torch.tensor([0.0,0.0,1.0])]*N_joints, 
        'R': [torch.eye(3)]*N_joints,
        'P': [torch.tensor([0.0,0.0,0.0])]+[torch.tensor([1.0,0.0,0.0]) / scale_down]*(N_joints-1),
        'H': [torch.eye(4)]+[torch.tensor([[1.0,0,0,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])]*(N_joints-1),
        'n_joints':N_joints}
    link_zonos = [zonotope(torch.tensor([[0.5,0.5,0.0],[0.0,0.0,0.01],[0.0,0.0,0.0]]).T / scale_down).to_polyZonotope()]*N_joints
    
    generate_dataset(n_links=N_joints, n_dims=n_dims, N=params.n_data, 
        num_obstacles_each_initial_condition=params.n_obs, output_dataset_filename=output_filename, 
        verbose=params.verbose, plot=True, save=params.save, signed=params.signed, 
        params=link_params, link_zonos=link_zonos)
