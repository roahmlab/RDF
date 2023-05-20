"""
the script to generate dataset to learn a signed distance function
dataset format:
{
    'qpos'    :  N by n_links, configurations (rad),
    'qvel'    :  N by n_links, angular velocity of each joint (rad/s),
    'k'       :  N by n_links, trajectory parameter (rad/s^2),
    'obstacle':  N by (3 * n_dims), center and side length of an obstacle x,y,a,b (m),
    's'       :  N by n_links, the shortest distance from the arm to the obstacle (m)
}
"""

import sys
sys.path.append('../')
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import pickle
import argparse
import numpy as np
import gurobipy as gp
from tqdm import tqdm
from gurobipy import GRB
import matplotlib.pyplot as plt
from reachability.conSet import zonotope
from reachability.joint_reachable_set.load_jrs_trig import preload_batch_JRS_trig
from reachability.joint_reachable_set.process_jrs_trig import process_batch_JRS_trig
from reachability.forward_occupancy.FO import forward_occupancy
from scipy.spatial import ConvexHull



def generate_dataset(params, link_zonos,n_links=7, n_dims=3, N=1e6, signed=False, num_obstacles_each_initial_condition = 5, output_dataset_filename="link.pkl", verbose=False, plot=False, save=False):
    N_joints = n_links
    n_timesteps = 100
    
    obstacle_center_range = 1.0 #3
    obstacle_sidelength_range = 0.4 #0.6 # not used now
    min_obstacle_sidelength = 0.1 #0.2 # fixed sidelength 0.1 now

    qpos_dataset = []
    qvel_dataset = []
    k_dataset = []
    obstacle_dataset = []
    s_dataset = []

    max_combs = 50
    jrs_tensor = preload_batch_JRS_trig()
    combs = [torch.tensor([0])]
    for i in range(max_combs):
        combs.append(torch.combinations(torch.arange(i+1),2))

    # pos limit and vel limit
    pos_coefficient = torch.tensor([limit[0] for limit in params['pos_lim']])
    vel_coefficient = torch.tensor(params['vel_lim'])

    for i_data in tqdm(range(N // num_obstacles_each_initial_condition)):
        # random configurations
        qpos = (torch.rand(n_links) * 2 - 1) * pos_coefficient
        qvel = (torch.rand(n_links) * 2 - 1) * vel_coefficient * 0
        k = (torch.rand(n_links) * 2 - 1) * 0
        
        num_non_collision_distance = 0

        if verbose:
            print(f"{i_data}-th data:Random initial condition:\nqpos={qpos},\nqvel={qvel},\nk={k}")

        # randomly generate obstacles
        obstacles = []
        obstacle_tensors = []
        obstacle_center = (torch.rand(n_dims) * 2 - 1) * obstacle_center_range
        obstacle_sidelength = torch.ones(n_dims) * min_obstacle_sidelength #torch.rand(n_dims) * obstacle_sidelength_range + min_obstacle_sidelength
        obstacle_generators = torch.zeros(n_dims, n_dims)
        obstacle_generators[range(n_dims), range(n_dims)] = obstacle_sidelength
        obstacle_zonotope = zonotope(torch.vstack((obstacle_center, obstacle_generators)))

        # compute forward occupancy
        _, R_trig1 = process_batch_JRS_trig(jrs_tensor, qpos, qvel, joint_axes=params['joint_axes'])
        FO_link1, _, _ = forward_occupancy(R_trig1, link_zonos, params)

        # CONVEXHULL: turn forward occupancy into convex hull
        obs_generators = obstacle_zonotope.generators
        convex_hulls = [] # n_links by n_obs, 7 by n_obs
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
                vertices_over_timesteps = torch.cat([vertices_over_timesteps, buffered_link_zonotope.polyhedron()])
            
            hull = ConvexHull(vertices_over_timesteps.cpu().numpy())
            convex_hulls.append(hull)
            A_b = hull.equations
            A, b = A_b[:,:-1], -A_b[:,-1]
            A_b_pairs_list.append((A,b))

        # if plot, plot the convex hull for the first obstacle
        if plot:
            fig, (ax,ax2) = plt.subplots(1,2,subplot_kw=dict(projection='3d'))
            for i_joint in range(n_links):
                hull = convex_hulls[i_joint] # TODO: [0] because all obstacles all the same
                points = hull.points
                points = points.reshape((1,)+points.shape)
                hull_patch = np.concatenate([points[:,s] for s in hull.simplices])
                ax2.add_collection3d(Poly3DCollection(hull_patch,edgecolor='orange',facecolor='orange',alpha=0.15,linewidths=0.5))
            for i_obs, obs in enumerate(obstacles):
                c = obs.center
                ax2.scatter(c[0], c[1], c[2], c='red', s=3)

        if plot:
            # plot the initial pos of the link 
            w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]])
            rot_skew_sym = (w@torch.vstack(params['joint_axes']).T).transpose(0,-1)
            qpos = qpos.reshape(qpos.shape+(1,1))
            R_q =  torch.eye(3) + torch.sin(qpos) * rot_skew_sym + (1-torch.cos(qpos)) * rot_skew_sym @ rot_skew_sym

            R, P = torch.eye(3), torch.zeros(3)  
            link_patches = []          
            for j in range(n_links):
                P = R @ params['P'][j] + P 
                R = R @ params['R'][j] @ R_q[j]
                link_patches.extend((R @ link_zonos[j].to_zonotope()+P).polyhedron_patch())
            ax.add_collection3d(Poly3DCollection(link_patches,edgecolor='blue',facecolor='blue',alpha=0.15,linewidths=0.5))

        # compute shortest distances
        obstacle_distances = np.zeros((num_obstacles_each_initial_condition, n_links))
        optimal_ys = np.zeros((num_obstacles_each_initial_condition, n_links, n_dims))
        
        while num_non_collision_distance < num_obstacles_each_initial_condition // 2:
            obstacle_center = (torch.rand(n_dims) * 2 - 1) * obstacle_center_range
            obstacle_sidelength = torch.ones(n_dims) * min_obstacle_sidelength #torch.rand(n_dims) * obstacle_sidelength_range + min_obstacle_sidelength
            obstacle_generators = torch.zeros(n_dims, n_dims)
            obstacle_generators[range(n_dims), range(n_dims)] = obstacle_sidelength
            obs = zonotope(torch.vstack((obstacle_center, obstacle_generators)))

            for i_joint in range(N_joints):
                
                A, b = A_b_pairs_list[i_joint]
                        
                try:
                    # in case solver error breaks the whole data generation process
                    obs_center = obs.center.cpu().numpy()
                    constraints_evaluation = A @ obs_center - b       

                    if not np.all(constraints_evaluation <= 0):
                        distance, y = optimize_for_closest_distance(A, b, obs_center, n_dims=n_dims)
                        distance = np.sqrt(distance)
                        if distance < 0.03:
                            break
                    else:
                        distance = 0.0
                        break
                    obstacle_distances[num_non_collision_distance, i_joint] = distance
                    optimal_ys[num_non_collision_distance][i_joint] = np.copy(y)
                                
                except Exception as e:
                    if verbose:
                        print(e)
                        print("Solver error; leaving out this part of data...")
                        
            if np.all(obstacle_distances[num_non_collision_distance] >= 0.03):
                num_non_collision_distance += 1
                obstacles.append(obs)
                obstacle_tensors.append(torch.cat([obstacle_center, obstacle_sidelength]))
            
        for i_collision_data in range(num_obstacles_each_initial_condition // 2):
            collision_link = np.random.randint(low=0, high=n_links)
            hull = convex_hulls[collision_link]
            bbox = [hull.min_bound, hull.max_bound]
            bbox[0] -= 0.03
            bbox[1] += 0.03
            distance = np.inf
            A, b = A_b_pairs_list[collision_link]
            
            while True:
                obstacle_center = np.array([np.random.uniform(bbox[0][0], bbox[1][0]), np.random.uniform(bbox[0][1], bbox[1][1]), np.random.uniform(bbox[0][2], bbox[1][2])])
                obstacle_center = torch.from_numpy(obstacle_center)
                obstacle_sidelength = torch.ones(n_dims) * min_obstacle_sidelength #torch.rand(n_dims) * obstacle_sidelength_range + min_obstacle_sidelength
                obstacle_generators = torch.zeros(n_dims, n_dims)
                obstacle_generators[range(n_dims), range(n_dims)] = obstacle_sidelength
                obs = zonotope(torch.vstack((obstacle_center, obstacle_generators)))
                try:
                    # in case solver error breaks the whole data generation process
                    obs_center = obs.center.cpu().numpy()
                    constraints_evaluation = A @ obs_center - b       

                    if np.all(constraints_evaluation <= 0):
                        # the obstacle center is always within the buffered FO (negative distance)
                        max_index = np.argmax(constraints_evaluation)
                        distance = constraints_evaluation[max_index]
                        kk = distance / np.sum(np.square(A[max_index]))
                        y = obs_center - kk * A[max_index]
                    else:
                        distance, y = optimize_for_closest_distance(A, b, obs_center, n_dims=n_dims)
                        distance = np.sqrt(distance)
                    if distance < 0.03:
                        #import pdb;pdb.set_trace()
                        for i_joint in range(N_joints):
                            A, b = A_b_pairs_list[i_joint]
                            try:
                                # in case solver error breaks the whole data generation process
                                obs_center = obs.center.cpu().numpy()
                                constraints_evaluation = A @ obs_center - b       
                                if np.all(constraints_evaluation <= 0):
                                    # the obstacle center is always within the buffered FO (negative distance)
                                    max_index = np.argmax(constraints_evaluation)
                                    distance = constraints_evaluation[max_index]
                                    kk = distance / np.sum(np.square(A[max_index]))
                                    y = obs_center - kk * A[max_index]
                                else:
                                    distance, y = optimize_for_closest_distance(A, b, obs_center, n_dims=n_dims)
                                    distance = np.sqrt(distance)

                                obstacle_distances[num_non_collision_distance+i_collision_data, i_joint] = distance
                                optimal_ys[num_non_collision_distance][i_joint] = np.copy(y)
                            except Exception:
                                pass
                        obstacles.append(obs)
                        obstacle_tensors.append(torch.cat([obstacle_center, obstacle_sidelength]))
                        break
                except Exception:
                    pass

        if plot:
            for i_obs in range(num_obstacles_each_initial_condition):
                obs = obstacles[i_obs]
                obs_patch = obs.polyhedron_patch()
                ax.add_collection3d(Poly3DCollection(obs_patch, edgecolor='orange', linewidths=0.2))
                c = obstacles[i_obs].center.cpu().numpy()
                for i_joint in range(n_links):
                    closest_point = optimal_ys[i_obs][i_joint]
                    ax2.plot([c[0], closest_point[0]], [c[1], closest_point[1]], [c[2], closest_point[2]], c='black', linewidth=1)

            if not os.path.exists("vizs3d_sdf"):
                os.mkdir("vizs3d_sdf")
            if not os.path.exists(f"vizs3d_sdf/viz{num_obstacles_each_initial_condition}"):
                os.mkdir(f'vizs3d_sdf/viz{num_obstacles_each_initial_condition}')
            #ax.axis('square')
            lim = 1.1
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.set_zlim([-lim, lim])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            plt.savefig(f"vizs3d_sdf/viz{num_obstacles_each_initial_condition}/{i_data}.jpg", dpi=600)
            plt.close()
            
        if verbose:
            for i_obs in range(num_obstacles_each_initial_condition):
                print("obs:", obstacle_tensors[i_obs])
                print("obs distance:", obstacle_distances[i_obs])

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
        prefix = "signed"
        if not os.path.exists("3d_signed_sdf_dataset"):
                os.mkdir("3d_signed_sdf_dataset")
        with open(f"3d_signed_sdf_dataset/{prefix}{n_dims}d{n_links}link{num_obstacles_each_initial_condition}obs{len(qpos_dataset)}size{output_dataset_filename}", "wb") as f:
            pickle.dump(dataset, f)
        print(f"Generated {len(qpos_dataset)} data.")

def optimize_for_closest_distance(A, b, y, n_dims=3):
    qp = gp.Model("qp")
    x = qp.addMVar(shape=(n_dims,), name="x", vtype=GRB.CONTINUOUS, ub=np.inf, lb=-np.inf)
    x_list = x.tolist()

    obj = (x_list[0] - y[0]) ** 2 + (x_list[1] - y[1]) ** 2 + (x_list[2] - y[2]) ** 2
    qp.setObjective(obj, GRB.MINIMIZE)

    qp.addConstr(A @ x - b <= 0, "zono1")
    qp.params.OutputFlag = 0
    qp.optimize()

    result = np.zeros(n_dims)
    for i, v in enumerate(qp.getVars()):
        result[i] = v.X
    
    return obj.getValue(), result[:n_dims]



def read_params():
    parser = argparse.ArgumentParser(description="SDF RTD Dataset Generation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--n_links', type=int, default=7)
    parser.add_argument('--n_dims', type=int, default=3)
    parser.add_argument('--n_obs', type=int, default=8)
    parser.add_argument('--n_data', type=int, default=64)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--output_path', type=str, default='.pkl')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--signed', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    from rdf.environments.robots.load_robot import load_sinlge_robot_arm_params
    params = read_params()
    N_joints = params.n_links
    n_dims = params.n_dims

    seed = params.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_filename = f"{seed}seed{params.output_path}"

    link_params, _ = load_sinlge_robot_arm_params('Kinova3')
    link_zonos = [l.to_polyZonotope() for l in link_params['link_zonos']]

    
    generate_dataset(n_links=link_params['n_joints'], n_dims=n_dims, N=params.n_data, 
        num_obstacles_each_initial_condition=params.n_obs, output_dataset_filename=output_filename, 
        verbose=params.verbose, plot=params.plot, save=params.save, signed=params.signed, 
        params=link_params, link_zonos=link_zonos)
    '''
    generate_dataset(n_links=link_params['n_joints'], n_dims=3, N=8, 
        num_obstacles_each_initial_condition=8, output_dataset_filename=output_filename, 
        verbose=1, plot=True, save=False, signed=True, 
        params=link_params, link_zonos=link_zonos)'''
