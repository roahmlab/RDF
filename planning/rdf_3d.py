import time
import torch
import numpy as np
from torch.autograd import grad
import sys
import cyipopt

sys.path.append('..')
from training.model import MLP

T_PLAN, T_FULL = 0.5, 1.0

class RdfNLP_3D():
    def __init__(self, qpos, qvel, qgoal, obstacle, model, device, vel_lim, pos_lim, actual_pos_lim, n_pos_lim, lim_flag, g_ka=torch.pi/24, n_links=7, dtype=torch.float) -> None:
        self.device = device
        self.model = model
        self.dtype = dtype
        self.n_links = n_links
        self.prev_x = np.zeros(n_links) * np.nan
        self.num_obstacles = obstacle.shape[0]
        self.num_obstacle_constarints = 1
        self.num_constraints = self.num_obstacle_constarints + \
            2 * self.n_links + 6 * n_pos_lim

        self.vel_lim = vel_lim
        self.pos_lim = pos_lim
        self.actual_pos_lim = actual_pos_lim
        self.lim_flag = lim_flag

        self.qpos = qpos.to(dtype=self.dtype, device='cpu')
        self.qvel = qvel.to(dtype=self.dtype, device='cpu')
        self.qgoal = qgoal.to(dtype=self.dtype, device='cpu')
        self.obstacle = obstacle.to(dtype=self.dtype, device='cpu')

        self.batched_qpos = (qpos / pos_lim[:,0]).to(device).float().repeat(
            self.num_obstacles, 1).view(self.num_obstacles, -1)
        self.batched_qvel = (qvel / vel_lim).to(device).float().repeat(
            self.num_obstacles, 1).view(self.num_obstacles, -1)
        self.obstacle = obstacle.to(self.device).float().view(
            self.num_obstacles, -1)[:, :obstacle.shape[1] // 2]
        self.g_ka = g_ka

    def wrap_cont_joint_to_pi(self, phases):
        phases_new = torch.clone(phases)
        phases_new[~self.lim_flag] = (
            phases[~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
        return phases_new

    def objective(self, x):
        qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
        return torch.sum(self.wrap_cont_joint_to_pi(qplan-self.qgoal) ** 2)

    def gradient(self, x):
        qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
        qplan_grad = 0.5*self.g_ka*T_PLAN**2
        return (2*qplan_grad*self.wrap_cont_joint_to_pi(qplan-self.qgoal)).numpy()

    def constraints(self, x):
        self.compute_constraints_jacobian(x)
        return self.cons

    def jacobian(self, x):
        self.compute_constraints_jacobian(x)
        return self.jac

    def compute_constraints_jacobian(self, x):
        if (self.prev_x != x).any():
            self.cons = torch.zeros((self.num_constraints))
            self.jac = torch.zeros((self.num_constraints, self.n_links))
            ka = torch.tensor(x, dtype=self.dtype)

            # position and velocity constraints
            # time to optimum of first half traj.
            t_peak_optimum = -self.qvel/(self.g_ka*ka)
            qpos_peak_optimum = (t_peak_optimum > 0)*(t_peak_optimum < T_PLAN)*(
                self.qpos+self.qvel*t_peak_optimum+0.5*(self.g_ka*ka)*t_peak_optimum**2).nan_to_num()
            grad_qpos_peak_optimum = torch.diag((t_peak_optimum > 0)*(
                t_peak_optimum < T_PLAN)*(0.5*self.qvel**2/(self.g_ka*ka**2)).nan_to_num())

            qpos_peak = self.qpos + self.qvel * T_PLAN + \
                0.5 * (self.g_ka * ka) * T_PLAN**2
            grad_qpos_peak = 0.5 * self.g_ka * T_PLAN**2 * \
                torch.eye(self.n_links, dtype=self.dtype)
            qvel_peak = self.qvel + self.g_ka * ka * T_PLAN
            grad_qvel_peak = self.g_ka * T_PLAN * \
                torch.eye(self.n_links, dtype=self.dtype)

            bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
            qpos_brake = qpos_peak + qvel_peak * \
                (T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
            # can be also, qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL+T_PLAN) + 0.5 * (self.g_ka * ka[0]) * T_PLAN * T_FULL
            grad_qpos_brake = 0.5 * self.g_ka * T_PLAN * T_FULL * \
                torch.eye(
                    self.n_links, dtype=self.dtype) 

            qpos_possible_max_min = torch.vstack(
                (qpos_peak_optimum, qpos_peak, qpos_brake))[:, self.lim_flag]
            qpos_ub = (qpos_possible_max_min -
                       self.actual_pos_lim[:, 0]).flatten()
            qpos_lb = (
                self.actual_pos_lim[:, 1] - qpos_possible_max_min).flatten()

            grad_qpos_ub = torch.vstack(
                (grad_qpos_peak_optimum[self.lim_flag], grad_qpos_peak[self.lim_flag], grad_qpos_brake[self.lim_flag]))
            grad_qpos_lb = - grad_qpos_ub

            self.cons[self.num_obstacle_constarints:] = torch.hstack(
                (qvel_peak-self.vel_lim, -self.vel_lim-qvel_peak, qpos_ub, qpos_lb))
            self.jac[self.num_obstacle_constarints:] = torch.vstack(
                (grad_qvel_peak, -grad_qvel_peak, grad_qpos_ub, grad_qpos_lb))

            # obstalce constraints
            k = torch.tensor(x).to(self.device, torch.float).repeat(self.num_obstacles, 1).requires_grad_()
            
            distances = self.model(qpos=self.batched_qpos, qvel=self.batched_qvel, obstacle=self.obstacle, k=k) # distances shape: (n_obs, n_links)

            min_distance_index = distances.argmin()
            min_distance_obs_index, min_distance_joint_index = min_distance_index // self.n_links, min_distance_index % self.n_links
            self.cons[0] = distances[min_distance_obs_index, min_distance_joint_index].detach().item()
            self.jac[0] = grad(outputs=distances[min_distance_obs_index, min_distance_joint_index], inputs=k)[0][min_distance_obs_index].cpu()
            self.prev_x = np.copy(x)

            self.cons = self.cons.numpy()
            self.jac = self.jac.numpy()
            

class RDF_3D_Planner():
    def __init__(self, model_path, device, n_links=7, n_dims=3):
        self.device = device
        self.dtype = torch.float32
        self.n_links = n_links
        self.n_dims = n_dims
        self.model = MLP(n_links=self.n_links, n_dims=self.n_dims,
                         num_hidden_layers=8, hidden_size=1024, fix_size=True)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def wrap_env(self, env):
        assert env.dimension == 3
        self.dimension = 3
        self.n_links = env.n_links
        self.n_obs = env.n_obs

        self.joint_axes = env.joint_axes.to(
            dtype=self.dtype, device=self.device)
        self.vel_lim = env.vel_lim.cpu()
        self.pos_lim = env.pos_lim.cpu()
        self.actual_pos_lim = env.pos_lim[env.lim_flag].cpu()
        self.n_pos_lim = int(env.lim_flag.sum().cpu())
        self.lim_flag = env.lim_flag.cpu()

    def plan(self, env, k0, buffer_size=0.0, time_limit=5.0):
        self.wrap_env(env)
        qpos, qvel, qgoal, obs_zonos = env.qpos, env.qvel, env.qgoal, env.obs_zonos
        obstacles = []
        for obs in obs_zonos:
            obstacles.append(torch.hstack(
                (obs.center[:self.n_dims], 0.1*torch.ones(self.n_dims))).unsqueeze(0))
        obstacles = torch.cat(obstacles, 0)

        M_obs = 1
        M = M_obs + 2*self.n_links + 6 * self.n_pos_lim
        nlp_obj = RdfNLP_3D(qpos, qvel, qgoal, obstacles, self.model, self.device, vel_lim=self.vel_lim,
                         pos_lim=self.pos_lim, actual_pos_lim=self.actual_pos_lim, n_pos_lim=self.n_pos_lim, lim_flag=self.lim_flag)
        nlp = cyipopt.Problem(
            n=self.n_links,
            m=M,
            problem_obj=nlp_obj,
            lb=[-1]*self.n_links,
            ub=[1]*self.n_links,
            cl=[buffer_size]*M_obs+[-1e20]*(M-M_obs),
            cu=[1e20]*M_obs+[-1e-6]*(M-M_obs), 
        )
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        nlp.add_option('max_wall_time', time_limit)
        k_opt, self.info = nlp.solve(k0.cpu().numpy())

        return nlp_obj.g_ka * torch.tensor(k_opt, dtype=self.dtype, device=self.device), self.info['status']