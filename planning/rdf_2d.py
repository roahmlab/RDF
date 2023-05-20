import time
import torch
import numpy as np
from torch.autograd import grad
import sys
import cyipopt

sys.path.append('..')
from training.model import MLP

T_PLAN, T_FULL = 0.5, 1.0
def wrap_to_pi(phases):
    return (phases + np.pi) % (2 * np.pi) - np.pi

class RdfNLP_2D():
    def __init__(self, qpos, qvel, qgoal, obstacle, model, device, g_ka=torch.pi/24, n_links=2) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.n_links = n_links
        self.prev_x = np.zeros(n_links) * np.nan
        self.num_obstacles = obstacle.shape[0]
        self.num_obstacle_constarints = 1
        self.num_constraints = self.num_obstacle_constarints + 2*self.n_links
        
        self.qpos = qpos.cpu().numpy()
        self.qvel = qvel.cpu().numpy()
        self.qgoal = qgoal.cpu().numpy()
        self.obstacle = obstacle.float()

        self.batched_qpos = qpos.to(device).float().repeat(self.num_obstacles,1).view(self.num_obstacles, -1) / torch.pi
        self.batched_qvel = qvel.to(device).float().repeat(self.num_obstacles,1).view(self.num_obstacles, -1) / (torch.pi / 2)
        self.obstacle = obstacle.to(self.device).float().view(self.num_obstacles, -1)[:, :obstacle.shape[1] // 2]
        self.g_ka = g_ka

    def objective(self,x):
        qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
        return np.sum(wrap_to_pi(qplan-self.qgoal) ** 2)

    def gradient(self,x):
        qplan = self.qpos + self.qvel*T_PLAN + 0.5*self.g_ka*x*T_PLAN**2
        return self.g_ka* T_PLAN ** 2 * wrap_to_pi(qplan - self.qgoal)
    
    def constraints(self, x): 
        self.compute_constraints_jacobian(x)
        return self.cons

    def jacobian(self, x):
        self.compute_constraints_jacobian(x)
        return self.jac

    def compute_constraints_jacobian(self, x):
        if (self.prev_x != x).any():
            self.cons = np.zeros((self.num_constraints))
            self.jac = np.zeros((self.num_constraints, self.n_links))
            
            # velocity constarints
            q_vel_peak = self.qvel + self.g_ka * x * T_PLAN
            grad_q_vel_peak = self.g_ka * T_PLAN * np.eye(self.n_links)
            self.cons[self.num_obstacle_constarints:] = np.hstack((q_vel_peak-np.pi/2, -np.pi/2-q_vel_peak))
            self.jac[self.num_obstacle_constarints:] = np.vstack((grad_q_vel_peak, -grad_q_vel_peak))

            # obstalce constraints
            k = torch.tensor(x).to(self.device, torch.float).repeat(self.num_obstacles).requires_grad_()
            k = k.view(self.num_obstacles,-1)
            distances = self.model(qpos=self.batched_qpos, qvel=self.batched_qvel, obstacle=self.obstacle, k=k) # distances shape: (n_obs, n_links)
            num_obs, num_predictions = distances.shape

            min_distance_index = distances.argmin()
            # print(f"predicted distances:{distances} for k={k[0]}")                
            self.cons[:self.num_obstacle_constarints] = distances.view(-1)[min_distance_index].detach().cpu().numpy()
            min_distance_obs_index, min_distance_joint_index = min_distance_index.item() // num_predictions, min_distance_index.item() % num_predictions
            self.jac[:self.num_obstacle_constarints] = grad( outputs=distances[min_distance_obs_index, min_distance_joint_index], inputs=k, retain_graph=True)[0][min_distance_obs_index].cpu().numpy()

            self.prev_x = np.copy(x)
            
            
class RDF_2D_Planner():
    def __init__(self, model_path, device, n_links=2, n_dims=2):
        self.device = device
        self.dtype = torch.get_default_dtype()
        self.n_links = n_links
        self.n_dims = n_dims
        self.model = MLP(n_links=self.n_links, n_dims=self.n_dims, num_hidden_layers=8, hidden_size=1024, fix_size=True)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def env_wrap(self,env):
        qpos = env.qpos 
        qvel = env.qvel 
        qgoal = env.qgoal 
        obstacles = []
        for obs in env.obs_zonos:
            obstacles.append(torch.hstack((obs.center[:self.n_dims],0.1/(self.n_links * 2)*torch.ones(self.n_dims))).unsqueeze(0))
        obstacles = torch.cat(obstacles,0)
        return qpos, qvel, qgoal, obstacles

    def plan(self,env,k0,buffer_size=0.0,time_limit=5.0):
        qpos, qvel, qgoal, obstacles = self.env_wrap(env)

        M_obs = 1
        M = M_obs + 2*self.n_links
        nlp_obj = RdfNLP_2D(qpos,qvel,qgoal,obstacles,self.model,self.device,n_links=self.n_links)
        nlp = cyipopt.Problem(
            n = self.n_links,
            m = M,
            problem_obj=nlp_obj,
            lb = [-1]*self.n_links,
            ub = [1]*self.n_links,
            cl = [buffer_size]*M_obs+[-1e20]*2*self.n_links,
            cu = [1e20]*M_obs+[-1e-6]*2*self.n_links,
        )
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-3)
        nlp.add_option('max_wall_time', time_limit)
        
        k_opt, self.info = nlp.solve(k0.cpu().numpy())

        return nlp_obj.g_ka * torch.tensor(k_opt,dtype=self.dtype,device=self.device), self.info['status']