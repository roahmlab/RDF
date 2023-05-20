"""
3D Arm Environment
Author: Yongseok Kwon
"""

import torch 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys
sys.path.append('..')
from reachability.conSet import zonotope, batchZonotope
from environments.robots.load_robot import load_sinlge_robot_arm_params

def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1

class Arm_3D:
    def __init__(self,
            robot='Kinova3', # robot model
            n_obs=1, # number of obstacles
            obs_size_max = [0.1,0.1,0.1], # maximum size of randomized obstacles in xyz
            obs_size_min = [0.1,0.1,0.1], # minimum size of randomized obstacle in xyz
            T_len=50, # number of discritization of time interval
            interpolate = True, # flag for interpolation
            check_collision = True, # flag for whehter check collision
            check_collision_FO = False, # flag for whether check collision for FO rendering
            check_joint_limit = True,
            collision_threshold = 1e-6, # collision threshold
            goal_threshold = 0.1, # goal threshold
            hyp_step = 0.3,
            hyp_dist_to_goal = 0.3,
            hyp_effort = 0.1, # hyperpara
            hyp_success = 50,
            hyp_collision = 50,
            hyp_action_adjust = 0.3,
            hyp_fail_safe = 1,
            hyp_stuck =50,
            hyp_timeout = 0,
            stuck_threshold = None,
            reward_shaping=True,
            gamma = 0.99, # discount factor on reward
            max_episode_steps = 300,
            FO_render_level = 2, # 0: no rendering, 1: a single geom, 2: seperate geoms for each links, 3: seperate geoms for each links and timesteps
            FO_render_freq = 10,
            ticks = False,
            scale = 1,
            max_combs = 200,
            dtype= torch.float,
            device = torch.device('cpu')
            ):
        self.max_combs = max_combs
        self.dtype = dtype
        self.device = device
        self.generate_combinations_upto()
        self.dimension = 3
        self.n_obs = n_obs
        self.obs_size_sampler = torch.distributions.Uniform(torch.tensor(obs_size_min,dtype=dtype,device=device),torch.tensor(obs_size_max,dtype=dtype,device=device),validate_args=False)
        self.scale = scale
        self.num_envs = 1

        #### load
        params, _ = load_sinlge_robot_arm_params(robot)
        self.dof = self.n_links = params['n_joints']
        self.joint_id = torch.arange(self.n_links,dtype=int,device=device)
        self.__link_zonos = [(self.scale*params['link_zonos'][j]).to(dtype=dtype,device=device) for j in range(self.n_links)] # NOTE: zonotope, should it be poly zonotope?
        self.link_polyhedron = [zono.polyhedron_patch() for zono in self.__link_zonos]
        self.link_zonos = [self.__link_zonos[j].to_polyZonotope() for j in range(self.n_links)]
        self.P0 = [self.scale*P.to(dtype=dtype,device=device) for P in params['P']]
        self.R0 = [R.to(dtype=dtype,device=device) for R in params['R']]
        self.joint_axes = torch.vstack(params['joint_axes']).to(dtype=dtype,device=device)
        w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]],dtype=dtype,device=device)
        self.rot_skew_sym = (w@self.joint_axes.T).transpose(0,-1)
        
        self.pos_lim = torch.tensor(params['pos_lim'],dtype=dtype,device=device)
        self.vel_lim = torch.tensor(params['vel_lim'],dtype=dtype,device=device)
        self.tor_lim = torch.tensor(params['tor_lim'],dtype=dtype,device=device)
        self.lim_flag = torch.tensor(params['lim_flag'],dtype=bool,device=device)

        self._pos_lim = self.pos_lim.clone()
        self._vel_lim = self.vel_lim.clone()
        self._tor_lim = self.tor_lim.clone()
        self._lim_flag = self.lim_flag.clone()
        self._actual_pos_lim = self._pos_lim[self._lim_flag]

        self.pos_sampler = torch.distributions.Uniform(self.pos_lim[:,1],self.pos_lim[:,0])
        self.full_radius = self.scale*0.8
        #self.full_radius = sum([(abs(self.P0[j])).max() for j in range(self.n_links)])        
        #### load

        self.fig_scale = 1
        self.interpolate = interpolate
        self.PI = torch.tensor(torch.pi,dtype=dtype,device=device)

        if interpolate:
            if T_len % 2 != 0:
                self.T_len = T_len + 1
            else: 
                self.T_len = T_len
            t_traj = torch.linspace(0,T_FULL,T_len+1,dtype=dtype,device=device)
            self.t_to_peak = t_traj[:int(T_PLAN/T_FULL*T_len)+1]
            self.t_to_brake = t_traj[int(T_PLAN/T_FULL*T_len):] - T_PLAN
        
        self.obs_buffer_length = torch.tensor([0.001,0.001],dtype=dtype,device=device)
        self.check_collision = check_collision
        self.check_collision_FO = check_collision_FO
        self.check_joint_limit = check_joint_limit
        self.collision_threshold = collision_threshold
        
        self.goal_threshold = goal_threshold
        self.hyp_step = hyp_step
        self.hyp_dist_to_goal = hyp_dist_to_goal
        self.hyp_effort = hyp_effort
        self.hyp_success = hyp_success
        self.hyp_collision = hyp_collision
        self.hyp_action_adjust = hyp_action_adjust
        self.hyp_fail_safe = hyp_fail_safe
        self.hyp_stuck = hyp_stuck
        self.hyp_timeout = hyp_timeout
        if stuck_threshold is None:
            self.stuck_threshold = max_episode_steps
        else:
            self.stuck_threshold = stuck_threshold
        self.reward_shaping = reward_shaping
        self.gamma = gamma

        self.fig = None
        self.render_flag = True
        assert FO_render_level<4
        self.FO_render_level = FO_render_level
        self.FO_render_freq = FO_render_freq
        self.ticks = ticks

    
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._frame_steps = 0     
        self.dtype = dtype
        self.device = device
        self.reset()

    def wrap_cont_joint_to_pi(self,phases,internal):
        phases_new = torch.clone(phases)
        phases_new[~self.lim_flag] = (phases[~self.lim_flag] + torch.pi) % (2 * torch.pi) - torch.pi
        return phases_new

    def generate_combinations_upto(self):
        self.combs = [torch.combinations(torch.arange(i,device=self.device),2) for i in range(self.max_combs+1)]
 
    def reset(self):
        self.qpos = self.pos_sampler.sample()
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = torch.zeros(self.n_links,dtype=self.dtype,device=self.device)
        self.qvel_init = torch.clone(self.qvel)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = self.pos_sampler.sample()
        if self.interpolate:
            T_len_to_brake = int((1-T_PLAN/T_FULL)*self.T_len)+1
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(T_len_to_brake,1)
            self.qvel_to_brake = torch.zeros(T_len_to_brake,self.n_links,dtype=self.dtype,device=self.device)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_links,dtype=self.dtype,device=self.device)            
        
        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)       
        Rg, Pg = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[j]
            Rg = Rg@self.R0[j]@R_qg[j]
            link_init.append(Ri@self.__link_zonos[j]+Pi)
            link_goal.append(Rg@self.__link_zonos[j]+Pg)

        obs_size = self.scale*self.obs_size_sampler.sample([self.n_obs])

        for o in range(self.n_obs):
            while True:
                obs_pos = self.scale*(torch.rand(3,dtype=self.dtype,device=self.device)*2*0.8-0.8)
                
                # NOTE
                #rho, th, psi 
                obs = zonotope(torch.vstack((obs_pos,torch.diag(obs_size[o]))))
                
                
                safe_flag = True
                for j in range(self.n_links):
                    buff = link_init[j]-obs
                    _,b = buff.polytope(self.combs)
                    if min(b) > -1e-5:
                        safe_flag = False
                        break
                    buff = link_goal[j]-obs
                    _,b = buff.polytope(self.combs)
                    if min(b) > -1e-5:
                        safe_flag = False
                        break

                if safe_flag:
                    self.obs_zonos.append(obs)
                    break

        self.fail_safe_count = 0
        if self.render_flag == False:
            self.obs_patches.remove()
            self.link_goal_patches.remove()
            self.FO_patches.remove()
            self.link_patches.remove()
        self.render_flag = True
        self.done = False
        self.collision = False

        self._elapsed_steps = 0   
        self.reward_com = 0

        return self.get_observations()

    def set_initial(self,qpos,qvel,qgoal,obs_pos,obs_size=None):
        if obs_size is None:
            obs_size = [self.scale*torch.tensor([.1,.1,.1],dtype=self.dtype,device=self.device) for _ in range(self.n_obs)]
        self.qpos = qpos.to(dtype=self.dtype,device=self.device)
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = qvel.to(dtype=self.dtype,device=self.device)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = qgoal.to(dtype=self.dtype,device=self.device)   
        if self.interpolate:
            T_len_to_peak = int((1-T_PLAN/T_FULL)*self.T_len)+1            
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(T_len_to_peak,1)
            self.qvel_to_brake = torch.zeros(T_len_to_peak,self.n_links,dtype=self.dtype,device=self.device)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_links,dtype=self.dtype,device=self.device)   
        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)       
        Rg, Pg = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[j]
            Rg = Rg@self.R0[j]@R_qg[j]
            link_init.append(Ri@self.__link_zonos[j]+Pi)
            link_goal.append(Rg@self.__link_zonos[j]+Pg)

        for pos,size in zip(obs_pos,obs_size):
            obs = zonotope(torch.vstack((pos.to(dtype=self.dtype,device=self.device),torch.diag(size).to(dtype=self.dtype,device=self.device))))
            for j in range(self.n_links):
                buff = link_init[j]-obs
                _,b = buff.polytope(self.combs)
                if min(b) > -1e-5:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                _,b = buff.polytope(self.combs)
                if min(b) > -1e-5:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'    
            self.obs_zonos.append(obs)
        self.fail_safe_count = 0
        if self.render_flag == False:
            self.obs_patches.remove()
            self.link_goal_patches.remove()
            self.FO_patches.remove()
            self.link_patches.remove()
        self.render_flag = True
        self.done = False
        self.collision = False

        self._elapsed_steps = 0
        self.reward_com = 0

        return self.get_observations()

    def step(self,ka,flag=-1):
        # -1: direct pass, 0: safe plan from armtd pass, 1: fail-safe plan from armtd pass
        ka = ka.detach()
        self.ka = ka.clamp((-torch.pi-self.qvel)/T_PLAN,(torch.pi-self.qvel)/T_PLAN) # velocity clamp
        self.joint_limit_check()

        self.step_flag = int(flag)
        self.safe = (flag <= 0) or self.exceed_joint_limit

        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        if self.interpolate:
            if self.safe:
                self.fail_safe_count = 0
                
                # to peak
                self.qpos_to_peak = wrap_to_pi(self.qpos + torch.outer(self.t_to_peak,self.qvel) + .5*torch.outer(self.t_to_peak**2,self.ka))
                self.qvel_to_peak = self.qvel + torch.outer(self.t_to_peak,self.ka)
                self.qpos = self.qpos_to_peak[-1]
                self.qvel = self.qvel_to_peak[-1]
                #to stop
                bracking_accel = (0 - self.qvel)/(T_FULL - T_PLAN)
                self.qpos_to_brake = wrap_to_pi(self.qpos + torch.outer(self.t_to_brake,self.qvel) + .5*torch.outer(self.t_to_brake**2,bracking_accel))
                self.qvel_to_brake = self.qvel + torch.outer(self.t_to_brake,bracking_accel)
                self.collision = self.collision_check(self.qpos_to_peak[1:])
            else:
                self.fail_safe_count +=1
                self.qpos_to_peak = torch.clone(self.qpos_to_brake)
                self.qvel_to_peak = torch.clone(self.qvel_to_brake)
                self.qpos = self.qpos_to_peak[-1]
                self.qvel = self.qvel_to_peak[-1]
                T_len_to_brake = int((1-T_PLAN/T_FULL)*self.T_len)+1
                self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(T_len_to_brake,1)
                self.qvel_to_brake = torch.zeros(T_len_to_brake,self.n_links,dtype=self.dtype,device=self.device)
                self.collision = self.collision_check(self.qpos_to_peak[1:])
        else:
            if self.safe:
                self.fail_safe_count = 0
                self.qpos = wrap_to_pi(self.qpos + self.qvel*T_PLAN + 0.5*self.ka*T_PLAN**2)
                self.qvel += self.ka*T_PLAN
                bracking_accel = (0 - self.qvel)/(T_FULL - T_PLAN)
                self.qpos_brake = wrap_to_pi(self.qpos + self.qvel*(T_FULL-T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2)
                self.qvel_brake = torch.zeros(self.n_links,dtype=self.dtype,device=self.device)
                self.collision = self.collision_check(self.qpos)
            else:
                self.fail_safe_count +=1
                self.qpos = torch.clone(self.qpos_brake)
                self.qvel = torch.clone(self.qvel_brake) 
                self.collision = self.collision_check(self.qpos)

        self._elapsed_steps += 1
        self.reward = self.get_reward(ka) # NOTE: should it be ka or self.ka ??
        self.reward_com *= self.gamma
        self.reward_com += self.reward
        observations = self.get_observations()
        info = self.get_info()
        return observations, self.reward, self.done, info
    
    def get_info(self):
        info ={'is_success':self.success,
                'collision':self.collision,
                'safe_flag':self.safe,
                'step_flag':self.step_flag,
                'stuck':self.stuck,
                }
        if self.collision:
            collision_info = {
                'qpos_collision':self.qpos_collision,
                'qpos_init':self.qpos_int,
                'qvel_int':torch.zeros(self.n_links,dtype=self.dtype,device=self.device),
                'obs_pos':[self.obs_zonos[o].center for o in range(self.n_obs)],
                'qgoal':self.qgoal
            }
            info['collision_info'] = collision_info

        if self.done:
            info["terminal_observation"] = self.get_observations()

        info["TimeLimit.truncated"] = self.timeout
        info['episode'] = {"r":self.reward_com,"l":self._elapsed_steps}
        return info

    def get_observations(self):
        observation = {'qpos':self.qpos,'qvel':self.qvel,'qgoal':self.qgoal}
        if self.n_obs > 0:
            observation['obstacle_pos']= torch.vstack([self.obs_zonos[o].center for o in range(self.n_obs)])
            observation['obstacle_size'] = torch.vstack([torch.diag(self.obs_zonos[o].generators) for o in range(self.n_obs)])
        return observation
    #ext_kwargs
    def get_reward(self, action, qpos=None, qgoal=None, collision=None, step_flag = None, safe=None, stuck=None, timeout=None):
        # Get the position and goal then calculate distance to goal
        if qpos is None:
            internal = True
            # deliver termination variable
            collision = self.collision 
            action_adjusted = self.step_flag == 0
            safe = self.safe
            stuck = self.stuck = self.fail_safe_count >= self.stuck_threshold
            goal_dist = torch.linalg.norm(self.wrap_cont_joint_to_pi(self.qpos-self.qgoal,internal=internal),dim=-1)
            success = self.success = bool(goal_dist < self.goal_threshold) 
            # compute done and timeout
            done = self.success or self.collision or self.stuck
            timeout = self.timeout = (self._elapsed_steps >= self._max_episode_steps) and (~done)
            self.done = done or self.timeout
            
        else: 
            internal = False
            action_adjusted = step_flag == 0
            goal_dist = torch.linalg.norm(self.wrap_cont_joint_to_pi(qpos-qgoal,internal=internal),dim=-1)
            success = bool(goal_dist < self.goal_threshold) and (~collision) 
                
        reward = 0.0
        # Reward shaing with dense reward
        if self.reward_shaping:
            # Step penalty 
            reward -= self.hyp_step
            # Goal-distance penalty
            reward -= self.hyp_dist_to_goal * goal_dist
            # Effort penalty
            reward -= self.hyp_effort * torch.linalg.norm(action)
        # Success reward 
        reward += self.hyp_success * success
        # Collision penalty
        reward -= self.hyp_collision * collision
        # Action adjustment peanlty 
        reward -= self.hyp_action_adjust * action_adjusted
        # Fail-safe penalty
        reward -= self.hyp_fail_safe * (1 - safe)
        # Stuck penalty
        reward -= self.hyp_stuck * stuck
        # Timeout penalty 
        reward -= self.hyp_timeout * timeout

        return float(reward)   

    def success_check(self, qpos=None, qgoal=None):
        if qpos is None:
            goal_dist = torch.linalg.norm(self.wrap_cont_joint_to_pi(self.qpos-self.qgoal,internal=True)) 
        else:
            goal_dist = torch.linalg.norm(self.wrap_cont_joint_to_pi(qpos-qgoal,internal=False))
        return bool(goal_dist < self.goal_threshold)

    def joint_limit_check(self):
        if self.check_joint_limit:
            t_peak_optimum = -self.qvel/self.ka # time to optimum of first half traj.
            qpos_peak_optimum = (t_peak_optimum>0)*(t_peak_optimum<T_PLAN)*(self.qpos+self.qvel*t_peak_optimum+0.5*self.ka*t_peak_optimum**2).nan_to_num()
            qpos_peak = self.qpos + self.qvel * T_PLAN + 0.5 * self.ka * T_PLAN**2
            qvel_peak = self.qvel + self.ka * T_PLAN

            bracking_accel = (0 - qvel_peak)/(T_FULL - T_PLAN)
            qpos_brake = qpos_peak + qvel_peak*(T_FULL - T_PLAN) + 0.5*bracking_accel*(T_FULL-T_PLAN)**2
            # can be also, qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL+T_PLAN) + 0.5 * self.ka * T_PLAN * T_FULL
            qpos_possible_max_min = torch.cat((qpos_peak_optimum.unsqueeze(-2),qpos_peak.unsqueeze(-2),qpos_brake.unsqueeze(-2)),-2)[:,self._lim_flag]

            qpos_ub = (qpos_possible_max_min - self._actual_pos_lim[:,0])
            qpos_lb = (self._actual_pos_lim[:,1] - qpos_possible_max_min)
            self.exceed_joint_limit = bool((abs(qvel_peak)>self._vel_lim).any(-1) + (qpos_ub>0).any() + (qpos_lb>0).any())

        self.exceed_joint_limit = False


    def collision_check(self,qs):

        if self.check_collision:
            
            R_q = self.rot(qs)
            if len(R_q.shape) == 4:
                time_steps = len(R_q)
                R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[:,j]
                    link = batchZonotope(self.link_zonos[j].Z.unsqueeze(0).repeat(time_steps,1,1))
                    link = R@link+P
                    for o in range(self.n_obs):
                        buff = link - self.obs_zonos[o]
                        _,b = buff.polytope(self.combs)
                        unsafe = b.min(dim=-1)[0]>1e-6
                        if any(unsafe):
                            self.qpos_collision = qs[unsafe]
                            return True

            else:
                time_steps = 1
                R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[j]
                    link = R@self.__link_zonos[j]+P
                    for o in range(self.n_obs):
                        buff = link - self.obs_zonos[o]
                        _,b = buff.polytope(self.combs)
                        if min(b) > 1e-6:
                            self.qpos_collision = qs
                            return True
  
        return False

    def render(self,FO_link=None,show=True,dpi=None,save_kwargs=None):
        if self.render_flag:
            if self.fig is None:
                if show:
                    plt.ion()
                #self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8],dpi=dpi)
                #self.ax = a3.Axes3D(self.fig)

                self.fig = plt.figure()
                self.ax = plt.axes(projection='3d')
                if not self.ticks:
                    plt.tick_params(which='both',bottom=False,top=False,left=False,right=False, labelbottom=False, labelleft=False)
                if save_kwargs is not None:
                    os.makedirs(save_kwargs['save_path'],exist_ok=True) 
                    self.ax.set_title(save_kwargs['text'],fontsize=10,loc='right')
            
            self.render_flag = False
            self.FO_patches = self.ax.add_collection3d(Poly3DCollection([]))
            self.link_patches = self.ax.add_collection3d(Poly3DCollection([]))

            #  Collect patches of polyhedron representation of obstacle
            obs_patches = []
            for o in range(self.n_obs):
                obs_patches.extend(self.obs_zonos[o].polyhedron_patch())
            self.obs_patches = self.ax.add_collection3d(Poly3DCollection(obs_patches,edgecolor='red',facecolor='red',alpha=0.2,linewidths=0.2))

            # Collect patches of polyhedron representation of link in goal configuration
            link_goal_patches = []
            R_q = self.rot(self.qgoal)
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)            
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[j]
                link_goal_patches.extend((self.link_polyhedron[j]@R.T+P).cpu())
            self.link_goal_patches = self.ax.add_collection3d(Poly3DCollection(link_goal_patches,edgecolor='gray',facecolor='gray',alpha=0.15,linewidths=0.5))

        # Collect patches of polyhedron representation of forward reachable set
        if FO_link is not None: 
            FO_patches = []
            if self.fail_safe_count == 0:
                g_ka = self.PI/24
                self.FO_patches.remove()
                for j in range(self.n_links): 
                    FO_link_slc = FO_link[j].slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1)).reduce(4)
                    #FO_link_slc = FO_link[j].to_batchZonotope().reduce(1)
                    for t in range(100):
                        if t % self.FO_render_freq == 0:
                            FO_patch = FO_link_slc[t].polyhedron_patch().detach()
                            FO_patches.extend(FO_patch)
                self.FO_patches = self.ax.add_collection3d(Poly3DCollection(FO_patches,alpha=0.03,edgecolor='green',facecolor='green',linewidths=0.2))

        if self.interpolate:
            timesteps = int(T_PLAN/T_FULL*self.T_len)
            if show and save_kwargs is None:
                plot_freq = 1
                R_q = self.rot(self.qpos_to_peak[1:]).unsqueeze(-3)
            elif show:
                plot_freq = timesteps//save_kwargs['frame_rate']
                R_q = self.rot(self.qpos_to_peak[1:]).unsqueeze(-3) 
            else:
                plot_freq = 1
                t_idx = torch.arange(timesteps+1,device=self.device)%(timesteps//save_kwargs['frame_rate'] ) == 1
                R_q = self.rot(self.qpos_to_peak[t_idx]).unsqueeze(-3)
                timesteps = len(R_q)

            link_trace_polyhedron =  torch.zeros(timesteps,12*self.n_links,3,3,dtype=self.dtype,device='cpu')
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,3,dtype=self.dtype,device=self.device)
            for j in range(self.n_links):
                P = R@self.P0[j] + P
                R = R@self.R0[j]@R_q[:,j]
                #import pdb;pdb.set_trace()
                link_trace_polyhedron[:,12*j:12*(j+1)] = (self.link_polyhedron[j]@R.transpose(-1,-2)+P.unsqueeze(-3)).cpu()
            
            for t in range(timesteps):                
                self.link_patches.remove()
                self.link_patches = self.ax.add_collection3d(Poly3DCollection(list(link_trace_polyhedron[t]), edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5))
                self.ax.set_xlim([-self.full_radius,self.full_radius])
                self.ax.set_ylim([-self.full_radius,self.full_radius])
                self.ax.set_zlim([-self.full_radius,self.full_radius])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                if save_kwargs is not None and t%plot_freq == 0:
                    filename = "/frame_"+"{0:04d}".format(self._frame_steps)                    
                    self.fig.savefig(save_kwargs['save_path']+filename,dpi=save_kwargs['dpi'])
                    self._frame_steps+=1

        else:
            R_q = self.rot()
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)
            link_trace_patches = []
            self.link_patches.remove()         
            for j in range(self.n_links):
                P = R@self.P0[j] + P 
                R = R@self.R0[j]@R_q[j]
                link_trace_patches.extend((self.link_polyhedron[j]@R.T+P).cpu())
            self.link_patches = self.ax.add_collection3d(Poly3DCollection(link_trace_patches, edgecolor='blue',facecolor='blue',alpha=0.2,linewidths=0.5))
            self.ax.set_xlim([-self.full_radius,self.full_radius])
            self.ax.set_ylim([-self.full_radius,self.full_radius])
            self.ax.set_zlim([-self.full_radius,self.full_radius])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if save_kwargs is not None:
                filename = "/frame_"+"{0:04d}".format(self._frame_steps)
                self.fig.savefig(save_kwargs['save_path']+filename,dpi=save_kwargs['dpi'])
                self._frame_steps+=1

    def close(self):
        if self.render_flag == False:
            self.obs_patches.remove()
            self.link_goal_patches.remove()
            self.FO_patches.remove()
            self.link_patches.remove()
        self.render_flag = True
        self._frame_steps = 0
        plt.close()
        self.fig = None 
        
    def rot(self,q=None):
        if q is None:
            q = self.qpos
        q = q.reshape(q.shape+(1,1))
        return torch.eye(3,dtype=self.dtype,device=self.device) + torch.sin(q)*self.rot_skew_sym + (1-torch.cos(q))*self.rot_skew_sym@self.rot_skew_sym
    
    @property
    def action_spec(self):
        pass
    @property
    def action_dim(self):
        pass
    @property 
    def action_space(self):
        pass 
    @property 
    def observation_space(self):
        pass 
    @property 
    def obs_dim(self):
        pass


class Locked_Arm_3D(Arm_3D):
    def __init__(self,
            robot='Kinova3', # robot model
            n_obs=1, # number of obstacles
            obs_size_max = [0.1,0.1,0.1], # maximum size of randomized obstacles in xyz
            obs_size_min = [0.1,0.1,0.1], # minimum size of randomized obstacle in xyz
            T_len=50, # number of discritization of time interval
            interpolate = True, # flag for interpolation
            check_collision = True, # flag for whehter check collision
            check_collision_FO = False, # flag for whether check collision for FO rendering
            check_joint_limit = True,
            collision_threshold = 1e-6, # collision threshold
            goal_threshold = 0.1, # goal threshold
            hyp_step = 0.3,
            hyp_dist_to_goal = 0.3,
            hyp_effort = 0.1, # hyperpara
            hyp_success = 50,
            hyp_collision = 50,
            hyp_action_adjust = 0.3,
            hyp_fail_safe = 1,
            hyp_stuck =50,
            hyp_timeout = 0,
            stuck_threshold = None,
            reward_shaping=True,
            gamma = 0.99, # discount factor on reward
            max_episode_steps = 300,
            FO_render_level = 2, # 0: no rendering, 1: a single geom, 2: seperate geoms for each links, 3: seperate geoms for each links and timesteps
            FO_render_freq = 10,
            ticks = False,
            scale = 1,
            max_combs = 200,
            dtype= torch.float,
            device = torch.device('cpu'),
            locked_idx = [],
            locked_qpos = [],
            ):
        self.unlocked_idx = []
        super().__init__(
            robot=robot,
            n_obs=n_obs, 
            obs_size_max = obs_size_max,
            obs_size_min = obs_size_min,
            T_len=T_len, 
            interpolate = interpolate, 
            check_collision = check_collision, 
            check_collision_FO = check_collision_FO, 
            check_joint_limit = check_joint_limit,
            collision_threshold = collision_threshold, 
            goal_threshold = goal_threshold,
            hyp_step = hyp_step,
            hyp_dist_to_goal = hyp_dist_to_goal,
            hyp_effort = hyp_effort,
            hyp_success = hyp_success,
            hyp_collision = hyp_collision,
            hyp_action_adjust = hyp_action_adjust,
            hyp_fail_safe = hyp_fail_safe,
            hyp_stuck = hyp_stuck,
            hyp_timeout = hyp_timeout,
            stuck_threshold = stuck_threshold,
            reward_shaping = reward_shaping,
            gamma = gamma, # discount factor on reward
            max_episode_steps = max_episode_steps,
            FO_render_level = FO_render_level,
            FO_render_freq = FO_render_freq,
            ticks = ticks,
            scale = scale,
            max_combs = max_combs,
            dtype= dtype,
            device = device)

        self.locked_idx = torch.tensor(locked_idx,dtype=int,device=device)
        self.locked_qpos = torch.tensor(locked_qpos,dtype=dtype,device=device)
        self.dof = self.n_links - len(locked_idx)
    
        self.unlocked_idx = torch.ones(self.n_links,dtype=bool,device=device)
        self.unlocked_idx[self.locked_idx] = False
        self.unlocked_idx = self.unlocked_idx.nonzero().reshape(-1)

        locked_pos_lim = self.pos_lim.clone()
        locked_pos_lim[self.locked_idx,0] = locked_pos_lim[self.locked_idx,1] = self.locked_qpos
        self.pos_sampler = torch.distributions.Uniform(locked_pos_lim[:,1],locked_pos_lim[:,0],validate_args=False)

        self.pos_lim = self.pos_lim[self.unlocked_idx]
        self.vel_lim = self.vel_lim[self.unlocked_idx]
        self.tor_lim = self.tor_lim[self.unlocked_idx]
        self.lim_flag = self.lim_flag[self.unlocked_idx] # NOTE, wrap???
        self.joint_id = self.joint_id[self.unlocked_idx]
        self.reset()

    def wrap_cont_joint_to_pi(self,phases,internal=True):
        if internal:
            phases_new = torch.clone(phases)
            idx = self.unlocked_idx[~self.lim_flag]
            phases_new[idx] = (phases[idx] + torch.pi) % (2 * torch.pi) - torch.pi
        else:
            phases_new = torch.zeros(self.n_links).to(phases.device, phases.dtype)
            phases_new[self.unlocked_idx] = phases
            idx = self.unlocked_idx[~self.lim_flag]
            phases_new[idx] = (phases_new[idx] + torch.pi) % (2 * torch.pi) - torch.pi
            phases_new = phases_new[self.unlocked_idx]
        return phases_new

    def set_initial(self,qpos,qvel,qgoal,obs_pos):
        self.qpos = qpos.to(dtype=self.dtype,device=self.device)
        self.qpos[self.locked_idx] = self.locked_qpos
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = qvel.to(dtype=self.dtype,device=self.device)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = qgoal.to(dtype=self.dtype,device=self.device)   
        self.qgoal[self.locked_idx] = self.locked_qpos

        if self.interpolate:
            T_len_to_peak = int((1-T_PLAN/T_FULL)*self.T_len)+1            
            self.qpos_to_brake = self.qpos.unsqueeze(0).repeat(T_len_to_peak,1)
            self.qvel_to_brake = torch.zeros(T_len_to_peak,self.n_links,dtype=self.dtype,device=self.device)        
        else:
            self.qpos_brake = self.qpos + 0.5*self.qvel*(T_FULL-T_PLAN)
            self.qvel_brake = torch.zeros(self.n_links,dtype=self.dtype,device=self.device)   
        self.obs_zonos = []

        R_qi = self.rot()
        R_qg = self.rot(self.qgoal)    
        Ri, Pi = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)       
        Rg, Pg = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)               
        link_init, link_goal = [], []
        for j in range(self.n_links):
            Pi = Ri@self.P0[j] + Pi 
            Pg = Rg@self.P0[j] + Pg
            Ri = Ri@self.R0[j]@R_qi[j]
            Rg = Rg@self.R0[j]@R_qg[j]
            link_init.append(Ri@self.__link_zonos[j]+Pi)
            link_goal.append(Rg@self.__link_zonos[j]+Pg)
        for pos in obs_pos:
            obs = zonotope(torch.vstack((pos.to(dtype=self.dtype,device=self.device),torch.eye(3,dtype=self.dtype,device=self.device))))
            for j in range(self.n_links):
                buff = link_init[j]-obs
                _,b = buff.polytope(self.combs)
                if min(b) > -1e-5:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                _,b = buff.polytope(self.combs)
                if min(b) > -1e-5:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'    
            self.obs_zonos.append(obs)
        self.fail_safe_count = 0
        if self.render_flag == False:
            self.obs_patches.remove()
            self.link_goal_patches.remove()
            self.FO_patches.remove()
            self.link_patches.remove()
        self.render_flag = True
        self.done = False
        self.collision = False

        self._elapsed_steps = 0
        self.reward_com = 0

        return self.get_observations()

    def step(self,ka,flag=-1):

        ka_all = torch.zeros(self.n_links,dtype=self.dtype,device=self.device)
        ka_all[self.unlocked_idx] = ka

        return super().step(ka_all,flag)
        
    
    def get_observations(self):
        observation = {'qpos':self.qpos[self.unlocked_idx],'qvel':self.qvel[self.unlocked_idx],'qgoal':self.qgoal[self.unlocked_idx]}
        if self.n_obs > 0:
            observation['obstacle_pos']= torch.vstack([self.obs_zonos[o].center for o in range(self.n_obs)])
            observation['obstacle_size'] = torch.vstack([torch.diag(self.obs_zonos[o].generators) for o in range(self.n_obs)])
        return observation


if __name__ == '__main__':

    env = Locked_Arm_3D(n_obs=3,T_len=50,interpolate=True,locked_idx=[1,2],locked_qpos = [0,0])
    for _ in range(3):
        for _ in range(10):
            env.step(torch.rand(env.dof))
            env.render()
            #env.reset()