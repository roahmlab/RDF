"""
2D Arm Environment
Author: Yongseok Kwon
"""

import torch 
import matplotlib.pyplot as plt 
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import os
import sys
sys.path.append('..')
from reachability.conSet import zonotope, polyZonotope, batchZonotope


def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1

class Arm_2D:
    def __init__(self,
            n_links=2, # number of links
            n_obs=1, # number of obstacles
            obs_size_max = [0.1,0.1], # maximum size of randomized obstacles in xy
            obs_size_min = [0.1,0.1], # minimum size of randomized obstacle in xy
            T_len=50, # number of discritization of time interval
            interpolate = True, # flag for interpolation
            check_collision = True, # flag for whehter check collision
            check_collision_FO = False, # flag for whether check collision for FO rendering
            collision_threshold = 1e-6, # collision threshold
            goal_threshold = 0.05, # goal threshold
            hyp_step = 0.3,
            hyp_dist_to_goal = 0.3,
            hyp_effort = 0.1, # hyperpara
            hyp_success = 50,
            hyp_collision = 50,
            hyp_action_adjust = 0.3,
            hyp_fail_safe = 1,
            hyp_stuck = 50,
            hyp_timeout = 0,
            stuck_threshold = None,
            reward_shaping=True,
            gamma = 0.99, # discount factor on reward
            max_episode_steps = 100,
            FO_render_level = 2, # 0: no rendering, 1: a single geom, 2: seperate geoms for each links, 3: seperate geoms for each links and timesteps
            ticks = False,
            dtype= torch.float,
            device = torch.device('cpu'),
            scale_down = 1
            ):

        self.dimension = 2
        self.scale_down = scale_down
        self.dof = self.n_links = n_links
        self.joint_id = torch.arange(self.n_links,dtype=int,device=device)
        self.n_obs = n_obs
        self.obs_size_sampler = torch.distributions.uniform.Uniform(torch.tensor(obs_size_min,dtype=dtype,device=device) / scale_down,torch.tensor(obs_size_max,dtype=dtype,device=device) / scale_down,validate_args=False)
        link_Z = torch.tensor([[0.5, 0, 0],[0.5,0,0],[0,0.01,0]],dtype=dtype,device=device) / scale_down
        self.link_zonos = [polyZonotope(link_Z,0)]*n_links
        self.__link_zonos = [zonotope(link_Z)]*n_links 
        self.link_polygon = [zono.project([0,1]).polygon() for zono in self.__link_zonos]
        self.P0 = [torch.tensor([0.0,0.0,0.0],dtype=dtype,device=device)]+[torch.tensor([1.0,0.0,0.0],dtype=dtype,device=device) / scale_down]*(n_links-1)
        self.R0 = [torch.eye(3,dtype=dtype,device=device)]*n_links
        self.joint_axes = torch.tensor([[0.0,0.0,1.0]]*n_links,dtype=dtype,device=device)
        w = torch.tensor([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0.0]]],dtype=dtype,device=device)
        self.rot_skew_sym = (w@self.joint_axes.T).transpose(0,-1)

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
        self.ticks = ticks

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._frame_steps = 0     
        self.dtype = dtype
        self.device = device
        self.reset()
        
    def reset(self):
        self.qpos = torch.rand(self.n_links,dtype=self.dtype,device=self.device)*2*torch.pi - torch.pi
        self.qpos_int = torch.clone(self.qpos)
        self.qvel = torch.zeros(self.n_links,dtype=self.dtype,device=self.device)
        self.qpos_prev = torch.clone(self.qpos)
        self.qvel_prev = torch.clone(self.qvel)
        self.qgoal = torch.rand(self.n_links,dtype=self.dtype,device=self.device)*2*torch.pi - torch.pi
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
        
        obs_size = self.obs_size_sampler.sample([self.n_obs])
        
        for o in range(self.n_obs):
            while True:
                r,th = torch.rand(2,dtype=self.dtype,device=self.device)
                #obs_pos = torch.rand(2)*2*self.n_links-self.n_links
                obs_pos = 3/4*self.n_links*r*torch.tensor([torch.cos(2*torch.pi*th),torch.sin(2*torch.pi*th)]) / self.scale_down
                obs = torch.hstack((torch.vstack((obs_pos,torch.diag(obs_size[o]))),torch.zeros(3,1)))
                obs = zonotope(obs)
                safe_flag = True
                for j in range(self.n_links):
                    buff = link_init[j]-obs
                    _,b = buff.project([0,1]).polytope()
                    if min(b) > -1e-5:
                        safe_flag = False
                        break
                    buff = link_goal[j]-obs
                    _,b = buff.project([0,1]).polytope()
                    if min(b) > -1e-5:
                        safe_flag = False
                        break

                if safe_flag:
                    self.obs_zonos.append(obs)
                    break

        self.fail_safe_count = 0
        if self.render_flag == False:
            self.one_time_patches.remove()
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
            obs_size = [torch.tensor([.1,.1],dtype=self.dtype,device=self.device) for _ in range(self.n_obs)]
            
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
            obs = torch.hstack((torch.vstack((pos.to(dtype=self.dtype,device=self.device),torch.diag(size).to(dtype=self.dtype,device=self.device))),torch.zeros(3,1,dtype=self.dtype,device=self.device)))
            obs = zonotope(obs)
            for j in range(self.n_links):
                buff = link_init[j]-obs
                _,b = buff.project([0,1]).polytope()
                if min(b) > -1e-5:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
                buff = link_goal[j]-obs
                _,b = buff.project([0,1]).polytope()
                if min(b) > -1e-5:
                    assert False, 'given obstacle position is in collision with initial and goal configuration.'
            self.obs_zonos.append(obs)

        self.fail_safe_count = 0
        if self.render_flag == False:
            self.one_time_patches.remove()
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
        self.step_flag = int(flag)
        self.safe = flag <= 0
        # -torch.pi<qvel+k*T_PLAN < torch.pi
        # (-torch.pi-qvel)/T_PLAN < k < (torch.pi-qvel)/T_PLAN
        ka = ka.detach()
        self.ka = ka.clamp((-torch.pi-self.qvel)/T_PLAN,(torch.pi-self.qvel)/T_PLAN) # velocity clamp
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
                self.qvel_brake = torch.zeros(self.n_links,self.n_links,dtype=self.dtype,device=self.device)
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
                'step_flag':self.step_flag,
                'safe_flag':self.safe,
                'stuck':self.stuck,
                }
        if self.collision:
            collision_info = {
                'qpos_collision':self.qpos_collision,
                'qpos_init':self.qpos_int,
                'qvel_int':torch.zeros(self.n_links,dtype=self.dtype,device=self.device),
                'obs_pos':[self.obs_zonos[o].center[:2] for o in range(self.n_obs)],
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
            observation['obstacle_pos']= torch.vstack([self.obs_zonos[o].center[:2] for o in range(self.n_obs)])
            observation['obstacle_size'] = torch.vstack([torch.diag(self.obs_zonos[o].generators) for o in range(self.n_obs)])
        return observation

    def get_reward(self, action, qpos=None, qgoal=None, collision=None, step_flag = None, safe=None, stuck=None, timeout=None):
        # Get the position and goal then calculate distance to goal
        if qpos is None:
            # deliver termination variable
            collision = self.collision 
            action_adjusted = self.step_flag == 0
            safe = self.safe
            stuck = self.stuck = self.fail_safe_count >= self.stuck_threshold
            goal_dist = torch.linalg.norm(wrap_to_pi(self.qpos-self.qgoal))
            success = self.success = bool(goal_dist < self.goal_threshold) 
            # compute done and timeout
            done = self.success or self.collision or self.stuck
            timeout = self.timeout = (self._elapsed_steps >= self._max_episode_steps) and (~done)
            self.done = done or self.timeout
            
        else: 
            action_adjusted = step_flag == 0
            goal_dist = torch.linalg.norm(wrap_to_pi(qpos-qgoal))
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
            goal_dist = torch.linalg.norm(wrap_to_pi(self.qpos-self.qgoal)) 
        else:
            goal_dist = torch.linalg.norm(wrap_to_pi(qpos-qgoal))
        return bool(goal_dist < self.goal_threshold)

    def collision_check(self,qs):

        if self.check_collision:
            
            R_q = self.rot(qs)
            if len(R_q.shape) == 4:
                time_steps = len(R_q)
                R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device)
                for j in range(self.n_links):
                    P = R@self.P0[j] + P
                    R = R@self.R0[j]@R_q[:,j]
                    link =batchZonotope(self.link_zonos[j].Z.unsqueeze(0).repeat(time_steps,1,1))
                    link = R@link+P
                    for o in range(self.n_obs):
                        buff = link - self.obs_zonos[o]
                        _,b = buff.project([0,1]).polytope()
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
                        _,b = buff.project([0,1]).polytope()
                        if min(b) > 1e-6:
                            self.qpos_collision = qs
                            return True
  
        return False

    def render(self,FO_link=None,show=True,dpi=None,save_kwargs=None):
        '''
        save_kwargs = {'frame_rate':frame_rate,'save_path':save_path, 'dpi':dpi}
        '''
        if self.render_flag:
            if self.fig is None:
                if show:
                    plt.ion()
                self.fig = plt.figure(figsize=[self.fig_scale*6.4,self.fig_scale*4.8],dpi=dpi)
                self.ax = self.fig.gca()
                if not self.ticks:
                    plt.tick_params(which='both',bottom=False,top=False,left=False,right=False, labelbottom=False, labelleft=False) 

                if save_kwargs is not None:
                    os.makedirs(save_kwargs['save_path'],exist_ok=True)
                    self.ax.set_title(save_kwargs['text'],fontsize=10,loc='right')

            self.render_flag = False
            self.FO_patches = self.ax.add_collection(PatchCollection([]))
            self.link_patches = self.ax.add_collection(PatchCollection([]))
            one_time_patches = []
            for o in range(self.n_obs):
                one_time_patches.append(self.obs_zonos[o].polygon_patch(edgecolor='red',facecolor='red'))
            R_q = self.rot(self.qgoal)
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(3,dtype=self.dtype,device=self.device) 
            for j in range(self.n_links):
                P = R@self.P0[j] + P
                R = R@self.R0[j]@R_q[j]
                link_goal_polygon = (self.link_polygon[0]@R[:2,:2].T+P[:2]).cpu()
                one_time_patches.append(Polygon(link_goal_polygon,alpha=.5,edgecolor='gray',facecolor='gray',linewidth=.2))
            self.one_time_patches = PatchCollection(one_time_patches, match_original=True)
            self.ax.add_collection(self.one_time_patches)

        if FO_link is not None: 
            FO_patches = []
            if self.fail_safe_count == 0:
                #g_ka = torch.maximum(self.PI/24,abs(self.qvel_prev/3)) # NOTE: is it correct?
                g_ka = self.PI/24
                self.FO_patches.remove()
                if self.FO_render_level == 3:
                    for j in range(self.n_links):
                        FO_link_slc = FO_link[j].to(dtype=self.dtype,device=self.device).slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1)) 
                        FO_link_polygons = FO_link_slc.polygon().detach()
                        FO_patches.extend([Polygon(polygon,alpha=0.1,edgecolor='green',facecolor='none',linewidth=.2) for polygon in FO_link_polygons])
                elif self.FO_render_level == 2:
                    for j in range(self.n_links):
                        FO_link_slc = FO_link[j].to(dtype=self.dtype,device=self.device).slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1)) 
                        FO_link_polygons = FO_link_slc.polygon().detach()
                        FO_patches.append(Polygon(FO_link_polygons.reshape(-1,2),alpha=0.3,edgecolor='none',facecolor='green',linewidth=.2))   
                '''
                elif self.FO_render_level == 1:
                    FO_link_polygons = []
                    for j in range(self.n_links):
                        FO_link_slc = FO_link[j].to(dtype=self.dtype,device=self.device).slice_all_dep((self.ka/g_ka).unsqueeze(0).repeat(100,1)) 
                        FO_link_polygons.append(FO_link_slc.polygon().detach().reshape(-1,2))
                    FO_patches.append(Polygon(torch.vstack(FO_link_polygons),alpha=0.3,edgecolor='none',facecolor='green',linewidth=.2))
                        
                '''
                '''
                NOTE: different methods to plot FO
                Option 0: Plot all 100 timesteps and joints FO as a single geometry

                Option 1: Plot all 100 timestep FO as a single geometry
                FO_patches.append(Polygon(FO_link_polygons.reshape(-1,2),alpha=0.3,edgecolor='none',facecolor='green',linewidth=.2))

                Options 2: Plot all 100 timestep FO with seperate geometries + possibly pass NaN value
                FO_patches.extend([Polygon(polygon,alpha=0.1,edgecolor='green',facecolor='none',linewidth=.2) for polygon in FO_link_polygons])
                
                Options 3: Plot all 100 timestep FO with seperate geometries + possibly pass NaN value
                FO_patches.append([Polygon(polygon[~polygon[:,0].isnan()],alpha=0.1,edgecolor='green',facecolor='none',linewidth=.2) for polygon in FO_link_polygons])

                Option 4: Just collect polytope for every single timestep without using batch computation
                for t in range(100): 
                    FO_patch = FO_link_slc[t].polygon_patch(alpha=0.1,edgecolor='green')
                    FO_patches.append(FO_patch)
                '''

                self.FO_patches = PatchCollection(FO_patches, match_original=True)
                self.ax.add_collection(self.FO_patches)            

        if self.interpolate:
            
            timesteps = int(T_PLAN/T_FULL*self.T_len) # NOTE: length match??
            if show and save_kwargs is None:
                plot_freq = 1
                R_q = self.rot(self.qpos_to_peak[1:])
            elif show:
                plot_freq = timesteps//save_kwargs['frame_rate']
                R_q = self.rot(self.qpos_to_peak[1:])      
            else:
                plot_freq = 1
                t_idx = torch.arange(timesteps+1,device=self.device)%(timesteps//save_kwargs['frame_rate'] ) == 1
                R_q = self.rot(self.qpos_to_peak[t_idx])
                timesteps = len(R_q)

            # NOTE: only compute one that in freq
            '''
            if not self.done:
                timesteps = int(T_PLAN/T_FULL*self.T_len)
            else:
                timesteps = self.until_goal
            '''
            link_trace_polygons =  torch.zeros(timesteps,5*self.n_links,2,dtype=self.dtype,device='cpu')
            R, P = torch.eye(3,dtype=self.dtype,device=self.device), torch.zeros(1,3,dtype=self.dtype,device=self.device)
            link_trace_patches = []
            for j in range(self.n_links):
                P = R@self.P0[j] + P
                R = R@self.R0[j]@R_q[:,j]
                link_trace_polygons[:,5*j:5*(j+1)] = (self.link_polygon[j]@R[:,:2,:2].transpose(-1,-2)+P[:,:2].unsqueeze(-2)).cpu()
            
            for t in range(timesteps):
                self.link_patches.remove()    
                link_trace_patches = [Polygon(link_trace_polygons[t,5*j:5*(j+1)],alpha=.5,edgecolor='blue',facecolor='blue',linewidth=.2) for j in range(self.n_links)]
                self.link_patches = PatchCollection(link_trace_patches, match_original=True)
                self.ax.add_collection(self.link_patches)
                ax_scale = 1.2
                axis_lim = ax_scale*self.n_links / self.scale_down
                
                plt.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
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
                link_trace_polygon = (self.link_polygon[0]@R[:2,:2].T+P[:2]).cpu()
                link_trace_patches.append(Polygon(link_trace_polygon,alpha=.5,edgecolor='blue',facecolor='blue',linewidth=.2))
            self.link_patches = PatchCollection(link_trace_patches, match_original=True)
            self.ax.add_collection(self.link_patches)
            ax_scale = 1.2
            axis_lim = ax_scale*self.n_links
            plt.axis([-axis_lim,axis_lim,-axis_lim,axis_lim])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            if save_kwargs is not None:
                filename = "/frame_"+"{0:04d}".format(self._frame_steps)                    
                self.fig.savefig(save_kwargs['save_path']+filename,dpi=save_kwargs['dpi'])
                self._frame_steps+=1

    def close(self):
        if self.render_flag == False:
            self.one_time_patches.remove()
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

if __name__ == '__main__':

    env = Arm_2D(n_obs=2)
    #from zonopy.optimize.armtd import ARMTD_planner
    #planner = ARMTD_planner(env)
    for _ in range(20):
        for _ in range(10):
            #ka, flag = planner.plan(env.qpos,env.qvel,env.qgoal,env.obs_zonos,torch.zeros(2))
            observations, reward, done, info = env.step(torch.rand(2))
            env.render()
            if done:
                env.reset()
                break
            
    '''

    env = Batch_Arm_2D()
    for _ in range(50):
        env.step(torch.rand(env.n_batches,2))
        env.render()
    '''    

