import torch
import argparse
import numpy as np
import time
import json
from tqdm import tqdm
from training.utils import set_random_seed
from planning.rdf_3d import RDF_3D_Planner
from planning.armtd_3d import ARMTD_3D_planner
from planning.sdf_3d import SDF_3D_Planner
from environments.wrappers.monitoring.video_recorder import VideoRecorder
import os


def evaluate_planner(planner, planner_name='rdf', device='cpu', n_envs=1000, n_steps=150, n_links=7, n_obs=5, record_episodes=False, video=False, buffer_size=0.0, time_limit=0.5):
    t_armtd = 0.0
    num_success = 0
    num_collision = 0
    num_no_solution = 0
    num_step = 0
    t_armtd_list = []
    t_success_list = []
    t_armtd_optimization_list = []
    video_folder = f'planning_videos/{planner_name}/3d{n_links}links'
    if record_episodes:
        success_episodes = set()
    assert 'rdf' in planner_name or 'armtd' in planner_name or 'sdf' in planner_name, f"Only support rdf or armtd, while received {planner_name}"
        
    for i_env in tqdm(range(n_envs)):
        set_random_seed(i_env)
        env = Arm_3D(n_obs=n_obs, T_len=24, max_episode_steps=n_steps)
        t_curr_trial = []
        
        if 'armtd' in planner_name:
            planner = ARMTD_3D_planner(env, device=device)
        if video:
            video_path = os.path.join(
                video_folder, f'video{i_env}')
            video_recorder = VideoRecorder(
                env, video_path, frame_rate=3, format='gif')
        was_stuck = False
        for i_step in range(n_steps):
            ts = time.time()
            k0 = torch.zeros(n_links)
            if 'rdf' in planner_name or 'sdf' in planner_name:
                ka, flag = planner.plan(env, k0, buffer_size=buffer_size, time_limit=time_limit)
            elif 'armtd' in planner_name:
                ka, flag, t_nlp = planner.plan(env, k0, time_limit=time_limit)
                t_armtd_optimization_list.append(t_nlp)
            t_elasped = time.time()-ts
            t_armtd += t_elasped
            t_armtd_list.append(t_elasped)
            t_curr_trial.append(t_elasped)

            observations, reward, done, info = env.step(ka.cpu(), flag != 0)
            if video:
                video_recorder.capture_frame()

            num_step += 1
            if 'collision' in info and info['collision']:
                num_collision += 1
                break
            elif 'is_success' in info and info['is_success']:
                num_success += 1
                t_success_list += t_curr_trial
                if record_episodes:
                    success_episodes.add(i_env)
                break
            elif done:
                break

            if flag != 0:
                if was_stuck:
                    num_step -= 1
                    break
                else:
                    was_stuck = True
                    if flag > 0 or flag == -5:
                        num_no_solution += 1
            else:
                was_stuck = False
        if video:
            video_recorder.close(True)

    stats = {
        'n_trials': n_envs,
        'n_links': n_links,
        'n_obs':n_obs,
        'time_limit': time_limit,
        'num_success': num_success,
        'num_collision': num_collision,
        'mean planning time': np.mean(np.array(t_armtd_list)),
        'std planning time': np.std(np.array(t_armtd_list)),
        'mean planning time for success trials': np.mean(np.array(t_success_list)),
        'std planning time for success trials': np.std(np.array(t_success_list)),
        'total planning time': t_armtd,
        'num_no_solution': num_no_solution,
        'num_step': num_step,
    } 
    if 'rdf' in planner_name or 'sdf' in planner_name:
        stats['buffer_size'] = buffer_size
    if 'armtd' in planner_name:
        stats['mean time for solving optimization'] = np.mean(np.array(t_armtd_optimization_list))
        stats['std time for solving optimization'] = np.std(np.array(t_armtd_optimization_list))
        
    # with open(f"planning_results/3d7links{n_obs}obs/{planner_name}_time_3d{n_links}links{n_envs}trials{n_obs}obs{n_steps}steps_{time_limit}limit.npy", 'wb') as f:
    #     np.save(f, np.array(t_armtd_list))
    with open(f"planning_results/3d7links{n_obs}obs/{planner_name}_stats_3d{n_links}links{n_envs}trials{n_obs}obs{n_steps}steps_{time_limit}limit.json", 'w') as f:
        if record_episodes:
            stats['success_episodes'] = list(success_episodes)
        json.dump(stats, f, indent=2)
        if record_episodes:
            stats['success_episodes'] = success_episodes
        
    return stats
    

def read_params():
    parser = argparse.ArgumentParser(description="Rdf Planning")
    # general env setting
    parser.add_argument("--planner", type=str, default='both') # rdf, armtd, both
    parser.add_argument('--n_links', type=int, default=7)
    parser.add_argument('--n_dims', type=int, default=3)
    parser.add_argument('--n_obs', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=500)
    parser.add_argument('--n_steps', type=int, default=400)
    parser.add_argument('--video',  action='store_true')
    parser.add_argument('--time_limit',  type=float, default=0.5)
    
    # model info
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # optimization info
    parser.add_argument('--buffer_size', type=float, default=0.0)
    parser.add_argument('--n_sdf_interpolate', type=int, default=100)

    # results info
    parser.add_argument('--compare',  action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    from environments.arm_3d import Arm_3D
    
    params = read_params()
    planner_name = params.planner
    model_path = params.model
    if model_path == '' and planner_name == 'rdf':
        model_path = 'trained_models/RDF3D/7links/3d-signed-convexhull.pth'
    elif model_path == '' and planner_name == 'sdf':
        model_path = 'trained_models/SDF3D/7links/3d-signed-convexhull.pth'
    device = torch.device(params.device)

    print(f"Running {params.n_envs}trials of 3D{params.n_links}Links{params.n_obs}obs with {params.n_steps} step limit and {params.time_limit}s time limit each step")
    print(f"Using device {device}")
    
    planning_result_dir = f'planning_results/3d7links{params.n_obs}obs'
    if not os.path.exists('planning_results'):
        os.mkdir('planning_results')
    if not os.path.exists(planning_result_dir):
        os.mkdir(planning_result_dir)
    
    stats = {}
    if planner_name == 'rdf' or planner_name == 'all' or planner_name == 'both':
        if model_path == '':
            model_path = 'trained_models/RDF3D/7links/3d-signed-convexhull.pth'
        rdf_planner = RDF_3D_Planner(model_path=model_path, device=device, n_links=params.n_links, n_dims=params.n_dims)
        stats['rdf'] = evaluate_planner(
            planner=rdf_planner,
            planner_name=f'rdf_b{params.buffer_size}_t{params.time_limit}', 
            device=device, 
            n_envs=params.n_envs, 
            n_steps=params.n_steps, 
            n_links=params.n_links,
            n_obs=params.n_obs, 
            record_episodes=params.compare, 
            video=params.video, 
            buffer_size=params.buffer_size,
            time_limit=params.time_limit,
        )
        
    if planner_name == 'sdf' or planner_name == 'all':
        if model_path == '':
            model_path = 'trained_models/SDF3D_KOPTEV/7links/3d-signed-convexhull.pth'
        sdf_planner = SDF_3D_Planner(model_path=model_path, device=device, n_links=params.n_links, n_dims=params.n_dims, n_interpolate=params.n_sdf_interpolate)
        stats['sdf'] = evaluate_planner(
            planner=sdf_planner,
            planner_name=f'sdf_b{params.buffer_size}_t{params.time_limit}_i{params.n_sdf_interpolate}', 
            device=device, 
            n_envs=params.n_envs, 
            n_steps=params.n_steps, 
            n_links=params.n_links,
            n_obs=params.n_obs, 
            record_episodes=params.compare, 
            video=params.video, 
            buffer_size=params.buffer_size,
            time_limit=params.time_limit,
        )

    if planner_name == 'armtd' or planner_name == 'all' or planner_name == 'both':
        stats['armtd'] = evaluate_planner(
            planner=None,
            planner_name=f'armtd_t{params.time_limit}', 
            device=device, 
            n_envs=params.n_envs, 
            n_steps=params.n_steps, 
            n_links=params.n_links, 
            n_obs=params.n_obs,
            record_episodes=params.compare, 
            video=params.video, 
            buffer_size=params.buffer_size,
            time_limit=params.time_limit,
        )

    print(f"statistics: {stats}")
    if params.compare and planner_name == 'both':
        print(f"The successes of RADAR-ARMTD={stats['rdf']['success_episodes'] - stats['armtd']['success_episodes']}")
        print(f"The successes of ARMTD-RADAR={stats['armtd']['success_episodes'] - stats['rdf']['success_episodes']}")