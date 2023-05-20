time_limits=(5.0 0.3 0.033 0.025)
n_obss=(2 5 10)

for n_obs in "${n_obss[@]}";do
    for time_limit in "${time_limits[@]}";do
        python run_statistics_armtd_3d.py --planner sdf --buffer_size 0.03  --compare --n_obs $n_obs --time_limit $time_limit --n_envs 500  --n_sdf_interpolate 100
        python run_statistics_armtd_3d.py --planner rdf --buffer_size 0.03  --compare --n_obs $n_obs --time_limit $time_limit --n_envs 500 
        python run_statistics_armtd_3d.py --planner armtd --compare --n_obs $n_obs --time_limit $time_limit --n_envs 500 
    done
done
