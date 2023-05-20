time_limits=(5.0 0.3 0.033)

for time_limit in "${time_limits[@]}";do
    python run_statistics_armtd_2d.py --n_links 2 --planner both --buffer_size 0.03 --compare --time_limit $time_limit

    python run_statistics_armtd_2d.py --n_links 4 --planner both --buffer_size 0.03 --compare --time_limit $time_limit  
    
    python run_statistics_armtd_2d.py --n_links 6 --planner both --buffer_size 0.03 --compare --time_limit $time_limit

    python run_statistics_armtd_2d.py --n_links 8 --planner both --buffer_size 0.035 --compare --time_limit $time_limit

    python run_statistics_armtd_2d.py --n_links 10 --planner both --buffer_size 0.035 --compare --time_limit $time_limit
done



