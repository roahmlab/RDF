python train.py \
    --device 0 \
    --n_links 7 \
    --n_dims 3 \
    --data convexhull_3d_7link_16obs_160size_signed_1-160seed.pkl \
    --num_hidden_layers 8 \
    --hidden_size 1024 \
    --loss MSE \
    --weight_decay 0.01 \
    --lr 0.0001 \
    --batch_size 1024 \
    --num_epochs 350 \
    --wandb_group_name 3D7LINKS \
    --eikonal 0.001 \
    --signed \
    --normalize \
    --fix_size