# Generate datasets for 2D Manipulator. Each dataset consists of 100 sub-datasets, 80% for training, 20% for testing.
cd generate_dataset/
n_links=(2 4 6 8 10)
for n_link in "${n_links[@]}";do
    for s in {0..100};do
        python generate_rdf_convexhull_2d_dataset_multiple_links.py  --n_links $n_link --n_dims 2 --n_obs 16 --n_data 32000 --verbose 0 --save --signed --seed $s
    done
done

# Generate datasets for 3D7Links Manipulator, in RDF style and in SDF style.
for s in {0..200};do
    python generate_rdf_convexhull_3d_dataset.py  --n_links 7 --n_dims 3 --n_obs 16 --n_data 32000 --verbose 0 --save --signed --seed $s
    python generate_sdf_convexhull_3d_dataset.py  --n_links 7 --n_dims 3 --n_obs 16 --n_data 32000 --verbose 0 --save --signed --seed $s
done

