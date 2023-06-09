<img src="figures/logo.png" width="250"/> 

# Reachability-based Trajectory Design with Neural Implicit Safety Constraints
[Project Page](https://roahmlab.github.io/RDF/) | [Paper](https://arxiv.org/abs/2302.07352) | [Dataset](https://drive.google.com/drive/folders/1sxRCtuAwi2Ua5BIVX0fLqOBlb95PcFN0?usp=share_link)

## Introduction
We present Reachability-based Signed Distance Functions (RDFs) as a neural implicit representation for robot safety. RDF, which can be constructed using supervised learning in a tractable fashion, accurately predicts the distance between the swept volume of a robot arm and an obstacle. RDF’s inference and gradient computations are fast and scale linearly with the dimension of the system; these features enable its use within a novel real-time trajectory planning framework as a continuous-time collision-avoidance constraint. The planning method using RDF is compared to a variety of state-of-the-art techniques and is demonstrated to successfully solve challenging motion planning tasks for high-dimensional systems faster and more reliably than all tested methods.

Paper: Reachability-based Trajectory Design with Neural Implicit Safety Constraints. [[Arxiv](https://arxiv.org/abs/2302.07352)]

## Dependency 
- Run `pip install -r requirements.txt` to collect all python dependencies.
- [IPOPT](https://coin-or.github.io/Ipopt/INSTALL.html) is required to run the planning experiments.
- [Gurobi](https://www.gurobi.com) is optionally required if the users would like to generate datasets. Generated datasets are also available at [Google Drive](https://drive.google.com/drive/folders/1sxRCtuAwi2Ua5BIVX0fLqOBlb95PcFN0?usp=share_link).
- [WANDB](https://wandb.ai/) is optionally required if the users would like to train the RDF models. Pretrained RDF models are available in `trained_models/`.
- [CORA 2021](https://tumcps.github.io/CORA/) is optionally required for users to compute JRS in `reachability/joint_reachable_set/gen_jrs_trig` with MATLAB script.

## Reproducing Results

### Building Datasets
Datasets are available at [Google Drive](https://drive.google.com/drive/folders/1sxRCtuAwi2Ua5BIVX0fLqOBlb95PcFN0?usp=share_link). After downloading the datasets, put them under `dataset/`. 

Users who would like to run building dataset on their own can run `bash scripts/generate_datasets.sh` in the repo home directory (i.e., `rdf/`).

Note that building the datasets generally takes days of time. To collect the datasets more efficiently, the paper chose to launch dataset collecting program (`*.py` in `generate_dataset/`) with multiple processes using different random seeds, then combine these sub-datasets together.

Users can follow similar practice by modifying the provided script and launching dataset collecteing programs in parallel or across different machines.

### Training RDF Models

Pretrained models are available in `trained_models/`.

If the users would like to run training on their own, an example script to train a 3D7Links Manipulator is provided in `scripts/train_rdf_model.sh`. Note that [WANDB](https://wandb.ai/) is used to monitor training and is thus a dependency to run training.

Users can follow the guide from WANDB to install it, and then run `bash scripts/train_rdf_model.sh` to launch training with the provided hyperparameters in the training script. Users are free to tune the hyperparameters for model performance. The hyperparameters we used to obtain the pre-trained models are present in the same directory as the models.

### Run Planning Experiments

[IPOPT](https://coin-or.github.io/Ipopt/INSTALL.html) is required to run the planning experiments as the framework to solve non-linear programming optimization problems.

To reproduce the 2D planning experiments, run `bash scripts/run_2d_planning.sh`. 

To reproduce the 3D planning experiments, run `bash scripts/run_3d_planning.sh`.

The results will be in `planning_results/` as generated by the planning program.

### Other Experiments
Here are the procedures to reproduce the other experiments in the paper.

1. Compare RDF with SDF
```
cd experiments
bash run_compare_rdf_and_sdf.sh
```

2. Compare RDF with QP
```
cd experiments
bash run_compare_time_with_QP.sh
```

3. Evaluate RDF models on testset

Download the testsets from [Google Drive](https://drive.google.com/drive/folders/1sxRCtuAwi2Ua5BIVX0fLqOBlb95PcFN0?usp=share_link) and create a directory named `test_dataset/` under the repo home directory (i.e., `rdf/`). Put the datasets in `rdf/test_dataset/`.

Then run
```
cd experiments
bash run_evaluate_model_on_testset.sh
```

## Credits
- `reachability/` referred some part of [CORA](https://tumcps.github.io/CORA/).
- `environments/robots/urdf_parser_py` is extracted from [urdf_parser_py](https://github.com/ros/urdf_parser_py) and modified to our end.

## Citation
The paper with an overview of the theoretical and implementation details is published in Robotics: Science and Systems (RSS 2023). If you use RDF in an academic work, please cite using the following BibTex entry:
```
@misc{michaux2023reachabilitybased,
      title={Reachability-based Trajectory Design with Neural Implicit Safety Constraints}, 
      author={Jonathan Michaux and Qingyi Chen and Yongseok Kwon and Ram Vasudevan},
      year={2023},
      eprint={2302.07352},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

