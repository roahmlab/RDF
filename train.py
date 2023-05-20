import torch
import training.utils as utils
from training.train_model import train_arm_model
import os
import json
import wandb
from datetime import datetime

if __name__ == '__main__':
    params = utils.read_params()
    utils.set_random_seed(params.seed)

    # Data
    dataloaders_dict = utils.make_dataloaders(params, training_proportion=0.8)
    
    # Model
    model, model_name = utils.make_model(params)

    # Logistics
    now = datetime.now()
    since = now.strftime("%m-%d-%H-%M")
    experiment_name = f"{model_name}_lr{params.lr}_eikonal{params.eikonal}_{since}"
    wandb.init(project='SDF RTD', name=experiment_name, group=params.wandb_group_name)
    experiment_name = "runs/" + experiment_name
    if not os.path.exists(experiment_name):
        if not os.path.exists('runs/'):
            os.mkdir('runs/')
        os.mkdir(experiment_name)
    with open(os.path.join(experiment_name, "training_setting.json"), 'w') as f:
        json.dump(params.__dict__, f, indent=4)
    print(f"Launching experiment with config {params.__dict__}")
    device = torch.device(f"cuda:{params.device}" if torch.cuda.is_available() else "cpu")
    
    wandb.config.update(params)
    print(f"Starting experiment {experiment_name} using device {device}")

    # Learning
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, betas=(params.beta1, params.beta2), weight_decay=params.weight_decay)
    if params.robot == 'arm':
        best_model, best_loss = train_arm_model(model, dataloaders_dict, optimizer, num_epochs=params.num_epochs, device=device, experiment_name=experiment_name, params=params)
    else:
        raise NotImplementedError(f"Such training is not implemented yet for {params.robot}")
    print(f"Training ends with best loss = {best_loss}")
    wandb.finish()