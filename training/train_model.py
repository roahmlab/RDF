import os
import copy
from tqdm import tqdm
import torch
import wandb
from training.utils import make_loss, compute_eikonal_loss

def train_arm_model(model, dataloaders, optimizer, device, experiment_name, num_epochs=25, params=None):
    model = model.to(device)
    best_loss = torch.inf
    best_model = None
    criterion = make_loss(params)
    if not os.path.exists(os.path.join(experiment_name,'saved_models')):
        os.mkdir(os.path.join(experiment_name,'saved_models'))
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            loss_dict = {}
            running_loss_dict = {}
            max_difference = torch.tensor(0.0).to(device)
            max_collision_prediction = torch.tensor(0.0).to(device)
            mean_difference = torch.tensor(0.0).to(device)

            # Iterate over data.
            for (qpos, qvel, obstacle, k, labels) in tqdm(dataloaders[phase]):
                qpos, qvel, obstacle, k = qpos.to(device), qvel.to(device), obstacle.to(device), k.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(qpos=qpos, qvel=qvel, obstacle=obstacle, k=k)
                    
                    loss_dict['distance_loss'] = criterion(outputs, labels)
                    if params.eikonal > 0.0:
                        # NOTE: obstacle should not have varying size
                        loss_dict['eikonal_loss'] = params.eikonal * compute_eikonal_loss(inputs=obstacle, outputs=outputs)

                    loss = 0.0
                    for k, v in loss_dict.items():
                        loss += v

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                for k, v in loss_dict.items():  
                    curr_loss = v.item() * labels.size(0)
                    if k in running_loss_dict:
                        running_loss_dict[k] += curr_loss
                    else:
                        running_loss_dict[k] = curr_loss
                outputs_data = outputs.detach()
                curr_difference = torch.abs(outputs_data - labels)
                mean_difference += curr_difference.mean(dim=1).sum()
                max_difference = torch.max(torch.max(curr_difference), max_difference)
                collision_mask = (labels <= 0.0)
                if torch.any(collision_mask):
                    max_collision_prediction = torch.max(torch.max(outputs_data[collision_mask]), max_collision_prediction)
            
            log = {
                f'{phase}/max pred vs. truth difference': max_difference,
                f'{phase}/max pred for collision states': max_collision_prediction,
                f'{phase}/mean pred vs. truth difference': mean_difference / len(dataloaders[phase].dataset),
            }
            epoch_loss = 0.0
            for k, v in running_loss_dict.items():
                curr_loss = v / len(dataloaders[phase].dataset)
                log[f'{phase}/epoch_{k}'] = curr_loss
                epoch_loss += curr_loss
            log[f'{phase}/epoch_loss'] = epoch_loss

            wandb.log(log, step=epoch)

            print(f'{phase} Loss: {epoch_loss}')

            if phase == 'val' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and (epoch - 1) % (num_epochs // 4) == 0:
                if params.model == 'LipMLP':
                    model.apply_normalized_weights()
                torch.save(model.state_dict(), os.path.join(experiment_name, 'saved_models', f'model_epoch{epoch}_loss{epoch_loss:.3f}.pth'))

    print(f'Best val Loss: {best_loss}')
    torch.save(best_model, os.path.join(experiment_name, 'saved_models', f'best_model_loss{best_loss:.3f}.pth'))

    return best_model, best_loss


