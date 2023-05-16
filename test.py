import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm

from panda_pushing_env import PandaImageSpacePushingEnv
from visualizers import GIFVisualizer, NotebookVisualizer
from learning_latent_dynamics import *
from utils import *

def test(model, val_loader, loss_fn) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param loss: Loss function
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0. # TODO: Modify the value
    # Initialize the validation loop
    model.eval()
    for batch in val_loader:
        loss = None
        states = batch['states']
        actions = batch['actions']
        loss = loss_fn(model, states, actions)
        val_loss += loss.item()
    return val_loss/len(val_loader)

if __name__ == "__main__":
    LR = 0.001
    NUM_EPOCHS = 2000
    BETA = 0.001
    LATENT_DIM = 16
    ACTION_DIM = 3
    POLY_ORDER = 2
    NUM_CHANNELS = 1
    
    collected_data = np.load('pushing_image_data.npy', allow_pickle=True)
    train_loader, val_loader, norm_constants = process_data_multiple_step(collected_data, batch_size=500, num_steps=NUM_STEPS)
    norm_tr = NormalizationTransform(norm_constants)
    
    test_data = np.load('pushing_image_validation_data.npy', allow_pickle=True)
    single_step_dataset = MultiStepDynamicsDataset(test_data, num_steps=1, transform=norm_tr)
    single_step_loader = DataLoader(single_step_dataset, batch_size=len(single_step_dataset))

    multi_step_dataset = MultiStepDynamicsDataset(test_data, num_steps=4, transform=norm_tr)
    multi_step_loader = DataLoader(multi_step_dataset, batch_size=len(multi_step_dataset))

    state_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()
    loss = MultiStepLoss(state_loss_fn, latent_loss_fn, lambda_state_loss=0.1, lambda_latent_loss=0.1,lambda_reg_loss=0.2)

    single_step_sindy_dynamics_model = SINDyModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, poly_order=POLY_ORDER, include_sine=True, num_channels=NUM_CHANNELS)
    model_path = 'single_step_sindy_dynamics_model.pt'
    single_step_sindy_dynamics_model.load_state_dict(torch.load(model_path))

    print(f'Single-step model evaluated on single-step loss: {test(single_step_sindy_dynamics_model, single_step_loader, loss)}')
    print(f'Single-step model evaluated on multi-step loss: {test(single_step_sindy_dynamics_model, multi_step_loader, loss)}')
    print('')

    # multi_step_sindy_dynamics_model = SINDyModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, poly_order=POLY_ORDER, include_sine=True, num_channels=NUM_CHANNELS)
    # model_path = 'multi_step_sindy_dynamics_model.pt'
    # multi_step_sindy_dynamics_model.load_state_dict(torch.load(model_path))
    # print(f'Multi-step model evaluated on single-step loss: {test(multi_step_sindy_dynamics_model, single_step_loader, loss)}')
    # print(f'Multi-step model evaluated on multi-step loss: {test(multi_step_sindy_dynamics_model, multi_step_loader, loss)}')