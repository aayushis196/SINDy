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

if __name__ == "__main__":
    LR = 0.001
    NUM_EPOCHS = 3000
    BETA = 0.001
    LATENT_DIM = 16
    ACTION_DIM = 3
    POLY_ORDER = 4
    NUM_CHANNELS = 1
    LIBRARY_DIM = library_size(LATENT_DIM, POLY_ORDER, True, True)
    print(LIBRARY_DIM*LATENT_DIM)
    SEQUENTIAL_THRESOLDING = True
    THRESOLDING_FREQUENCY = 300
    COEFFICIENT_THRESOLD = 0.001
    coefficient_mask = torch.ones(LIBRARY_DIM,LATENT_DIM)

    # Compute normalization constants
    collected_data = np.load('pushing_image_data.npy', allow_pickle=True)
    train_loader, val_loader, norm_constants = process_data_multiple_step(collected_data, batch_size=500, num_steps=1)
    norm_tr = NormalizationTransform(norm_constants)

    single_step_sindy_dynamics_model = SINDyModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, poly_order=POLY_ORDER, include_sine=True, sequential_thresholding=True, num_channels=NUM_CHANNELS)

    state_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()
    multistep_loss = MultiStepLoss(state_loss_fn, latent_loss_fn, lambda_state_loss=0.1, lambda_latent_loss=0.1,lambda_reg_loss=0.3)


    optimizer = optim.Adam(single_step_sindy_dynamics_model.parameters(), lr=LR)
    pbar = tqdm(range(NUM_EPOCHS))

    # single_step_sindy_dynamics_model.train()
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = 0.
        val_loss_i = 0.
        # --- Your code here
        # train_loss =0.
        for batch_idx, (data) in enumerate(train_loader):
            # --- Your code here
            input_state = data['states']
            input_action = data['actions']
            # recons_state, mu, logvar, latent_state = single_step_latent_dynamics_model(input_state)
            single_step_sindy_dynamics_model.zero_grad()
            if SEQUENTIAL_THRESOLDING and (epoch_i % THRESOLDING_FREQUENCY ==0) and (epoch_i >0):
                coefficient_mask[single_step_sindy_dynamics_model.sindy_coefficients < COEFFICIENT_THRESOLD] = 0 
                print('THRESHOLDING: %d active coefficients' % torch.sum(coefficient_mask))
            # else:
            #     loss = multistep_loss(single_step_sindy_dynamics_model,input_state,input_action)
            loss = multistep_loss(single_step_sindy_dynamics_model,input_state,input_action, coefficient_mask)
            loss.backward()

            optimizer.step()
            train_loss_i += loss.item()

        train_loss_i = train_loss_i/len(train_loader)
        train_losses.append(train_loss_i)

        # Initialize the validation loop
        val_loss_i = 0.
        
        for batch_idx, (data) in enumerate(val_loader):
            input_state = data['states']
            input_action = data['actions']
            # single_step_sindy_dynamics_model.eval()
            with torch.no_grad():
                loss = multistep_loss(single_step_sindy_dynamics_model,input_state,input_action,coefficient_mask)
                val_loss_i += loss.item()

        val_loss_i = val_loss_i/len(val_loader)
        val_losses.append(val_loss_i)
        pbar.set_description(f'Train Loss: {train_loss_i:.4f}, Validation Loss: {val_loss_i:.4f}')

    # losses = train_losses
    # plot train loss and test loss:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
    axes = [axes]
    axes[0].plot(train_losses)
    axes[0].grid()
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_yscale('log')
    plt.show(block=True)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
    axes = [axes]
    axes[0].plot(val_losses)
    axes[0].grid()
    axes[0].set_title('Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_yscale('log')
    plt.show(block=True)
    # ---
    traj = collected_data[0]
    evaluate_model_plot(single_step_sindy_dynamics_model,traj, norm_tr,coefficient_mask)
    
    string = str(train_losses)
    with open('train_loss_polyorder3.txt','w') as fp:
        fp.write(string)
    string = str(val_losses)
    with open('val_loss_polyorder3.txt','w') as fp:
        fp.write(string)

    ## save model:
    save_path = 'single_step_sindy_dynamics_model_polyorder3.pt'
    torch.save(single_step_sindy_dynamics_model.state_dict(), save_path)

    torch.set_printoptions(profile='full')
    string = str(single_step_sindy_dynamics_model.sindy_coefficients.data)
    with open('sindy_coefficients_polyorder3.txt','w') as fp:
        fp.write(string)
    print('THRESHOLDING: %d active coefficients' % torch.sum(coefficient_mask))
