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
import warnings
warnings.filterwarnings("ignore")

def test(model, val_loader, loss_fn,coefficient_mask=None) -> float:
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
        loss = loss_fn(model, states, actions,coefficient_mask)
        val_loss += loss.item()
    return val_loss/len(val_loader)

if __name__ == "__main__":
    LATENT_DIM = 16
    ACTION_DIM = 3
    POLY_ORDER = 2
    LIBRARY_DIM = library_size(LATENT_DIM, POLY_ORDER, True, True)
    coefficient_mask = torch.ones(LIBRARY_DIM,LATENT_DIM)
    NUM_CHANNELS = 1

    collected_data = np.load('pushing_image_data.npy', allow_pickle=True)
    train_loader, val_loader, norm_constants = process_data_multiple_step(collected_data, batch_size=500, num_steps=1)
    norm_tr = NormalizationTransform(norm_constants)

    single_step_sindy_dynamics_model = SINDyModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, poly_order=POLY_ORDER, include_sine=True, sequential_thresholding=False, num_channels=NUM_CHANNELS)
    model_path = 'single_step_sindy_dynamics_model_polyorder2.pt'
    single_step_sindy_dynamics_model.load_state_dict(torch.load(model_path))

    #Test accuracy
    test_data = np.load('pushing_image_validation_data.npy', allow_pickle=True)
    single_step_dataset = MultiStepDynamicsDataset(test_data, num_steps=1, transform=norm_tr)
    single_step_loader = DataLoader(single_step_dataset, batch_size=len(single_step_dataset))

    state_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()
    loss = MultiStepLoss(state_loss_fn, latent_loss_fn, lambda_state_loss=0.1, lambda_latent_loss=0.1,lambda_reg_loss=0.3)
    print('The file will take 2 to 5 minutes to run.')
    print(f'Single-step model evaluated on single-step loss: {test(single_step_sindy_dynamics_model, single_step_loader, loss,coefficient_mask)}')
    print('')

    #show reconstruction
    traj = collected_data[0]
    evaluate_model_plot(single_step_sindy_dynamics_model,traj, norm_tr,coefficient_mask)

    # print(single_step_sindy_dynamics_model.sindy_coefficients.data)
    torch.set_printoptions(profile='full')
    string = str(single_step_sindy_dynamics_model.sindy_coefficients.data)
    with open('sindy_coefficients.txt','w') as fp:
        fp.write(string)
    print("sindy coefficients are written to text file: sindy_coefficients")


    ##run contoller
    target_state = np.array([0.7, 0., 0.])

    env = PandaImageSpacePushingEnv(visualizer=None, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True)
    state_0 = env.reset()
    env.object_target_pose = env._planar_pose_to_world_pose(target_state)
    controller = PushingLatentController(env, single_step_sindy_dynamics_model, latent_space_pushing_cost_function,norm_constants, num_samples=100, horizon=10)
    # controller = PushingImgSpaceController(env, single_step_sindy_dynamics_model, img_space_pushing_cost_function, norm_constants, num_samples=100, horizon=10)

    state = state_0

    # num_steps_max = 100
    num_steps_max = 20

    for i in range(num_steps_max):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        # check if we have reached the goal
        end_pose = env.get_object_pos_planar()
        goal_distance = np.linalg.norm(end_pose[:2]-target_state[:2]) # evaluate only position, not orientation
        goal_reached = goal_distance < BOX_SIZE
        if done or goal_reached:
            print('Total steps to reach goal:',i)
            break

    print(f'GOAL REACHED: {goal_reached}')
            
    # plt.close(fig)