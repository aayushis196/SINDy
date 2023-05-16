import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from mppi import MPPI
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE
import control
from scipy.special import binom

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random_trajectory(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, 32, 32, num_channels) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    num_channels = 1;
    collected_data = [];
    #print(env.action_space);
    for i in range(num_trajectories):
      states = np.empty(shape=(trajectory_length+1,32,32,num_channels), dtype=np.uint8);
      actions = np.empty(shape=(trajectory_length,env.action_space.shape[0]), dtype=np.float32);
      env.reset()
      states[0] = env.observation_space.sample();
      for j in range(trajectory_length):
        action_j = env.action_space.sample()
        stat, rew, done, inf = env.step(action_j);
        actions[j] = action_j;
        states[j+1] = stat;
        # if done:
        #   env.reset()

      print("Finished:",i,"/",num_trajectories)
      #states = np.float32(states)
      #actions = np.float32(actions)
      dict_data = {'states':  states,
              'actions': actions};
      collected_data.append(dict_data)


    # ---
    return collected_data


class NormalizationTransform(object):

    def __init__(self, norm_constants):
        self.norm_constants = norm_constants
        self.mean = norm_constants['mean']
        self.std = norm_constants['std']

    def __call__(self, sample):
        """
        Transform the sample by normalizing the 'states' using the provided normalization constants.
        :param sample: dictionary containing {'states', 'actions'}
        :return:
        """
        # --- Your code here
        sample['states'] = self.normalize_state(sample['states']) 


        # ---
        return sample

    def inverse(self, sample):
        """
        Transform the sample by de-normalizing the 'states' using the provided normalization constants.
        :param sample: dictionary containing {'states', 'actions'}
        :return:
        """
        # --- Your code here
        sample['states'] = self.denormalize_state(sample['states']) 


        # ---
        return sample

    def normalize_state(self, state):
        """
        Normalize the state using the provided normalization constants.
        :param state: <torch.tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.tensor> of shape (..., num_channels, 32, 32)
        """
        # --- Your code here
        state = (state -self.mean )/ self.std


        # ---
        return state

    def denormalize_state(self, state_norm):
        """
        Denormalize the state using the provided normalization constants.
        :param state_norm: <torch.tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.tensor> of shape (..., num_channels, 32, 32)
        """
        # --- Your code here
        state = state_norm*self.std + self.mean


        # ---
        return state


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'states': x_t,x_{t+1}, ... , x_{t+num_steps}
     'actions': u_t, ..., u_{t+num_steps-1},
    }
    where:
     states: torch.float32 tensor of shape (batch_size, num_steps+1, state_size)
     actions: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.

    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.

    :return train_loader: <torch.utils.data.DataLoader> for training
    :return val_loader: <torch.utils.data.DataLoader> for validation
    :return normalization_constants: <dict> containing the mean and std of the states.

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_data = None
    val_data = None
    normalization_constants = {
        'mean': None,
        'std': None,
    }
    # Your implemetation needs to do the following:
    #  1. Initialize dataset
    #  2. Split dataset,
    #  3. Estimate normalization constants for the train dataset states.
    # --- Your code here
    dataset = MultiStepDynamicsDataset(collected_data,num_steps=num_steps)
    val_set_len = 0.2
    train_data, val_data = random_split(dataset, [int((1-val_set_len)*len(dataset)), int(val_set_len*len(dataset))])
    state_tensor = train_data[0]['states'].to(dtype=torch.float32)
    # print(state_tensor.shape)
    for i in range(len(train_data)-1):
      sample = train_data[i+1]
      state = sample['states'].to(dtype=torch.float32)
      state_tensor = torch.cat([state_tensor, state],dim=0)
    # print(state_tensor.shape)
    normalization_constants['mean'] = torch.mean(state_tensor)
    normalization_constants['std'] = torch.std(state_tensor)
    # print(normalization_constants['mean'])
    # print(normalization_constants['std'])
    # ---
    norm_tr = NormalizationTransform(normalization_constants)
    train_data.dataset.transform = norm_tr
    val_data.dataset.transform = norm_tr

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)


    return train_loader, val_loader, normalization_constants


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.
    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'states':[x_{t}, x_{t+1},..., x_{t+num_steps} ] -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'actions': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
    }

    Observation: If num_steps=1, this dataset is equivalent to SingleStepDynamicsDataset.
    """

    def __init__(self, collected_data, num_steps=4, transform=None):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps
        self.transform = transform

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (states, actions).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'states': None,
            'actions': None,
        }
        # --- Your code here       
        traj_index = item // self.trajectory_length
        state_index = item % self.trajectory_length
        
        sample['states'] = torch.from_numpy(self.data[traj_index]["states"][state_index:state_index + self.num_steps +1])
        sample['actions'] = torch.from_numpy(self.data[traj_index]["actions"][state_index:state_index + self.num_steps])
        
        if self.transform is not None:
            sample = self.transform(sample) 

        sample['states'] = sample['states'].to(torch.float32)
        sample['actions'] = sample['actions'].to(torch.float32)
        sample['states'] = sample['states'].permute(0,3,1,2)
        # ---
        return sample


class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # Weight of the KL divergence term

    def forward(self, x_hat, x, mu, logvar):
        """
        Compute the VAE loss.
        vae_loss = MSE(x, x_hat) + beta * KL(N(\mu, \sigma), N(0, 1))
        where KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param x: <torch.tensor> ground truth tensor of shape (batch_size, state_size)
        :param x_hat: <torch.tensor> reconstructed tensor of shape (batch_size, state_size)
        :param mu: <torch.tensor> of shape (batch_size, state_size)
        :param logvar: <torch.tensor> of shape (batch_size, state_size)
        :return: <torch.tensor> scalar loss
        """
        loss = None
        # --- Your code here

        KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = torch.nn.functional.mse_loss(x,x_hat) + self.beta *KL_loss


        # ---
        return loss


class MultiStepLoss(nn.Module):
    def __init__(self, state_loss_fn, latent_loss_fn, lambda_state_loss=0.1, lambda_latent_loss=0.2,lambda_reg_loss=0.5):
        super().__init__()
        self.state_loss = state_loss_fn
        self.latent_loss = latent_loss_fn
        self.lambda_state_loss = lambda_state_loss
        self.lambda_latent_loss = lambda_latent_loss
        self.lambda_reg_loss = lambda_reg_loss

    def forward(self, model, states, actions,coefficient_mask=None):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        Here the loss is computed based on 3 terms:
         - reconstruction loss: enforces good encoding and reconstruction of the states (no dynamics).
         - latent loss: enforces dynamics on latent space by matching the state encodings with the dynamics on latent space.
         - prediction loss: enforces reconstruction of the predicted states by matching the predictions with the targets.

         :param model: <nn.Module> model to be trained.
         :param states: <torch.tensor> tensor of shape (batch_size, traj_size + 1, state_size)
         :param actions: <torch.tensor> tensor of shape (batch_size, traj_size, action_size)
        """
        # compute reconstruction loss -- compares the encoded-decoded states with the original states
        rec_loss = 0.
        # --- Your code here
        latent_values = model.encode(states)
        decoded_states = model.decode(latent_values)
        rec_loss = self.state_loss(decoded_states, states)
        # ---
        # propagate dynamics on latent space as well as reconstructed states
        pred_latent_values = []
        pred_states = []
        prev_z = latent_values[:, 0, :]  # get initial latent value
        prev_state = states[:, 0, :]  # get initial state value
        for t in range(actions.shape[1]):
            next_z = None
            next_state = None
            # --- Your code here
            next_z = model.sindy_dynamics(prev_z, actions[:,t,:],coefficient_mask)
            next_state = model(prev_state,actions[:,t,:],coefficient_mask)

            pred_latent_values.append(next_z)
            pred_states.append(next_state)


            # ---
            prev_z = next_z
            prev_state = next_state
        pred_states = torch.stack(pred_states, dim=1)
        pred_latent_values = torch.stack(pred_latent_values, dim=1)
        # compute prediction loss -- compares predicted state values with the given states
        pred_loss = 0.
        # --- Your code here
        pred_loss = self.state_loss(pred_states, states[:,1:,:,:,:])


        # ---

        # compute latent loss -- compares predicted latent values with the encoded latent values for states
        lat_loss = 0.
        # --- Your code here

        lat_loss = self.latent_loss(pred_latent_values, latent_values[:,1:,:])

        # ---
        # regularization loss for sindy coefficients
        sindy_reg_loss = torch.mean(torch.abs(model.sindy_coefficients))
        multi_step_loss = rec_loss + self.lambda_state_loss*pred_loss + self.lambda_latent_loss * lat_loss + self.lambda_reg_loss*sindy_reg_loss

        return multi_step_loss


class StateEncoder(nn.Module):
    """
    Embeds the state into a latent space.
    State shape: (..., num_channels, 32, 32)
    latent shape: (..., latent_dim)
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        # --- Your code here
        self.encoder= torch.nn.Sequential(
            torch.nn.Conv2d(self.num_channels,4, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(4,4,kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Flatten(),
            torch.nn.Linear(100,100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(100,self.latent_dim)
          )
        # ---

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return latent_state: <torch.Tensor> of shape (..., latent_dim)
        """
        latent_state = None
        input_shape = state.shape
        state = state.reshape(-1, self.num_channels, 32, 32)
        # --- Your code here
        latent_state = self.encoder(state)


        # ---
        # convert to original multi-batch shape
        latent_state = latent_state.reshape(*input_shape[:-3], self.latent_dim)
        return latent_state


class StateVariationalEncoder(nn.Module):
    """
    Embeds the state into a latent space.
    State shape: (..., num_channels, 32, 32)
    latent shape: (..., latent_dim)
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        # --- Your code here
        self.vae= torch.nn.Sequential(
            torch.nn.Conv2d(self.num_channels,4, kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(4,4,kernel_size=(5,5), stride=(1,1), padding=(0,0)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Flatten(),
            torch.nn.Linear(100,100),
            torch.nn.ReLU(inplace=True)
          )
        self.mulinear = torch.nn.Linear(100, self.latent_dim) 
        self.varlinear = torch.nn.Linear(100, self.latent_dim)   


        # ---

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return: 2 <torch.Tensor>
          :mu: <torch.Tensor> of shape (..., latent_dim)
          :log_var: <torch.Tensor> of shape (..., latent_dim)
        """
        mu = None
        log_var = None
        input_shape = state.shape
        state = state.reshape(-1, self.num_channels, 32, 32)
        # --- Your code here
        latent_state = self.vae(state)
        mu = self.mulinear(latent_state)
        log_var = self.varlinear(latent_state)
        # ---
        # convert to original multi-batch shape
        mu = mu.reshape(*input_shape[:-3], self.latent_dim)
        log_var = log_var.reshape(*input_shape[:-3], self.latent_dim)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        """
        Reparametrization trick to sample from N(mu, std) from N(0,1)
        :param mu: <torch.Tensor> of shape (..., latent_dim)
        :param logvar: <torch.Tensor> of shape (..., latent_dim)
        :return: <torch.Tensor> of shape (..., latent_dim)
        """
        sampled_latent_state = None
        # --- Your code here
        std = torch.exp(logvar)
        std = torch.sqrt(std)
        eps = torch.randn(std.shape)
        sampled_latent_state = mu + (eps*std)

        # ---
        return sampled_latent_state


class StateDecoder(nn.Module):
    """
    Reconstructs the state from a latent space.
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        # --- Your code here
        self.decoder= torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim,500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500,500),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(500,self.num_channels*32*32)
          )


        # ---

    def forward(self, latent_state):
        """
        :param latent_state: <torch.Tensor> of shape (..., latent_dim)
        :return decoded_state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        """
        decoded_state = None
        input_shape = latent_state.shape
        latent_state = latent_state.reshape(-1, self.latent_dim)

        # --- Your code here
        decoded_state = self.decoder(latent_state)


        # ---

        decoded_state = decoded_state.reshape(*input_shape[:-1], self.num_channels, 32, 32)

        return decoded_state


class StateVAE(nn.Module):
    """
    State AutoEncoder
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.encoder = StateVariationalEncoder(latent_dim, num_channels)
        self.decoder = StateDecoder(latent_dim, num_channels)

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return:
            reconstructed_state: <torch.Tensor> of shape (..., num_channels, 32, 32)
            mu: <torch.Tensor> of shape (..., latent_dim)
            log_var: <torch.Tensor> of shape (..., latent_dim)
            latent_state: <torch.Tensor> of shape (..., latent_dim)
        """
        reconstructed_state = None # decoded states from the latent_state
        mu, log_var = None, None # mean and log variance obtained from encoding state
        latent_state = None # sample from the latent space feeded to the decoder
        # --- Your code here
        mu, log_var = self.encoder(state)
        latent_state = self.reparameterize(mu,log_var)
        reconstructed_state = self.decoder(latent_state)

        # ---
        return reconstructed_state, mu, log_var, latent_state

    def encode(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.Tensor> of shape (..., latent_dim)
        """
        latent_state = None
        # --- Your code here
        mu, log_var = self.encoder(state)
        latent_state = self.reparameterize(mu,log_var)


        # ---
        return latent_state

    def decode(self, latent_state):
        """
        :param latent_state: <torch.Tensor> of shape (..., latent_dim)
        :return: <torch.Tensor> of shape (..., num_channels, 32, 32)
        """
        reconstructed_state = None
        # --- Your code here
        reconstructed_state = self.decoder(latent_state)


        # ---
        return reconstructed_state

    def reparameterize(self, mu, logvar):
        return self.encoder.reparameterize(mu, logvar)


class SINDyModel(nn.Module):
    """
    Model the dynamics in latent space via residual learning z_{t+1} = z_{t} + f(z_{t},a_{t})
    Use StateEncoder and StateDecoder encoding-decoding the state into latent space.
    where
        z_{t}  = encoder(x_{t})
        z_{t+1} = sindy_model(z_{t}, a_{t})
        x_{t+1} = decoder(z_{t+1})

    Latent dynamics model must be a Linear 3-layer network with 100 units in each layer and ReLU activations.
    The input to the latent_dynamics_model must be the latent states and actions concatentated along the last dimension.
    """

    def __init__(self, latent_dim, action_dim, poly_order, include_sine, sequential_thresholding, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_channels = num_channels
        self.poly_order = poly_order
        self.include_sine= include_sine
        self.sequential_thresholding = sequential_thresholding
        self.library_dim = library_size(self.latent_dim, self.poly_order, self.include_sine, True)
        self.sindy_coefficients = nn.Parameter(torch.ones(self.library_dim,self.latent_dim))

        self.encoder = StateEncoder(self.latent_dim, self.num_channels) 
        self.decoder = StateDecoder(self.latent_dim, self.num_channels)
        # ---

    def forward(self, state, action, coefficient_mask=None):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., num_channels, 32, 32)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., num_channels, 32, 32)
        """
        next_state = None
        # --- Your code here
        latent_state = self.encode(state)
        next_latent_state = self.sindy_dynamics(latent_state, action,coefficient_mask)
        next_state = self.decode(next_latent_state)

        # ---
        return next_state

    def encode(self, state):
        """
        Encode a state into the latent space
        :param state: torch tensor of shape (..., num_channels, 32, 32)
        :return: latent_state: torch tensor of shape (..., latent_dim)
        """
        latent_state = None
        # --- Your code here

        latent_state = self.encoder(state)

        # ---
        return latent_state

    def decode(self, latent_state):
        """
        Decode a latent state into the original space.
        :param latent_state: torch tensor of shape (..., latent_dim)
        :return: state: torch tensor of shape (..., num_channels, 32, 32)
        """
        state = None
        # --- Your code here
        state = self.decoder(latent_state)


        # ---
        return state

    def sindy_dynamics(self, latent_state, action,coefficient_mask=None):
        """
        Compute the dynamics in latent space
        z_{t+1} = z_{t} + latent_dynamics_model(z_{t}, a_{t})
        :param latent_state: torch tensor of shape (..., latent_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_latent_state: torch tensor of shape (..., latent_dim)
        """
        next_latent_state = None
        # --- Your code here

        #print("Hello 0")
        # add z_derivative????????????????????????????????
        next_latent_state = torch.cat([latent_state, action], dim=1)
        theta = sindy_library(next_latent_state,self.latent_dim,self.poly_order,self.include_sine)
        if coefficient_mask is not None:
            next_latent_state = torch.matmul(theta, coefficient_mask*self.sindy_coefficients)
        else:
            next_latent_state = torch.matmul(theta,self.sindy_coefficients)

        # ---
        return next_latent_state


class LatentDynamicsModel(nn.Module):
    """
    Model the dynamics in latent space via residual learning z_{t+1} = z_{t} + f(z_{t},a_{t})
    Use StateEncoder and StateDecoder encoding-decoding the state into latent space.
    where
        z_{t}  = encoder(x_{t})
        z_{t+1} = z_{t} + latent_dynamics_model(z_{t}, a_{t})
        x_{t+1} = decoder(z_{t+1})

    Latent dynamics model must be a Linear 3-layer network with 100 units in each layer and ReLU activations.
    The input to the latent_dynamics_model must be the latent states and actions concatentated along the last dimension.
    """

    def __init__(self, latent_dim, action_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_channels = num_channels
        self.encoder = None
        self.decoder = None
        self.latent_dynamics_model = None
        # --- Your code here
        self.encoder = StateEncoder(latent_dim, num_channels)
        self.decoder = StateDecoder(latent_dim, num_channels)
        self.latent_dynamics_model =  torch.nn.Sequential(
            torch.nn.Linear(latent_dim+action_dim,100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(100,100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(100,self.latent_dim)
          )

        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., num_channels, 32, 32)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., num_channels, 32, 32)
        """
        next_state = None
        # --- Your code here
        latent_state = self.encode(state)
        next_latent_state = self.latent_dynamics(latent_state, action)
        next_state = self.decode(next_latent_state)
        # ---
        return next_state

    def encode(self, state):
        """
        Encode a state into the latent space
        :param state: torch tensor of shape (..., num_channels, 32, 32)
        :return: latent_state: torch tensor of shape (..., latent_dim)
        """
        latent_state = None
        # --- Your code here
        latent_state = self.encoder(state)


        # ---
        return latent_state

    def decode(self, latent_state):
        """
        Decode a latent state into the original space.
        :param latent_state: torch tensor of shape (..., latent_dim)
        :return: state: torch tensor of shape (..., num_channels, 32, 32)
        """
        state = None
        # --- Your code here
        state = self.decoder(latent_state)


        # ---
        return state

    def latent_dynamics(self, latent_state, action):
        """
        Compute the dynamics in latent space
        z_{t+1} = z_{t} + latent_dynamics_model(z_{t}, a_{t})
        :param latent_state: torch tensor of shape (..., latent_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_latent_state: torch tensor of shape (..., latent_dim)
        """
        next_latent_state = None
        # --- Your code here
        latent_input = torch.cat([latent_state, action], dim=1)
        next_latent_state =latent_state +  self.latent_dynamics_model(latent_input)


        # ---
        return next_latent_state


def latent_space_pushing_cost_function(latent_state, action, target_latent_state):
    """
    Compute the state cost for MPPI on a setup without obstacles in latent space.
    :param state: torch tensor of shape (B, latent_dim)
    :param action: torch tensor of shape (B, action_size)
    :param target_latent_state: torch tensor of shape (latent_dim,)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """ 
    cost = None
    # --- Your code here
    target_latent_state_stack = target_latent_state.repeat(latent_state.shape[0],1)
    cost = torch.matmul((latent_state - target_latent_state), (latent_state - target_latent_state).T)
    cost = torch.diagonal(cost)

    # ---
    return cost


def img_space_pushing_cost_function(state, action, target_state):
    """
    Compute the state cost for MPPI on a setup without obstacles in state space (images).
    :param state: torch tensor of shape (B, w, h, num_channels)
    :param action: torch tensor of shape (B, action_size)
    :param target_state: torch tensor of shape (w, h, num_channels)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    cost = None
    # --- Your code here
    target_state_stack = target_state.unsqueeze(0).repeat(state.shape[0],1,1,1)
    cost = torch.mean((state-target_state)**2,dim=(1,2,3))

    # ---
    return cost


class PushingImgSpaceController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, norm_constants, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.norm_constants = norm_constants
        self.target_state = torch.as_tensor(self.env.get_target_state(), dtype=torch.float32).permute(2, 0, 1)
        self.target_state_norm = (self.target_state - self.norm_constants['mean']) / self.norm_constants['std']
        self.cost_function = cost_function
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.1 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        self.mppi = MPPI(self._compute_dynamics,
                         self._compute_costs,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, wrapped_state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, wrapped_state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        unwrap_state = self._unwrap_state(state)
        next_state = self.model(unwrap_state, action)
        next_state = self._wrap_state(next_state)
        # ---
        return next_state

    def _compute_costs(self, state, action):
        """
        Compute the cost for each state-action pair.
        You need to call self.cost_function to compute the cost.
        :param state: torch tensor of shape (B, wrapped_state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
        """
        cost = None
        # --- Your code here
        state = self._unwrap_state(state)
        cost = self.cost_function(state, action, self.target_state_norm)


        # ---
        return cost

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (height, width, num_channels) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be sent to the mppi controller. Note that MPPI works with torch tensors.
         - You may need to normalize the state to the same space used for training the model.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state).float()
        mean = self.norm_constants['mean']
        std = self.norm_constants['std']
        state_tensor = (state_tensor - mean) / std
        state_tensor = state_tensor.permute(2,0,1)
        state_tensor = self._wrap_state(state_tensor)

        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        action = action_tensor.detach().numpy()


        # ---
        return action

    def _wrap_state(self, state):
        # convert state from shape (..., num_channels, height, width) to shape (..., num_channels*height*width)
        wrapped_state = None
        # --- Your code here
        # print(state.shape)
        B = state.shape[:-3]
        N = state.shape[-3]
        H = state.shape[-2]
        W = state.shape[-1]
        wrapped_state = state.reshape(*B,N*H*W)

        # ---
        return wrapped_state

    def _unwrap_state(self, wrapped_state):
        # convert state from shape (..., num_channels*height*width) to shape (..., num_channels, height, width)
        state = None
        # --- Your code here
        B = wrapped_state.shape[:-1]
        N = 1
        W = 32
        H = 32
        state = wrapped_state.reshape(*B,N,H,W)

        # ---
        return state


class PushingLatentController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, norm_constants, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.norm_constants = norm_constants
        self.target_state = torch.as_tensor(self.env.get_target_state(), dtype=torch.float32).permute(2, 0, 1)
        self.target_state_norm = (self.target_state - self.norm_constants['mean']) / self.norm_constants['std']
        self.latent_target_state = self.model.encode(self.target_state_norm)
        self.cost_function = cost_function
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = model.latent_dim  # Note that the state size is the latent dimension of the model
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.1 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        self.mppi = MPPI(self._compute_dynamics,
                         self._compute_costs,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action, coefficient_mask=None):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, latent_dim)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, latent_dim) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        # next_state = self.model.latent_dynamics(state, action)
        next_state = self.model.sindy_dynamics(state, action, coefficient_mask)

        # ---
        return next_state

    def _compute_costs(self, state, action):
        """
        Compute the cost for each state-action pair.
        You need to call self.cost_function to compute the cost.
        :param state: torch tensor of shape (B, latent_dim)
        :param action: torch tensor of shape (B, action_size)
        :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
        """
        cost = None
        # --- Your code here
        cost = self.cost_function(state,action,self.latent_target_state)


        # ---
        return cost

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (height, width, num_channels) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be sent to the mppi controller. Note that MPPI works with torch tensors.
         - You may need to normalize the state to the same space used for training the model.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state).float()
        mean = self.norm_constants['mean']
        std = self.norm_constants['std']
        state_tensor = (state_tensor - mean) / std
        state_tensor = state_tensor.permute(2,0,1)
        latent_tensor = self.model.encode(state_tensor)

        # ---
        action_tensor = self.mppi.command(latent_tensor)
        # --- Your code here
        action = action_tensor.detach().numpy()


        # ---
        return action


def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_library(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D torch tensor of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D torch tensor containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = []
    library.append(torch.ones(z.shape[0])) #[tf.ones(tf.shape(z)[0])]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(z[:,i]* z[:,j]) #library.append(torch.matmul(z[:,i], z[:,j])) 

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k]) #library.append(torch.matmul((torch.matmul(z[:,i],z[:,j]),z[:,k]))) #l

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]) #library.append(torch.matmul(torch.matmul((torch.matmul(z[:,i],z[:,j]),z[:,k]),z[:,p])))#

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q]) #library.append(torch.matmul(torch.matmul(torch.matmul((torch.matmul(z[:,i],z[:,j]),z[:,k]),z[:,p])),z[:,q])) #

    if include_sine:
        for i in range(latent_dim):
             library.append(torch.sin(z[:,i])) #library.append(tf.sin(z_combined[:,i]))

    return torch.stack(library, dim=1)