import torch
from torch import nn as nn
from torch.nn import functional as F


class IntrinsicRewardModule(nn.Module):
    """The base class for an intrinsic reward method."""

    def calculate_reward(self, obs, next_obs, actions):
        return NotImplemented

    def calculate_loss(self, obs, next_obs, actions):
        return NotImplemented


class DummyIntrinsicRewardModule(IntrinsicRewardModule):
    """Used as a dummy for vanilla DQN."""

    def calculate_reward(self, obs, next_obs, actions):
        return torch.Tensor([0.0]).unsqueeze(0)


class RNDNetwork(IntrinsicRewardModule):
    """Implementation of Random Network Distillation (RND)"""

    def __init__(self, num_obs, num_out, alpha=0.4):
        super().__init__()

        self.target = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, num_out), nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, num_out), nn.ReLU(),
        )
        self.alpha = alpha

    def calculate_loss(self, obs, next_obs, actions):
        # TODO
        # Calculate target and predictor forward.
        # Make sure to not backpropagate gradients through the target network, i.e.
        #   keep the target network static.
        # Compute MSE loss, i.e. through F.mse_loss(...)
        return NotImplemented

    def calculate_reward(self, obs, next_obs, actions):
        # TODO
        # We calculate the reward as the sum of absolute difference between
        #   target and predictor outputs.
        # Scale them using self.alpha
        # Force them into the interval [0.0, 1.0]
        # Podem fer un clip o escalar perquè no sigui tant gran la diferència entre el reward bàsic i l'intrínsec
        return torch.Tensor([0.0])


class ICMNetwork(IntrinsicRewardModule):
    """Implementation of Intrinsic Curiosity Module (ICM)"""

    def __init__(self, num_obs, num_feature, num_act, alpha=10.0, beta=0.5):
        super().__init__()

        # Xarxa neuronal per obtenir les features importants
        self.feature = nn.Sequential(nn.Linear(num_obs, num_feature), nn.ReLU(), )

        self.inverse_dynamics = nn.Sequential(
            nn.Linear(num_feature * 2, num_act), nn.ReLU(), nn.Softmax(dim=1)
        )

        self.forward_dynamics = nn.Sequential(
            nn.Linear(num_feature + num_act, num_feature), nn.ReLU(),
        )

        self.alpha = alpha
        self.beta = beta
        self.num_actions = num_act
        self.num_feat = num_feature

    def calculate_loss(self, obs, next_obs, actions):
        actions = actions.unsqueeze(1)
        # These are the ground-truth action probabilities
        # I.e. probability of 100% for the action performed
        actions_target = torch.zeros(obs.size()[0], self.num_actions)
        for i, a in enumerate(actions):
            actions_target[i, int(a)] = 1.0

        # TODO
        # Inverse dynamics loss
        # - First, encode the current and next observations through the feature network
        # - Use both of the encodings to predict the action that has been performed
        #       Hint: Have a look at how the input layer sizes are defined
        # - Calculate the cross entropy loss between prediction and ground-truth
        inverse_dynamics_loss = 0.0

        # TODO
        # Forward dynamics loss
        # - Use both the obs features and the one-hot encodings of the performed actions as input
        #       to the forward_dynamics model
        # - Calculate, how much the forward prediction differs from the actual next_obs encoding
        # - Calculate the MSE loss between prediction and ground-truth, multiply by 0.5
        forward_dynamics_loss = 0.0

        # Add up
        loss = (
                1.0 - self.beta
               ) * inverse_dynamics_loss + self.beta * forward_dynamics_loss
        return loss

    def calculate_reward(self, obs, next_obs, actions):
        # One-hot encoding/probability matrix for the performed actions
        actions_one_hot = torch.zeros((obs.size()[0], self.num_actions))
        for i, a in enumerate(actions):
            actions_one_hot[i, int(a)] = 1.0

        # TODO
        # - The reward is defined as the MAE between ground truth and prediction of the forward model
        # - Use L1 loss (MAE) instead of MSE to compute the differences between the two
        # - Multiply by the scaling factor alpha
        reward = 0.0
        
        return reward
