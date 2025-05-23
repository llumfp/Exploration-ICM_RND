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
        with torch.no_grad():
            target_output = self.target(obs)
        predictor_output = self.predictor(obs)
        loss = F.mse_loss(predictor_output, target_output)
        return loss

    def calculate_reward(self, obs, next_obs, actions):
        # TODO
        # We calculate the reward as the sum of absolute difference between
        #   target and predictor outputs.
        # Scale them using self.alpha
        # Force them into the interval [0.0, 1.0]
        # Podem fer un clip o escalar perquè no sigui tant gran la diferència entre el reward bàsic i l'intrínsec. O optar per un clamp.
        with torch.no_grad():
            target_output = self.target(next_obs)
        predictor_output = self.predictor(next_obs)

        # Using MSE for reward, consistent with RND literature for surprise
        reward = F.mse_loss(predictor_output, target_output, reduction='none').mean(dim=-1)
        reward = self.alpha * reward
        reward = torch.clamp(reward, 0.0, 1.0)
        return reward.unsqueeze(-1)


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
        actions_input = actions.unsqueeze(1)
        # These are the ground-truth action probabilities
        # I.e. probability of 100% for the action performed
        actions_target = torch.zeros(obs.size()[0], self.num_actions, device=obs.device)
        for i, a in enumerate(actions_input):
            actions_target[i, int(a)] = 1.0

        # TODO
        # Inverse dynamics loss
        # - First, encode the current and next observations through the feature network
        # - Use both of the encodings to predict the action that has been performed
        #       Hint: Have a look at how the input layer sizes are defined
        # - Calculate the cross entropy loss between prediction and ground-truth
        phi_obs = self.feature(obs)
        phi_next_obs = self.feature(next_obs)

        # Inverse dynamics
        action_pred = self.inverse_dynamics(torch.cat((phi_obs, phi_next_obs), dim=1))
        inverse_dynamics_loss = F.cross_entropy(action_pred, actions_target)

        # TODO
        # Forward dynamics loss
        # - Use both the obs features and the one-hot encodings of the performed actions as input
        #       to the forward_dynamics model
        # - Calculate, how much the forward prediction differs from the actual next_obs encoding
        # - Calculate the MSE loss between prediction and ground-truth, multiply by 0.5
        # Forward dynamics
        # We need to detach phi_obs here as per original ICM paper, to prevent the feature net from predicting its own output
        phi_next_obs_pred = self.forward_dynamics(torch.cat((phi_obs.detach(), actions_target), dim=1)) 
        forward_dynamics_loss = 0.5 * F.mse_loss(phi_next_obs_pred, phi_next_obs.detach()) # also detach target next_obs features

        # Add up
        loss = (1.0 - self.beta) * inverse_dynamics_loss + self.beta * forward_dynamics_loss
        return loss

    def calculate_reward(self, obs, next_obs, actions):
        # One-hot encoding/probability matrix for the performed actions
        actions_one_hot = torch.zeros((obs.size()[0], self.num_actions), device=obs.device)
        for i, a in enumerate(actions):
            actions_one_hot[i, int(a)] = 1.0

        # TODO
        # - The reward is defined as the MAE between ground truth and prediction of the forward model
        # - Use L1 loss (MAE) instead of MSE to compute the differences between the two
        # - Multiply by the scaling factor alpha
        with torch.no_grad(): # Rewards should not contribute to gradients
            phi_obs = self.feature(obs)
            phi_next_obs = self.feature(next_obs)
            
            phi_next_obs_pred = self.forward_dynamics(torch.cat((phi_obs, actions_one_hot), dim=1))
            
            # MAE for reward
            reward = F.l1_loss(phi_next_obs_pred, phi_next_obs, reduction='none').mean(dim=-1)
            reward = self.alpha * reward
        
        return reward.unsqueeze(-1)
