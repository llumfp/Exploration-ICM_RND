import random
import gymnasium as gym
#import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import pickle


from exploration import DummyIntrinsicRewardModule, RNDNetwork, ICMNetwork
from helper import episode_reward_plot

class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.buffer = [None] * capacity

        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def put(self, obs, action, reward, next_obs, truncated, terminated):
        """Put a tuple of (obs, action, rewards, next_obs, done) into the replay buffer.
        The max length specified by capacity should never be exceeded. 
        The oldest elements inside the replay buffer should be overwritten first.
        """
        self.buffer[self.ptr] = (obs, action, reward, next_obs, truncated, terminated)

        self.size = min(self.size + 1, self.capacity)
        self.ptr = (self.ptr + 1) % self.capacity

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer.
        Should return 5 lists of, each for every attribute stored (i.e. obs_lst, action_lst, ....)
        """
        return zip(*random.sample(self.buffer[: self.size], batch_size))

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        return self.size


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)


class DQN:
    """The DQN method."""

    def __init__(
        self,
        env,
        replay_size=20000,
        batch_size=32,
        gamma=0.99,
        sync_after=5,
        lr=0.03,
        verbose=False,
        reward_module=None,
        render=False,
    ):
        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError("Continuous actions not implemented!")

        # Some variables
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose
        self.render = render

        # Initialize DQN network
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim)
        # Initialize DQN target network, load parameters from DQN network
        self.dqn_target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        # Set up optimizer, only needed for DQN network
        self.optim_dqn = optim.RMSprop(self.dqn_net.parameters(), lr=lr)

        # Initialize reward module
        if reward_module == "RND":
            self.intrinsic_reward_module = RNDNetwork(self.obs_dim, 128)
            self.optim_reward = optim.RMSprop(
                self.intrinsic_reward_module.parameters(), lr=lr,
            )
        elif reward_module == "ICM":
            self.intrinsic_reward_module = ICMNetwork(self.obs_dim, 256, self.act_dim)
            self.optim_reward = optim.RMSprop(
                self.intrinsic_reward_module.parameters(), lr=lr / 50.0
            )
        else:
            # This module only has a 'calculate_reward(...)' method which returns 0.0
            # Used for vanilla DQN
            self.intrinsic_reward_module = DummyIntrinsicRewardModule()

    def learn(self, time_steps):
        # We use them for our reward plots
        lrval = []
        episode_length = 0
        intrinsic_rewards = []
        episode_lengths = []
        intrinsic_episode_rewards = []

        obs,_ = self.env.reset()
        # Save best episode reward so far here
        min_episode_len = 1337
        for timestep in range(1, time_steps + 1):
            if self.render and timestep % 15 == 0:
                # Render every 15th frame to save resources
                self.env.render()
                #pass

            epsilon = epsilon_by_timestep(timestep)
            # Note: epsilon is only used for vanilla DQN
            action = self.predict(obs, epsilon)

            # Do environment step
            next_obs, extrinsic_reward, terminated, truncated,  _ = self.env.step(action)
            done = truncated or terminated

            # Compute additional intrinsic reward
            intrinsic_reward = self.intrinsic_reward_module.calculate_reward(
                torch.Tensor(obs).unsqueeze(0),
                torch.Tensor(next_obs).unsqueeze(0),
                torch.Tensor([action]).unsqueeze(0),
            ).item()

            # Sum them up
            reward = extrinsic_reward + intrinsic_reward

            # Add to replay buffer
            self.replay_buffer.put(obs, action, reward, next_obs, truncated, terminated)
            obs = next_obs

            # Logging
            intrinsic_episode_rewards.append(intrinsic_reward)
            episode_length += 1
            if done:
                obs, _ = self.env.reset()
                intrinsic_rewards.append(sum(intrinsic_episode_rewards))
                if (
                    self.verbose
                    and min_episode_len > episode_length
                    and episode_length < 200
                ):
                    min_episode_len = episode_length
                    print(f"[t={timestep}]: Solved after {min_episode_len}!")
                episode_lengths.append(episode_length)
                episode_length = 0
                intrinsic_episode_rewards = []

            if len(self.replay_buffer) > self.batch_size:
                # Update
                # Get data from replay buffer
                obs_, actions, rewards, next_obs_, truncateds, terminateds = self.replay_buffer.get(
                    self.batch_size
                )
                # Convert to Tensors
                obs_ = torch.stack([torch.Tensor(ob) for ob in obs_])
                next_obs_ = torch.stack(
                    [torch.Tensor(next_ob) for next_ob in next_obs_]
                )
                rewards = torch.Tensor(rewards)
                truncateds = torch.Tensor(truncateds)
                terminateds = torch.Tensor(terminateds)
                # Has to be torch.LongTensor in order to being able to use as index for torch.gather()
                actions = torch.LongTensor(actions)

                # Update DQN
                dqn_loss = self.compute_msbe_loss(
                    obs_, actions, rewards, next_obs_, truncateds, terminateds
                )
                self.optim_dqn.zero_grad()
                dqn_loss.backward()
                self.optim_dqn.step()

                # Update reward module
                # Note: We don't do this in case of vanilla DQN, thus the isinstance(...) check
                if not isinstance(
                    self.intrinsic_reward_module, DummyIntrinsicRewardModule
                ):
                    # Update reward module
                    intrinsic_loss = self.intrinsic_reward_module.calculate_loss(
                        obs_, next_obs_, actions
                    )
                    self.optim_reward.zero_grad()
                    intrinsic_loss.backward()
                    self.optim_reward.step()

            if timestep % self.sync_after == 0:
                # Update target network
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

            if timestep % 1000 == 0 and len(episode_lengths) >= 7:
                #print(timestep, episode_lengths[-7:])
                rval=self.test_policy_10()
                lrval.append(rval)
                print(' ',timestep, rval,end='')
                # Comentar perquè era un intent de gràfic, no definitiu
                # # Plot
                # episode_reward_plot(
                #     episode_lengths,
                #     intrinsic_rewards,
                #     timestep,
                #     window_size=7,
                #     step_size=1,
                # )
        return lrval, (episode_lengths, intrinsic_rewards)

    def test_policy_10(self):
        """Tests the policy for 100 episodes."""
        env = deepcopy(self.env)
        time = []
        for i in range(100):
            obs,_ = env.reset()
            done = False
            t=0
            while not done:
                # Note: epsilon is only used for vanilla DQN
                action = self.predict(obs, 0)
                t=t+1
                # Do environment step
                obs, extrinsic_reward, terminated, truncated,  _ = env.step(action)
                done = truncated or terminated
            time.append(t)
        return np.array(time).mean()

    def predict(self, state, epsilon=0.0):
        # We turn off epsilon greedy when using an intrinsic reward module
        #### ERROR? : original had "or" instead of "or not"
        if random.random() > epsilon or  isinstance(
            self.intrinsic_reward_module, DummyIntrinsicRewardModule
        ):
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.dqn_net.forward(state)
            action = q_value.argmax().item()
        else:
            action = random.randrange(self.act_dim)
        return action

    def compute_msbe_loss(self, obs, actions, rewards, next_obs, truncateds, terminateds):
        # Compute q_values and next_q_values
        q_values = self.dqn_net(obs)
        next_q_values = self.dqn_target_net(next_obs)
        # Select Q-values of actions actually taken
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Calculate max over next Q-values
        next_q_values = next_q_values.max(1)[0]
        # The target we want to update our network towards
        dones = truncateds + terminateds - truncateds * terminateds
#        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - terminateds)
        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)
        # Calculate DQN loss
        dqn_loss = F.mse_loss(q_values, expected_q_values)

        return dqn_loss


# def epsilon_by_timestep(
#     timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000
# ):
#     """Linearly decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps."""
#     return max(
#         epsilon_final,
#         epsilon_start - (timestep / frames_decay) * (epsilon_start - epsilon_final),
#     )
def epsilon_by_timestep(
    timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000
):
    """Linearly decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps."""
    return 0.2


def test_policy_100(env,dqn):
    """Tests the policy for 100 episodes."""
    time = []
    for i in range(100):
        obs,_ = env.reset()
        done = False
        t=0
        while not done:
            # Note: epsilon is only used for vanilla DQN
            action = dqn.predict(obs, 0)
            t=t+1
            # Do environment step
            obs, extrinsic_reward, terminated, truncated,  _ = env.step(action)
            done = truncated or terminated
        time.append(t)
    return np.array(time).mean()


if __name__ == "__main__":
    # Switch this to 'True' for realtime rendering during training
    # Note: This might slow down training a bit
    render = False

    plt.ion()
    # Vanilla DQN

    #Uncomment this to use the default MountainCar environment
    #See what happens
    print('Original MountainCar')
    ll=[]
    for i in range(1): # Podem fer 3 execucions
        env = gym.make('MountainCar-v0', render_mode="human") if render else gym.make('MountainCar-v0')
        dqn = DQN(env, verbose=True, render=render)
        l1,_ = dqn.learn(50000)
        ll.append(l1)
        print(' Mean: ',test_policy_100(env,dqn))
    plt.show()
    plt.plot(np.array(ll).mean(axis=0))
    #3fig2,axes2=plt.subplots(1)
    plt.title('Original MountainCar')
    plt.xlabel('Time steps')
    plt.ylabel('Episode length')
    plt.grid()
    #plt.subplot(4,1,1)
    #axes2.plot(np.array(ll).mean(axis=0))
    plt.ylim((0, 210)) 
    plt.savefig('plots/original.png')
    plt.show()
    print(ll)
    file = open('models/original', 'wb')
    pickle.dump(ll, file)
    file.close()

    # Entorn costumitzats per tenir funció de reforç esparsa, però NO informativa
    from env import MountainCarCustomized
    env = MountainCarCustomized()
    print('Customized MountainCar sparsified with epsilon exploration')
    ll2=[]
    for i in range(1): # Podem fer 3 execucions
        env = MountainCarCustomized()
        dqn = DQN(env, verbose=True, render=render)
        l1,_ = dqn.learn(50000)
        ll2.append(l1)
    print(' Mean: ',test_policy_100(env,dqn))
    plt.show()
    plt.title('Customized MountainCar with epsilon exploration')
    plt.xlabel('Time steps')
    plt.ylabel('Episode length')
    plt.grid()
    plt.plot(np.array(ll2).mean(axis=0))
    plt.ylim((0, 210)) 
    plt.savefig('plots/customized-eps.png')
    plt.show()
    print(ll2)
    file = open('models/Cust-eps', 'wb')
    pickle.dump(ll2, file)
    file.close()

    # DQN + ICM
    print('Customized MountainCar sparsified with ICM')
    ll4=[]
    for i in range(1): # Podem fer 3 execucions
        env = MountainCarCustomized()
        dqn = DQN(env, verbose=True, reward_module="ICM", render=render)
        l1,_ = dqn.learn(50000)
        ll4.append(l1)
        print(' Mean: ',test_policy_100(env,dqn))
    plt.plot(np.array(ll4).mean(axis=0))
    plt.title('Customized MountainCar with ICM')
    plt.xlabel('Time steps')
    plt.ylabel('Episode length')
    plt.grid()
    plt.ylim((0, 210)) 
    plt.savefig('plots/Customized-ICM.png')
    plt.show()
    print(ll4)
    file = open('models/Cust-ICM', 'wb')
    pickle.dump(ll4, file)
    file.close()

    # DQN + RND
    print('Customized MountainCar sparsified RND exploration')
    ll3=[]
    for i in range(1): # Podem fer 3 execucions
        env = MountainCarCustomized()
        dqn = DQN(env, verbose=True, reward_module="RND", render=render)
        l1,_ = dqn.learn(50000)
        ll3.append(l1)
    print(' Mean: ',test_policy_100(env,dqn))
    plt.plot(np.array(ll3).mean(axis=0))
    plt.title('Customized MountainCar with RND')
    plt.xlabel('Time steps')
    plt.ylabel('Episode length')
    plt.grid()
    plt.ylim((0, 210)) 
    plt.savefig('plots/customized-RND.png')
    plt.show()
    print(ll3)
    file = open('models/Cust-RND', 'wb')
    pickle.dump(ll3, file)
    file.close()



