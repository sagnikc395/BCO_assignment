import gymnasium as gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device("cpu")


# This will be useful for implementing BCO
def collect_random_interaction_data(num_iters):
    states = []
    next_states = []
    actions = []

    env = gym.make("MountainCar-v0")

    for i in range(num_iters):
        obs, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            states.append(obs)
            next_states.append(next_obs)
            actions.append(a)
            obs = next_obs

    env.close()

    return np.array(states), np.array(next_states), np.array(actions)


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos


def collect_policy_interaction_data(pi, num_episodes, epsilon=0.3):
    # roll out the current policy with epsilon-greedy exploration and record (s,s',a). used for the BCO(alpha) post demo phase, learned policy will visit regions of the state space that a uniformly random policy almost never reaches and the epsilon noise keeps every action represented so the inverse dynamics model can tell them apart.
    states, next_states, actions = [], [], []
    env = gym.make("MountainCar-v0")
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = torch.argmax(
                        pi(torch.from_numpy(obs).float().unsqueeze(0))
                    ).item()
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            states.append(obs)
            next_states.append(next_obs)
            actions.append(a)
            obs = next_obs
    env.close()
    return np.array(states), np.array(next_states), np.array(actions)


def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s, a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).long().to(device)

    return obs_torch, acs_torch, obs2_torch


def train_policy(obs, acs, nn_policy, num_train_iters):
    pi_optimizer = Adam(nn_policy.parameters(), lr=0.1)
    # action space is discrete so our policy just needs to classify which action to take
    # we typically train classifiers using a cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()

    # run BC using all the demos in one giant batch
    for i in range(num_train_iters):
        # zero out automatic differentiation from last time
        pi_optimizer.zero_grad()
        # run each state in batch through policy to get predicted logits for classifying action
        pred_action_logits = nn_policy(obs)
        # now compute loss by comparing what the policy thinks it should do with what the demonstrator didd
        loss = loss_criterion(pred_action_logits, acs)
        print("iteration", i, "bc loss", loss)
        # back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        # perform update on policy parameters
        pi_optimizer.step()


class PolicyNetwork(nn.Module):
    """
    Simple neural network with two layers that maps a 2-d state to a prediction
    over which of the three discrete actions should be taken.
    The three outputs corresponding to the logits for a 3-way classification problem.

    """

    def __init__(self):
        super().__init__()

        # This layer has 2 inputs corresponding to car position and velocity
        self.fc1 = nn.Linear(2, 8)
        # This layer has three outputs corresponding to each of the three discrete actions
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        # this method performs a forward pass through the network, applying a non-linearity (ReLU) on the
        # outputs of the first layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class InverseDynamicsNetwork(nn.Module):
    """
    features are standardized with statistics computed from the random interaction data.
    an actions that changes the velocity by only +/-0.001 per step, so without normalization the signal that identifies the action is buried at a much smaller scale than positon.
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)

        # normalization statistics stored as buffers so they move with the model
        self.register_buffer("feat_mean", torch.zeros(4))
        self.register_buffer("feat_std", torch.ones(4))

    @staticmethod
    def make_features(obs, next_obs):
        return torch.cat([obs, next_obs - obs], dim=1)

    def set_normalization(self, feats):
        self.feat_mean = feats.mean(dim=0)
        self.feat_std = feats.std(dim=0) + 1e-8

    def forward(self, obs, next_obs):
        x = self.make_features(obs, next_obs)
        x = (x - self.feat_mean) / self.feat_std
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0", render_mode="human")
    else:
        env = gym.make("MountainCar-v0")

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs, _ = env.reset()
        while not done:
            # take the action that the network assigns the highest logit value to
            # Note that first we convert from numpy to tensor and then we get the value of the
            # argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))


def train_inverse_dynamics(inv_dyn, states, next_states, actions, num_iters, lr=1e-2):
    # train the inverse dynamics model with the full-batch cross entropy on (s,s',a)
    optimizer = Adam(inv_dyn.parameters(), lr=lr)
    loss_criterion = nn.CrossEntropyLoss()

    for i in range(num_iters):
        optimizer.zero_grad()
        logits = inv_dyn(states, next_states)
        loss = loss_criterion(logits, actions)
        loss.backward()
        optimizer.step()
        # batching
        if i % 100 == 0 or i == num_iters - 1:
            acc = (logits.argmax(dim=1) == actions).float().mean().item()
            print(
                f"inverse dynamics iteration {i} , loss {loss.item():.4f} train accuracy: {acc:.3f}"
            )


# convert to tensors
def to_tensors(states, next_states, actions):
    return (
        torch.from_numpy(np.asarray(states)).float().to(device),
        torch.from_numpy(np.asarray(next_states)).float().to(device),
        torch.from_numpy(np.asarray(actions)).long().to(device),
    )


def fit_inverse_dynamics(states, next_states, actions, num_iters):
    # build an invdynamicsnetwork, normalize on this data and train it
    inv_dyn = InverseDynamicsNetwork().to(device)
    inv_dyn.set_normalization(InverseDynamicsNetwork.make_features(states, next_states))
    train_inverse_dynamics(inv_dyn, states, next_states, actions, num_iters)
    return inv_dyn


def label_demos(inv_dyn, obs, next_obs):
    # most likely action for each action-free demo transition
    with torch.no_grad():
        return inv_dyn(obs, next_obs).argmax(dim=1)


def inverse_dynamics(obs, next_obs, num_inv_dyn_iters, num_random_epsiodes=5):
    """
    BCO step
     1. Collect (s, a, s') tuples by interacting with the environment using a
         random policy. Here the actions ARE known because we chose them.
      2. Train an inverse dynamics model p(a | s, s') on that data.
      3. Use the model to label the action-free demonstration transitions
         (obs, next_obs) with the most likely action.
    """
    # self supervised interaction data
    data = to_tensors(*collect_random_interaction_data(num_random_epsiodes))
    print(f"collected {len(data[2])} random transitions for inverse dynamics")

    # train inverse dynamics model
    inv_dyn = fit_inverse_dynamics(*data, num_inv_dyn_iters)

    # infer actions for the demos
    return label_demos(inv_dyn, obs, next_obs), data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--num_demos",
        default=1,
        type=int,
        help="number of human demonstrations to collect",
    )
    parser.add_argument(
        "--num_bc_iters", default=100, type=int, help="number of iterations to run BC"
    )
    parser.add_argument(
        "--num_inv_dyn_iters",
        default=1000,
        type=int,
        help="number of iterations to train inverse dynamics model",
    )
    parser.add_argument(
        "--num_evals",
        default=6,
        type=int,
        help="number of times to run policy after training for evaluation",
    )

    args = parser.parse_args()

    # collect human demos
    demos = collect_human_demos(args.num_demos)

    # process demos
    obs, ground_truth_acts, next_obs = torchify_demos(demos)

    # TODO: ADD CODE TO TRAIN INVERSE DYNAMICS MODEL AND ESTIMATE ACTIONS
    estimated_acts = inverse_dynamics(obs, next_obs, args.num_inv_dyn_iters)

    # train policy WITHOUT ground truth actions
    pi = PolicyNetwork()
    train_policy(obs, estimated_acts, pi, args.num_bc_iters)

    # evaluate learned policy
    evaluate_policy(pi, args.num_evals)
