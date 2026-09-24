import gymnasium as gym
import argparse
import csv
import os
import pickle
import random
from datetime import datetime
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device("cpu")


def collect_random_interaction_data(num_iters, action_repeat=1, seed=None):
    # action_repeat > 1 holds each uniformly random action for that many steps.
    # The actions are still random (no demo information is used), but holding them
    # builds momentum so the car covers more of the track. action_repeat=1 is the
    # original behavior.
    # seed: the environment and its action space have their own random generators,
    # which np.random.seed does not touch, so they are seeded here for reproducibility.
    states = []
    next_states = []
    actions = []

    env = gym.make("MountainCar-v0")
    env.action_space.seed(seed)

    for i in range(num_iters):
        obs, _ = env.reset(seed=None if seed is None else seed * 100000 + i)
        done = False
        t = 0
        while not done:
            if t % action_repeat == 0:
                a = env.action_space.sample()
            t += 1
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


def train_policy(obs, acs, nn_policy, num_train_iters, verbose=True):
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
        if verbose:
            print("iteration", i, "bc loss", loss)
        # back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        # perform update on policy parameters
        pi_optimizer.step()
    return loss.item()


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

    def __init__(self, hidden_dim=64, num_layers=2, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.fc1 = nn.Linear(4, hidden_dim)
        # num_layers - 1 extra hidden layers after fc1 (num_layers=2 gives the original fc1 -> fc2 -> fc3)
        self.hidden = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        )
        self.fc3 = nn.Linear(hidden_dim, 3)

        # normalization statistics stored as buffers so they move with the model
        self.register_buffer("feat_mean", torch.zeros(4))
        self.register_buffer("feat_std", torch.ones(4))

    @staticmethod
    def make_features(obs, next_obs):
        return torch.cat([obs, next_obs - obs], dim=1)

    def set_normalization(self, feats):
        # with normalize=False the buffers stay at mean 0, std 1 (raw inputs)
        if not self.normalize:
            return
        self.feat_mean = feats.mean(dim=0)
        self.feat_std = feats.std(dim=0) + 1e-8

    def forward(self, obs, next_obs):
        x = self.make_features(obs, next_obs)
        x = (x - self.feat_mean) / self.feat_std
        x = F.relu(self.fc1(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.fc3(x)


# evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True, eval_seed=None):
    if human_render:
        env = gym.make("MountainCar-v0", render_mode="human")
    else:
        env = gym.make("MountainCar-v0")

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs, _ = env.reset(seed=None if eval_seed is None else eval_seed + i)
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

    env.close()
    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))
    return policy_returns


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

    # final training accuracy after the last update
    with torch.no_grad():
        return (
            (inv_dyn(states, next_states).argmax(dim=1) == actions)
            .float()
            .mean()
            .item()
        )


# convert to tensors
def to_tensors(states, next_states, actions):
    return (
        torch.from_numpy(np.asarray(states)).float().to(device),
        torch.from_numpy(np.asarray(next_states)).float().to(device),
        torch.from_numpy(np.asarray(actions)).long().to(device),
    )


def fit_inverse_dynamics(
    states,
    next_states,
    actions,
    num_iters,
    hidden_dim=64,
    num_layers=2,
    normalize=True,
    lr=1e-2,
):
    # build an invdynamicsnetwork, normalize on this data and train it
    inv_dyn = InverseDynamicsNetwork(hidden_dim, num_layers, normalize).to(device)
    inv_dyn.set_normalization(InverseDynamicsNetwork.make_features(states, next_states))
    train_acc = train_inverse_dynamics(
        inv_dyn, states, next_states, actions, num_iters, lr
    )
    return inv_dyn, train_acc


def label_demos(inv_dyn, obs, next_obs):
    # most likely action for each action-free demo transition
    with torch.no_grad():
        return inv_dyn(obs, next_obs).argmax(dim=1)


def inverse_dynamics(
    obs,
    next_obs,
    num_inv_dyn_iters,
    num_random_episodes=5,
    hidden_dim=64,
    num_layers=2,
    normalize=True,
    lr=1e-2,
    action_repeat=1,
    seed=None,
):
    """
    BCO step
     1. Collect (s, a, s') tuples by interacting with the environment using a
         random policy. Here the actions ARE known because we chose them.
      2. Train an inverse dynamics model p(a | s, s') on that data.
      3. Use the model to label the action-free demonstration transitions
         (obs, next_obs) with the most likely action.
    """
    # self supervised interaction data
    data = to_tensors(
        *collect_random_interaction_data(num_random_episodes, action_repeat, seed)
    )
    print(f"collected {len(data[2])} random transitions for inverse dynamics")

    # train inverse dynamics model
    inv_dyn, train_acc = fit_inverse_dynamics(
        *data, num_inv_dyn_iters, hidden_dim, num_layers, normalize, lr
    )
    stats = {
        "num_random_transitions": len(data[2]),
        "random_pos_min": data[0][:, 0].min().item(),
        "random_pos_max": data[0][:, 0].max().item(),
        "inv_dyn_train_acc": train_acc,
    }

    # infer actions for the demos
    return label_demos(inv_dyn, obs, next_obs), stats


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
    # inverse dynamics experiment settings
    parser.add_argument(
        "--hidden_dim",
        default=64,
        type=int,
        help="hidden units per layer of the inverse dynamics model",
    )
    parser.add_argument(
        "--num_layers",
        default=2,
        type=int,
        help="number of hidden layers of the inverse dynamics model",
    )
    parser.add_argument(
        "--inv_dyn_lr",
        default=1e-2,
        type=float,
        help="Adam learning rate for the inverse dynamics model",
    )
    parser.add_argument(
        "--num_random_episodes",
        default=5,
        type=int,
        help="random-policy episodes (200 steps each) used to train inverse dynamics",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="turn off input standardization of the inverse dynamics model",
    )
    parser.add_argument(
        "--action_repeat",
        default=1,
        type=int,
        help="hold each random action for this many steps when collecting random data (1 = original)",
    )  # bookkeeping
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="seeds torch, numpy, the random interaction data and the network init",
    )
    parser.add_argument(
        "--eval_seed",
        default=1000,
        type=int,
        help="evaluation episode i starts from env.reset(seed=eval_seed + i); "
        "kept the same across runs so every configuration faces the same starts",
    )
    parser.add_argument(
        "--tag",
        default="",
        type=str,
        help="label saved with the results row, e.g. width, iters",
    )
    parser.add_argument(
        "--demo_file",
        default="demos.pkl",
        type=str,
        help="demos are loaded from here if it exists, otherwise collected and saved here",
    )
    parser.add_argument(
        "--results_file",
        default="results.csv",
        type=str,
        help="one row (config + metrics) is appended here per run",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="evaluate the policy without opening a window",
    )
    parser.add_argument(
        "--quiet_bc",
        action="store_true",
        help="do not print the BC loss every iteration",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # collect human demos once, then reuse them so every run is tested on the same demos
    if os.path.exists(args.demo_file):
        with open(args.demo_file, "rb") as f:
            demos = pickle.load(f)
        print(f"loaded {len(demos)} demo transitions from {args.demo_file}")
    else:
        demos = collect_human_demos(args.num_demos)
        with open(args.demo_file, "wb") as f:
            pickle.dump(demos, f)
        print(f"saved {len(demos)} demo transitions to {args.demo_file}")

    # process demos
    obs, ground_truth_acts, next_obs = torchify_demos(demos)

    # train inverse dynamics model on random interaction data and estimate demo actions
    estimated_acts, inv_stats = inverse_dynamics(
        obs,
        next_obs,
        args.num_inv_dyn_iters,
        num_random_episodes=args.num_random_episodes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        normalize=not args.no_normalize,
        lr=args.inv_dyn_lr,
        action_repeat=args.action_repeat,
        seed=args.seed,
    )
    # report only: ground truth actions are never used for training the inverse dynamics model
    names = ["left", "noop", "right"]
    correct = estimated_acts == ground_truth_acts
    acc = correct.float().mean().item()
    print(f"Inverse dynamics accuracy on demo actions: {acc:.3f}")
    per_action = {}
    for a, name in enumerate(names):
        m = ground_truth_acts == a
        n = int(m.sum().item())
        a_acc = correct[m].float().mean().item() if n > 0 else float("nan")
        per_action[f"demo_n_{name}"] = n
        per_action[f"demo_acc_{name}"] = a_acc
        if n > 0:
            print(f"  {name}: {n} steps, accuracy {a_acc:.3f}")

    # confusion matrix: which action each true action was predicted as (printed only)
    print(
        "confusion (rows = true, cols = predicted):  "
        + "  ".join(f"{n:>5s}" for n in names)
    )
    for a, name in enumerate(names):
        counts = [
            int(((ground_truth_acts == a) & (estimated_acts == p)).sum().item())
            for p in range(3)
        ]
        print(f"  {name:>5s}" + " " * 36 + "  ".join(f"{c:5d}" for c in counts))

    # where the demos went vs where the random training data went
    demo_pos = obs[:, 0]
    in_range = (demo_pos >= inv_stats["random_pos_min"]) & (
        demo_pos <= inv_stats["random_pos_max"]
    )
    n_in, n_out = int(in_range.sum().item()), int((~in_range).sum().item())
    coverage = {
        "demo_pos_min": demo_pos.min().item(),
        "demo_pos_max": demo_pos.max().item(),
        "demo_n_in_range": n_in,
        "demo_acc_in_range": correct[in_range].float().mean().item()
        if n_in > 0
        else float("nan"),
        "demo_n_out_range": n_out,
        "demo_acc_out_range": correct[~in_range].float().mean().item()
        if n_out > 0
        else float("nan"),
    }
    print(
        f"demo positions [{coverage['demo_pos_min']:.2f}, {coverage['demo_pos_max']:.2f}], "
        f"random data positions [{inv_stats['random_pos_min']:.2f}, {inv_stats['random_pos_max']:.2f}]"
    )
    print(
        f"  inside random data range:  {n_in} steps, accuracy {coverage['demo_acc_in_range']:.3f}"
    )
    print(
        f"  outside random data range: {n_out} steps, accuracy {coverage['demo_acc_out_range']:.3f}"
    )

    # train policy WITHOUT ground truth actions
    pi = PolicyNetwork()
    bc_loss = train_policy(
        obs, estimated_acts, pi, args.num_bc_iters, verbose=not args.quiet_bc
    )

    # evaluate learned policy
    returns = evaluate_policy(
        pi, args.num_evals, human_render=not args.no_render, eval_seed=args.eval_seed
    )

    # save config + results as one row
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "num_evals": args.num_evals,
        "num_demo_transitions": len(ground_truth_acts),
        "num_random_episodes": args.num_random_episodes,
        "action_repeat": args.action_repeat,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "normalize": not args.no_normalize,
        "inv_dyn_lr": args.inv_dyn_lr,
        "num_inv_dyn_iters": args.num_inv_dyn_iters,
        "num_bc_iters": args.num_bc_iters,
        **inv_stats,
        "demo_acc": acc,
        **per_action,
        **coverage,
        "bc_final_loss": bc_loss,
        "return_mean": float(np.mean(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "success_rate": float(np.mean(np.array(returns) > -200)),
    }
    new_file = not os.path.exists(args.results_file)
    if not new_file:
        with open(args.results_file, newline="") as f:
            header = next(csv.reader(f), [])
        if header != list(row.keys()):
            raise SystemExit(
                f"{args.results_file} was written by an older version with different columns; "
                f"use a new --results_file name"
            )
    with open(args.results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"saved results to {args.results_file}")
