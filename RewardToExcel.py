#!/usr/bin/env python3

import os
import pickle
import numpy as np
import torch
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from imitation.rewards import reward_nets
from imitation.util.networks import RunningNorm

import Env  # registers PalletStacking-v0


# ====== Paths / Config (keep consistent with your training script) ======
CHECKPOINT_DIR       = "checkpoints"
FINAL_POLICY_PATH    = "airl_policy_final.zip"
FINAL_REWARD_PATH    = "airl_reward_net_final.pt"
REWARD_CSV_PATH      = os.path.join(CHECKPOINT_DIR, "airl_reward_samples.csv")

N_ENVS               = 1
RENDER_MODE          = "direct"   # or "human" if you want to watch
SEED                 = 42
N_EPISODES_TO_LOG    = 5          # how many episodes to sample
MAX_STEPS_PER_EPISODE = 1024      # safety cap, must match your env logic


# ====== Helpers duplicated from training script ======
def build_env(n_envs=N_ENVS, render_mode=RENDER_MODE, seed=SEED):
    """
    Build the VecEnv exactly like in training so obs/action spaces match.
    """
    venv = make_vec_env(
        "PalletStacking-v0",
        n_envs=n_envs,
        env_kwargs={"render_mode": render_mode},
        seed=seed,
    )
    return venv


def build_reward_net(venv):
    """
    Must match the architecture used during training, so that the
    saved state_dict can be loaded.
    """
    return reward_nets.BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )


def load_policy_and_reward(venv):
    """
    Load the trained PPO policy and the trained reward network from checkpoints.
    """
    # Load PPO policy
    if not os.path.isfile(FINAL_POLICY_PATH):
        raise FileNotFoundError(
            f"Could not find final policy at {FINAL_POLICY_PATH}. "
            f"Run training first or adjust FINAL_POLICY_PATH."
        )

    policy = PPO.load(FINAL_POLICY_PATH, env=venv, print_system_info=False)

    # Rebuild reward net and load weights
    if not os.path.isfile(FINAL_REWARD_PATH):
        raise FileNotFoundError(
            f"Could not find final reward net at {FINAL_REWARD_PATH}. "
            f"Run training first or adjust FINAL_REWARD_PATH."
        )

    reward_net = build_reward_net(venv)
    state_dict = torch.load(FINAL_REWARD_PATH, map_location="cpu")
    reward_net.load_state_dict(state_dict)

    # Put reward net on same device as policy (usually cpu or cuda)
    device = policy.device
    reward_net.to(device)

    return policy, reward_net


def rollout_with_reward_to_records(venv, policy, reward_net,
                                   n_episodes=N_EPISODES_TO_LOG,
                                   max_steps_per_episode=MAX_STEPS_PER_EPISODE):
    """
    Run the learned policy in the environment, evaluate the learned reward
    at each step, and collect everything into a list of dicts.

    Each row will contain:
      - episode, step, env_idx
      - reward (from IRL reward_net.predict_processed)
      - obs_0, obs_1, ..., obs_n
      - act_0, act_1, ..., act_m
    """

    records = []

    # VecEnv reset: sometimes returns (obs, info), sometimes just obs
    reset_out = venv.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    num_envs = venv.num_envs
    ep_counts = np.zeros(num_envs, dtype=int)
    ep_steps = np.zeros(num_envs, dtype=int)

    # We stop when every env has finished at least n_episodes
    done_target = n_episodes

    while ep_counts.min() < done_target:
        # Predict actions using learned policy
        actions, _ = policy.predict(obs, deterministic=True)

        # Step env
        next_obs, _, dones, infos = venv.step(actions)

        # Compute learned reward for this transition batch
        # reward_net.predict_processed expects numpy arrays
        # shape: (batch_size, ...)
        rewards = reward_net.predict_processed(
            state=np.array(obs),
            action=np.array(actions),
            next_state=np.array(next_obs),
            done=np.array(dones, dtype=np.float32),
        )

        # Store a row per env
        for env_id in range(num_envs):
            # Skip if this env already completed enough episodes
            if ep_counts[env_id] >= done_target:
                continue

            row = {
                "episode": int(ep_counts[env_id]),
                "step": int(ep_steps[env_id]),
                "env": int(env_id),
                "reward": float(rewards[env_id]),
            }

            # Flatten obs and action for Excel
            obs_flat = np.asarray(obs[env_id]).ravel()
            act_flat = np.asarray(actions[env_id]).ravel()

            for j, val in enumerate(obs_flat):
                row[f"obs_{j}"] = float(val)

            for j, val in enumerate(act_flat):
                row[f"act_{j}"] = float(val)

            records.append(row)

        # Update episode/step counters
        ep_steps += 1

        for env_id, done in enumerate(dones):
            if done:
                ep_counts[env_id] += 1
                ep_steps[env_id] = 0

        # Safety break to avoid infinite loops if something goes weird
        if ep_steps.max() > max_steps_per_episode:
            print(
                f"[warning] Reached max_steps_per_episode={max_steps_per_episode}. "
                "Stopping rollout early."
            )
            break

        obs = next_obs

    return records


# ====== Main: run rollout, dump to CSV ======
if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("[info] Building environment...")
    venv = build_env()

    print("[info] Loading policy and reward net from checkpoints...")
    policy, reward_net = load_policy_and_reward(venv)

    print(f"[info] Collecting samples from {N_EPISODES_TO_LOG} episodes...")
    records = rollout_with_reward_to_records(
        venv, policy, reward_net,
        n_episodes=N_EPISODES_TO_LOG,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    )

    print(f"[info] Collected {len(records)} transitions, saving to CSV...")
    df = pd.DataFrame(records)
    df.to_csv(REWARD_CSV_PATH, index=False)

    print(f"[done] Wrote reward samples to: {REWARD_CSV_PATH}")
