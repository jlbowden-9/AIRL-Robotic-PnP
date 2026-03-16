import os
import pickle
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms.adversarial import airl
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Trajectory
from imitation.rewards import reward_nets
from imitation.util.networks import RunningNorm

import Env

# Basic Config
Render = 0

if Render==1:
    mode = "direct"
else:
    mode = "human"

CHUNK_TIMESTEPS   = 100_000
TOTAL_TIMESTEPS   = 300_000
N_ENVS            = 1
RENDER_MODE       = mode
SEED              = 42
Place             = False #change from true to false - False = pick
load_steps_flag = True # set this to load individual parts sometimes helpful not t load reward if disc is to strong
load_policy_flag = True
load_reward_flag = False


# Saved checkpoints
CHECKPOINT_DIR    = "checkpoints"
POLICY_PATH       = os.path.join(CHECKPOINT_DIR, "ppo_policy.zip")
REWARD_PATH       = os.path.join(CHECKPOINT_DIR, "airl_reward_net.pt")
META_PATH         = os.path.join(CHECKPOINT_DIR, "meta.npz")

FINAL_POLICY_PATH = "airl_policy_final.zip"
FINAL_REWARD_PATH = "airl_reward_net_final.pt"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def build_env(n_envs=N_ENVS, render_mode=RENDER_MODE, seed=SEED):
    # NOTE: Do NOT wrap with RolloutInfoWrapper (it’s for single envs, not VecEnv)
    venv = make_vec_env(
        "PalletStacking-v0",
        n_envs=n_envs,
        env_kwargs={"render_mode": render_mode},
        seed=seed,
    )
    return venv

def build_generator(venv):
    policy_kwargs = dict(net_arch=[256, 256])
    algo = PPO(
        policy="MlpPolicy", # MlpPolicy
        env=venv,
        batch_size=256, # orig 64
        ent_coef=0.01, # orig 0.0 effects the amount of exploration from the PPO Higher (more random and exploritive)
        learning_rate=3e-4, # original 3e-4 - faster learning but more risk to be unstable
        gamma=0.99, #orig: .95 priority of the future return (this is used in MPC too)
        clip_range=0.15, #restricts the amount of learning from 1 iter to the next
        vf_coef=0.1, # orig. 0.1 weight of the value function?
        n_epochs=10,
        seed=SEED,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    return algo

def build_reward_net(venv):
    return reward_nets.BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )


def save_checkpoint(rl_algo, airl_trainer, steps_done: int):
    rl_algo.save(POLICY_PATH)
    torch.save(airl_trainer.reward_train.state_dict(), REWARD_PATH)
    np.savez(META_PATH, steps_done=int(steps_done))
    print(f"[checkpoint] Saved at {steps_done:,} steps.")

def load_checkpoint(rl_algo, reward_net, *, load_policy=True, load_reward=True, load_steps=True):
    steps_done = 0

    # Load PPO even if reward/meta are missing
    if load_policy and os.path.isfile(POLICY_PATH):
        loaded = PPO.load(POLICY_PATH, env=rl_algo.get_env(), print_system_info=False)
        rl_algo.policy = loaded.policy
        rl_algo.set_parameters(loaded.get_parameters())
        #rl_algo.clip_range_vf = getattr(loaded, "clip_range_vf", None)

    # Load reward only if you want it
    if load_reward and os.path.isfile(REWARD_PATH):
        state_dict = torch.load(REWARD_PATH, map_location="cpu")
        reward_net.load_state_dict(state_dict)

    # Load steps only if you want it
    if load_steps and os.path.isfile(META_PATH):
        steps_done = int(np.load(META_PATH)["steps_done"])

    print(f"[checkpoint] Loaded: policy={load_policy and os.path.isfile(POLICY_PATH)}, "
          f"reward={load_reward and os.path.isfile(REWARD_PATH)}, steps_done={steps_done:,}")
    return steps_done



# =========================
# Evaluation (VecEnv-safe)
# =========================
def eval_policy_vec(model, vec_env, n_episodes=5, deterministic=True):

    reset_out = vec_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    ep_returns = []
    ep_ret = np.zeros(vec_env.num_envs, dtype=np.float64)

    while len(ep_returns) < n_episodes:
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(actions)
        ep_ret += rewards

        for i, done in enumerate(dones):
            if done:
                ep_returns.append(float(ep_ret[i]))
                ep_ret[i] = 0.0
                if len(ep_returns) >= n_episodes:
                    break

        if isinstance(obs, tuple):  # just in case a wrapper returns (obs, info)
            obs = obs[0]

    return float(np.mean(ep_returns)), ep_returns


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Load expert demos
    if Place:
        if os.path.isfile("expert_trajs_Place.pkl"):
            with open("expert_trajs_Place.pkl", "rb") as f:
                expert_trajs = pickle.load(f)
            if isinstance(expert_trajs, Trajectory):
                expert_trajs = [expert_trajs]
        else:
            with open("expert_trajs_Place.pkl", "rb") as f:
                expert_traj = pickle.load(f)
            expert_trajs = [expert_traj] if isinstance(expert_traj, Trajectory) else expert_traj
    else:
        if os.path.isfile("expert_trajs_Pick.pkl"):
            with open("expert_trajs_Pick.pkl", "rb") as f:
                expert_trajs = pickle.load(f)
            if isinstance(expert_trajs, Trajectory):
                expert_trajs = [expert_trajs]
        else:
            with open("expert_trajs_Pick.pkl", "rb") as f:
                expert_traj = pickle.load(f)
            expert_trajs = [expert_traj] if isinstance(expert_traj, Trajectory) else expert_traj

    # Envs & models
    venv = build_env()
    rl_algo = build_generator(venv)
    reward_net = build_reward_net(venv)
    # AIRL trainer
    airl_trainer = airl.AIRL(
        demonstrations=expert_trajs,
        venv=venv,
        gen_algo=rl_algo,
        reward_net=reward_net,
        demo_batch_size=256,
        n_disc_updates_per_round=1,
        gen_replay_buffer_capacity=50000,
        log_dir="airl_logs",
    )

    # Resume if checkpoint exists
    steps_done = load_checkpoint(rl_algo, reward_net, load_policy=load_policy_flag, load_reward=load_reward_flag, load_steps=load_steps_flag)

    # Chunked training + periodic eval + checkpoints
    target = TOTAL_TIMESTEPS
    while steps_done < target:
        chunk = min(CHUNK_TIMESTEPS, target - steps_done)
        print(f"[train] Chunk: {chunk:,} steps (done {steps_done:,}/{target:,})")

        airl_trainer.train(total_timesteps=chunk)

        mean_ret, ep_rets = eval_policy_vec(rl_algo, venv, n_episodes=5, deterministic=True)

        print(f"[eval] mean ep return: {mean_ret:.3f}  (per-ep: {np.round(ep_rets, 2)})")

        steps_done += chunk
        save_checkpoint(rl_algo, airl_trainer, steps_done)

    # Save finals
    rl_algo.save(FINAL_POLICY_PATH)
    torch.save(airl_trainer.reward_train.state_dict(), FINAL_REWARD_PATH)
    print(f"[done] Saved final policy to {FINAL_POLICY_PATH} and reward to {FINAL_REWARD_PATH}")