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
from imitation.util import logger as imit_logger

import Env

# Basic Config
Render =1

if Render == 1:
    mode = "direct"
else:
    mode = "human"

CHUNK_TIMESTEPS = 500_000
TOTAL_TIMESTEPS = 500_000
N_ENVS = 1
RENDER_MODE = mode
SEED = 42
Place = False  # change from true to false - False = pick
load_steps_flag = True
load_policy_flag = True
load_reward_flag = True

# Saved checkpoints
CHECKPOINT_DIR = "checkpoints"
POLICY_PATH = os.path.join(CHECKPOINT_DIR, "ppo_policy.zip")
REWARD_PATH = os.path.join(CHECKPOINT_DIR, "airl_reward_net.pt")
META_PATH = os.path.join(CHECKPOINT_DIR, "meta.npz")

FINAL_POLICY_PATH = "airl_policy_final.zip"
FINAL_REWARD_PATH = "airl_reward_net_final.pt"

# Logging
LOG_DIR = "airl_logs"
LOGGER_FORMATS = ["stdout", "log", "csv", "json", "tensorboard"]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def build_env(n_envs=N_ENVS, render_mode=RENDER_MODE, seed=SEED):
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
        policy="MlpPolicy",
        env=venv,
        batch_size=256,
        ent_coef=0.01,
        learning_rate=2e-4,
        gamma=0.99,
        clip_range=0.2,
        vf_coef=0.1,
        n_epochs=10,
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

    if load_policy and os.path.isfile(POLICY_PATH):
        loaded = PPO.load(POLICY_PATH, env=rl_algo.get_env(), print_system_info=False)
        rl_algo.policy = loaded.policy
        rl_algo.set_parameters(loaded.get_parameters())

    if load_reward and os.path.isfile(REWARD_PATH):
        state_dict = torch.load(REWARD_PATH, map_location="cpu")
        reward_net.load_state_dict(state_dict)

    if load_steps and os.path.isfile(META_PATH):
        steps_done = int(np.load(META_PATH)["steps_done"])

    print(
        f"[checkpoint] Loaded: "
        f"policy={load_policy and os.path.isfile(POLICY_PATH)}, "
        f"reward={load_reward and os.path.isfile(REWARD_PATH)}, "
        f"steps_done={steps_done:,}"
    )
    return steps_done


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

        if isinstance(obs, tuple):
            obs = obs[0]

    return float(np.mean(ep_returns)), ep_returns


def get_reward_param_stats(reward_net):
    with torch.no_grad():
        params = [p.detach().cpu().reshape(-1) for p in reward_net.parameters() if p.requires_grad]
        if len(params) == 0:
            return {
                "reward/param_mean": 0.0,
                "reward/param_std": 0.0,
                "reward/param_l2": 0.0,
                "reward/param_max_abs": 0.0,
            }

        flat = torch.cat(params)
        return {
            "reward/param_mean": float(flat.mean().item()),
            "reward/param_std": float(flat.std(unbiased=False).item()),
            "reward/param_l2": float(torch.norm(flat, p=2).item()),
            "reward/param_max_abs": float(flat.abs().max().item()),
        }


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

    # File-backed logger: stdout + csv + json + tensorboard
    custom_logger = imit_logger.configure(
        folder=LOG_DIR,
        format_strs=LOGGER_FORMATS,
    )

    # AIRL trainer
    airl_trainer = airl.AIRL(
        demonstrations=expert_trajs,
        venv=venv,
        gen_algo=rl_algo,
        reward_net=reward_net,
        demo_batch_size=256,
        n_disc_updates_per_round=1,
        log_dir=LOG_DIR,
        custom_logger=custom_logger,
        init_tensorboard=True,
    )

    # Resume if checkpoint exists
    steps_done = load_checkpoint(
        rl_algo,
        reward_net,
        load_policy=load_policy_flag,
        load_reward=load_reward_flag,
        load_steps=load_steps_flag,
    )

    # Chunked training + periodic eval + checkpoints
    target = TOTAL_TIMESTEPS
    while steps_done < target:
        chunk = min(CHUNK_TIMESTEPS, target - steps_done)
        print(f"[train] Chunk: {chunk:,} steps (done {steps_done:,}/{target:,})")

        airl_trainer.train(total_timesteps=chunk)

        mean_ret, ep_rets = eval_policy_vec(rl_algo, venv, n_episodes=5, deterministic=True)
        steps_done += chunk

        print(f"[eval] mean ep return: {mean_ret:.3f}  (per-ep: {np.round(ep_rets, 2)})")

        # Save your own eval metrics into the same logger outputs
        airl_trainer.logger.record("eval/mean_ep_return", float(mean_ret))
        airl_trainer.logger.record("eval/std_ep_return", float(np.std(ep_rets)))
        airl_trainer.logger.record("eval/min_ep_return", float(np.min(ep_rets)))
        airl_trainer.logger.record("eval/max_ep_return", float(np.max(ep_rets)))
        airl_trainer.logger.record("eval/steps_done", int(steps_done))

        for i, ret in enumerate(ep_rets):
            airl_trainer.logger.record(f"eval/ep_return_{i}", float(ret))

        # Optional: trend reward-net parameter drift over time
        reward_stats = get_reward_param_stats(airl_trainer.reward_train)
        for key, value in reward_stats.items():
            airl_trainer.logger.record(key, value)

        # Flush to CSV / JSON / TensorBoard
        airl_trainer.logger.dump(steps_done)

        save_checkpoint(rl_algo, airl_trainer, steps_done)

    # Save finals
    rl_algo.save(FINAL_POLICY_PATH)
    torch.save(airl_trainer.reward_train.state_dict(), FINAL_REWARD_PATH)
    print(f"[done] Saved final policy to {FINAL_POLICY_PATH} and reward to {FINAL_REWARD_PATH}")