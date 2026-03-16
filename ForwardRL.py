import os
import glob
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from imitation.rewards import reward_nets
from imitation.util.networks import RunningNorm

import Env


# =========================
# Config (edit these)
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENV_ID = "PalletStacking-v0"

ENV_KWARGS_TRAIN = {"render_mode": "direct"}
ENV_KWARGS_EVAL = {"render_mode": "direct"}

TRAIN_N_ENVS = 1
EVAL_N_ENVS = 1

TOTAL_TIMESTEPS = 500_000

LOG_DIR = "stage2_ppo_logs_pick"
CKPT_DIR = "stage2_ppo_checkpoints_pick"
BEST_DIR = "stage2_ppo_best_pick"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# --- AIRL reward net from Stage 1 (pick) ---
REWARD_NET_PATH = "Pick reward/airl_reward_net.pt"

# Optional: load a PPO policy to calibrate AIRL reward scale (can be None)
CALIBRATE_WITH_PPO_POLICY = None
CALIBRATION_STEPS = 20_000
TARGET_REWARD_STD = 1.0

# Combined reward:
#   r_total = W_AIRL * (AIRL_SCALE * r_airl) + W_ENV * r_env
W_AIRL = 1.0
W_ENV = 1.0

# Safety clipping on the combined reward (helps PPO stability)
REWARD_CLIP = 10.0

# CRITICAL FIX:
# Use SubprocVecEnv for *all* envs (even n_envs=1) so train/eval/calib do NOT share a PyBullet client.
FORCE_SUBPROC_ALWAYS = True
SUBPROC_START_METHOD = "spawn"  # Windows-safe


# =========================
# VecEnv wrapper: combine AIRL reward with env reward
# =========================
class VecAirlPlusEnvReward(VecEnvWrapper):
    """
    Adds AIRL reward to the environment reward.

      r_total = w_airl * (airl_scale * r_airl) + w_env * r_env
    """

    def __init__(
        self,
        venv,
        reward_net,
        device="cpu",
        airl_scale=1.0,
        w_airl=1.0,
        w_env=1.0,
        reward_clip=None,
        add_debug_info=True,
    ):
        super().__init__(venv)
        self.reward_net = reward_net.to(device)
        self.device = torch.device(device)

        self.airl_scale = float(airl_scale)
        self.w_airl = float(w_airl)
        self.w_env = float(w_env)
        self.reward_clip = reward_clip
        self.add_debug_info = bool(add_debug_info)

        self._last_obs = None
        self._last_actions = None

    def reset(self):
        obs = self.venv.reset()
        # Some VecEnv impls return (obs, info)
        if isinstance(obs, tuple):
            obs = obs[0]
        self._last_obs = obs
        return obs

    def step_async(self, actions):
        self._last_actions = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        next_obs, env_rewards, dones, infos = self.venv.step_wait()

        obs_t = self._last_obs
        acts_t = self._last_actions
        done_t = dones.astype(np.float32)

        # AIRL reward (frozen net)
        with torch.no_grad():
            r_airl = self.reward_net(
                torch.as_tensor(obs_t, dtype=torch.float32, device=self.device),
                torch.as_tensor(acts_t, dtype=torch.float32, device=self.device),
                torch.as_tensor(next_obs, dtype=torch.float32, device=self.device),
                torch.as_tensor(done_t, dtype=torch.float32, device=self.device),
            ).detach().cpu().numpy().reshape(-1)

        r_airl = self.airl_scale * r_airl
        r_env = np.asarray(env_rewards, dtype=np.float32).reshape(-1)

        r_total = self.w_airl * r_airl + self.w_env * r_env

        if self.reward_clip is not None:
            r_total = np.clip(r_total, -self.reward_clip, self.reward_clip)

        if self.add_debug_info:
            for i, info in enumerate(infos):
                info["r_airl"] = float(r_airl[i])
                info["r_env"] = float(r_env[i])
                info["r_total"] = float(r_total[i])

        self._last_obs = next_obs
        return next_obs, r_total.astype(np.float32), dones, infos


# =========================
# Build env + load reward net
# =========================
def build_venv(env_id: str, env_kwargs: dict, seed: int, n_envs: int):
    """
    IMPORTANT: For PyBullet envs, train/eval/calib must NOT live in the same Python process,
    otherwise they can share/reset the same global Bullet connection and delete each other's bodies.
    """
    use_subproc = FORCE_SUBPROC_ALWAYS or (n_envs > 1)
    vec_cls = SubprocVecEnv if use_subproc else None
    vec_kwargs = {"start_method": SUBPROC_START_METHOD} if vec_cls is SubprocVecEnv else None

    # Ensure kwargs isn't shared/mutated across builds
    env_kwargs = dict(env_kwargs) if env_kwargs is not None else {}

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        seed=seed,
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_kwargs,
    )


def load_reward_net(venv, path: str):
    net = reward_nets.BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    state = torch.load(path, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def calibrate_reward_scale(
    base_env,
    reward_net,
    device,
    steps=10_000,
    target_std=1.0,
    ppo_policy_path=None,
):
    """
    Computes a static scale factor so that AIRL reward std ≈ target_std
    under either (a) a loaded PPO policy or (b) random actions.
    """
    tmp_env = VecAirlPlusEnvReward(
        base_env,
        reward_net=reward_net,
        device=device,
        airl_scale=1.0,
        w_airl=1.0,
        w_env=0.0,  # isolate AIRL reward only
        reward_clip=None,
        add_debug_info=False,
    )

    model = None
    if ppo_policy_path is not None and os.path.isfile(ppo_policy_path):
        try:
            model = PPO.load(ppo_policy_path, env=tmp_env, print_system_info=False)
            print(f"[calib] Using PPO policy for calibration: {ppo_policy_path}")
        except Exception as e:
            print(f"[calib] Could not load PPO policy ({e}). Falling back to random actions.")
            model = None

    obs = tmp_env.reset()
    rewards = []

    for _ in range(int(steps)):
        if model is None:
            act = np.stack([tmp_env.action_space.sample() for _ in range(tmp_env.num_envs)], axis=0)
        else:
            act, _ = model.predict(obs, deterministic=True)

        obs, r, dones, _infos = tmp_env.step(act)
        rewards.extend(np.asarray(r, dtype=np.float32).reshape(-1).tolist())

        if np.any(dones):
            obs = tmp_env.reset()

    rewards = np.asarray(rewards, dtype=np.float32)
    r_std = float(rewards.std())
    r_mean = float(rewards.mean())
    print(f"[calib] AIRL raw: mean={r_mean:.4f}, std={r_std:.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")

    if r_std < 1e-6:
        print("[calib] Reward std ~ 0. Using scale=1.0 (check reward net outputs).")
        return 1.0

    scale = float(target_std / r_std)
    print(f"[calib] Using AIRL_SCALE={scale:.6f} to target std≈{target_std}")
    return scale


def build_or_resume_ppo(train_env):
    """
    Resume from latest checkpoint in CKPT_DIR if present, else start new PPO.
    """
    ckpt_pattern = os.path.join(CKPT_DIR, "ppo_stage2_*_steps.zip")
    ckpts = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime)

    if ckpts:
        latest = ckpts[-1]
        print(f"[stage2] Resuming PPO from checkpoint: {latest}")
        return PPO.load(latest, env=train_env, device=DEVICE, print_system_info=False)

    best_path = os.path.join(BEST_DIR, "best_model.zip")
    if os.path.isfile(best_path):
        print(f"[stage2] No checkpoint found. Resuming from best model: {best_path}")
        return PPO.load(best_path, env=train_env, device=DEVICE, print_system_info=False)

    print("[stage2] No checkpoint/best model found. Starting PPO from scratch.")
    policy_kwargs = dict(net_arch=[128, 128])
    return PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=DEVICE,
    )


if __name__ == "__main__":
    # --- Build *separate-process* base envs (critical for PyBullet isolation) ---
    base_train_env = build_venv(ENV_ID, ENV_KWARGS_TRAIN, SEED, n_envs=TRAIN_N_ENVS)
    base_eval_env = build_venv(ENV_ID, ENV_KWARGS_EVAL, SEED + 123, n_envs=EVAL_N_ENVS)

    # Load frozen AIRL reward net
    reward_net = load_reward_net(base_train_env, REWARD_NET_PATH)

    # Calibrate AIRL scale
    calib_env = build_venv(ENV_ID, ENV_KWARGS_EVAL, SEED + 10, n_envs=1)
    AIRL_SCALE = calibrate_reward_scale(
        base_env=calib_env,
        reward_net=reward_net,
        device=DEVICE,
        steps=CALIBRATION_STEPS,
        target_std=TARGET_REWARD_STD,
        ppo_policy_path=CALIBRATE_WITH_PPO_POLICY,
    )
    # IMPORTANT: cleanly close calibration env (especially if SubprocVecEnv)
    calib_env.close()

    # Wrap training + eval with combined reward
    train_env = VecAirlPlusEnvReward(
        base_train_env,
        reward_net=reward_net,
        device=DEVICE,
        airl_scale=AIRL_SCALE,
        w_airl=W_AIRL,
        w_env=W_ENV,
        reward_clip=REWARD_CLIP,
        add_debug_info=True,
    )
    eval_env = VecAirlPlusEnvReward(
        base_eval_env,
        reward_net=reward_net,
        device=DEVICE,
        airl_scale=AIRL_SCALE,
        w_airl=W_AIRL,
        w_env=W_ENV,
        reward_clip=REWARD_CLIP,
        add_debug_info=True,
    )

    # Ensure SB3 logs ep_rew_mean using wrapped reward
    train_env = VecMonitor(train_env)
    eval_env = VecMonitor(eval_env)

    print("train_env.num_envs =", train_env.num_envs)
    print("eval_env.num_envs  =", eval_env.num_envs)
    print(f"[stage2] Reward mix: W_AIRL={W_AIRL}, W_ENV={W_ENV}, AIRL_SCALE={AIRL_SCALE:.6f}, clip={REWARD_CLIP}")
    print(f"[stage2] VecEnv isolation: FORCE_SUBPROC_ALWAYS={FORCE_SUBPROC_ALWAYS}, start_method={SUBPROC_START_METHOD}")

    # Build or resume PPO
    model = build_or_resume_ppo(train_env)

    # Callbacks
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=CKPT_DIR,
        name_prefix="ppo_stage2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=BEST_DIR,
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    print("[stage2] Training PPO with combined reward (AIRL + env reward)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[ckpt_cb, eval_cb])

    final_path = os.path.join(BEST_DIR, "ppo_stage2_final.zip")
    model.save(final_path)
    print(f"[stage2] Saved final PPO model to: {final_path}")

    # Clean shutdown
    eval_env.close()
    train_env.close()
