import os
import glob
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from imitation.rewards import reward_nets
from imitation.util.networks import RunningNorm

import Env

SEED = 42

TRAIN_N_ENVS = 1
EVAL_N_ENVS = 1
CALIB_N_ENVS = 1

TRAIN_RENDER = 1
TRAIN_RENDER_MODE = "human" if TRAIN_RENDER else "direct"

TOTAL_TIMESTEPS = 500_000

LOG_DIR = "stage2_ppo_logs"
CKPT_DIR = "stage2_ppo_checkpoints"
BEST_DIR = "stage2_ppo_best"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PICK_REWARD_NET_PATH = os.path.join("Pick reward", "airl_reward_net.pt")
PLACE_REWARD_NET_PATH = os.path.join("Place reward", "airl_reward_net.pt")

REWARD_CLIP = 500.0
TARGET_REWARD_STD = 1.0
CALIBRATION_STEPS = 0

W_AIRL = .05
W_ENV = 1.0

STAGE_INDEX = 3
PICK_STAGE_VALUE = 0
PICK_ERR_INDEX = 5
PLACE_ERR_INDEX = 6

PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.02
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_TARGET_KL = None
PPO_N_EPOCHS = 10
PPO_BATCH_SIZE = 256
PPO_N_STEPS = 2048
PPO_POLICY_NET_ARCH = [256, 256]
RESET_POLICY_LOG_STD = None


def _const_schedule(v):
    v = float(v)
    return lambda _progress_remaining: v


def apply_ppo_hparams(model: PPO):
    model.gamma = float(PPO_GAMMA)
    model.gae_lambda = float(PPO_GAE_LAMBDA)
    model.ent_coef = float(PPO_ENT_COEF)
    model.vf_coef = float(PPO_VF_COEF)
    model.max_grad_norm = float(PPO_MAX_GRAD_NORM)
    model.target_kl = PPO_TARGET_KL

    model.n_epochs = int(PPO_N_EPOCHS)
    model.batch_size = int(PPO_BATCH_SIZE)

    if PPO_N_STEPS is not None and int(PPO_N_STEPS) != int(model.n_steps):
        model.n_steps = int(PPO_N_STEPS)

    model.learning_rate = float(PPO_LEARNING_RATE)
    model.lr_schedule = _const_schedule(PPO_LEARNING_RATE)
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = float(PPO_LEARNING_RATE)

    model.clip_range = _const_schedule(PPO_CLIP_RANGE)

    if RESET_POLICY_LOG_STD is not None and hasattr(model.policy, "log_std"):
        with torch.no_grad():
            model.policy.log_std.data.fill_(float(RESET_POLICY_LOG_STD))


class RewardComponentsTBCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_sums = None
        self._ep_len = None

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_sums = {
            "r_env": np.zeros(n, dtype=np.float64),
            "r_airl": np.zeros(n, dtype=np.float64),
            "r_total": np.zeros(n, dtype=np.float64),
            "pick_error": np.zeros(n, dtype=np.float64),
            "place_error": np.zeros(n, dtype=np.float64),
            "error": np.zeros(n, dtype=np.float64),
        }
        self._ep_len = np.zeros(n, dtype=np.int32)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)

        for info in infos:
            if not isinstance(info, dict):
                continue
            self.logger.record_mean("rollout/r_env", float(info.get("r_env", 0.0)))
            self.logger.record_mean("rollout/r_airl_selected", float(info.get("r_airl_selected", 0.0)))
            self.logger.record_mean("rollout/r_total", float(info.get("r_total", 0.0)))
            self.logger.record_mean("rollout/stage", float(info.get("stage", 0.0)))
            self.logger.record_mean("rollout/pick_error", float(info.get("pick_error", 0.0)))
            self.logger.record_mean("rollout/place_error", float(info.get("place_error", 0.0)))
            self.logger.record_mean("rollout/error", float(info.get("error", 0.0)))

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            self._ep_sums["r_env"][i] += float(info.get("r_env", 0.0))
            self._ep_sums["r_airl"][i] += float(info.get("r_airl_selected", 0.0))
            self._ep_sums["r_total"][i] += float(info.get("r_total", 0.0))
            self._ep_sums["pick_error"][i] += float(info.get("pick_error", 0.0))
            self._ep_sums["place_error"][i] += float(info.get("place_error", 0.0))
            self._ep_sums["error"][i] += float(info.get("error", 0.0))
            self._ep_len[i] += 1

            if dones is not None and bool(dones[i]):
                L = max(int(self._ep_len[i]), 1)

                self.logger.record_mean("episode/r_env_sum", self._ep_sums["r_env"][i])
                self.logger.record_mean("episode/r_airl_sum", self._ep_sums["r_airl"][i])
                self.logger.record_mean("episode/r_total_sum", self._ep_sums["r_total"][i])

                self.logger.record_mean("episode/r_env_mean", self._ep_sums["r_env"][i] / L)
                self.logger.record_mean("episode/r_airl_mean", self._ep_sums["r_airl"][i] / L)
                self.logger.record_mean("episode/r_total_mean", self._ep_sums["r_total"][i] / L)

                self.logger.record_mean("episode/pick_error_mean", self._ep_sums["pick_error"][i] / L)
                self.logger.record_mean("episode/place_error_mean", self._ep_sums["place_error"][i] / L)
                self.logger.record_mean("episode/error_mean", self._ep_sums["error"][i] / L)

                self.logger.record_mean("episode/len", float(L))
                self.logger.record_mean("episode/is_success", float(info.get("is_success", 0.0)))

                self._ep_sums["r_env"][i] = 0.0
                self._ep_sums["r_airl"][i] = 0.0
                self._ep_sums["r_total"][i] = 0.0
                self._ep_sums["pick_error"][i] = 0.0
                self._ep_sums["place_error"][i] = 0.0
                self._ep_sums["error"][i] = 0.0
                self._ep_len[i] = 0

        return True


class VecPickPlaceThreeReward(VecEnvWrapper):
    def __init__(
        self,
        venv,
        pick_reward_net,
        place_reward_net,
        device="cpu",
        pick_scale=1.0,
        place_scale=1.0,
        w_airl=1.0,
        w_env=1.0,
        reward_clip=500.0,
        stage_index=3,
        pick_stage_value=0,
    ):
        super().__init__(venv)
        self.pick_reward_net = pick_reward_net.to(device)
        self.place_reward_net = place_reward_net.to(device)
        self.device = torch.device(device)

        self.pick_scale = float(pick_scale)
        self.place_scale = float(place_scale)
        self.w_airl = float(w_airl)
        self.w_env = float(w_env)
        self.reward_clip = reward_clip

        self.stage_index = int(stage_index)
        self.pick_stage_value = int(pick_stage_value)

        self._last_obs = None
        self._last_actions = None

    def reset(self):
        obs = self.venv.reset()
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

        stage = np.rint(np.asarray(obs_t[:, self.stage_index], dtype=np.float32)).astype(np.int32)
        pick_err = np.asarray(obs_t[:, PICK_ERR_INDEX], dtype=np.float32)
        place_err = np.asarray(obs_t[:, PLACE_ERR_INDEX], dtype=np.float32)

        use_pick = (stage == self.pick_stage_value)
        err_selected = np.where(use_pick, pick_err, place_err).astype(np.float32)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32, device=self.device)
            act_tensor = torch.as_tensor(acts_t, dtype=torch.float32, device=self.device)
            next_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            done_tensor = torch.as_tensor(done_t, dtype=torch.float32, device=self.device)

            r_pick = self.pick_reward_net(obs_tensor, act_tensor, next_tensor, done_tensor)
            r_place = self.place_reward_net(obs_tensor, act_tensor, next_tensor, done_tensor)

            r_pick = r_pick.detach().cpu().numpy().reshape(-1) * self.pick_scale
            r_place = r_place.detach().cpu().numpy().reshape(-1) * self.place_scale

        r_airl = np.where(use_pick, r_pick, r_place).astype(np.float32)

        r_env = np.asarray(env_rewards, dtype=np.float32).reshape(-1)
        r_total = self.w_airl * r_airl + self.w_env * r_env

        if self.reward_clip is not None:
            r_total = np.clip(r_total, -self.reward_clip, self.reward_clip)

        for i, info in enumerate(infos):
            info["stage"] = int(stage[i])
            info["pick_error"] = float(pick_err[i])
            info["place_error"] = float(place_err[i])
            info["error"] = float(err_selected[i])
            info["r_pick"] = float(r_pick[i])
            info["r_place"] = float(r_place[i])
            info["r_airl_selected"] = float(r_airl[i])
            info["r_env"] = float(r_env[i])
            info["r_total"] = float(r_total[i])

        self._last_obs = next_obs
        return next_obs, r_total.astype(np.float32), dones, infos


def build_venv(render_mode: str, seed: int, n_envs: int):
    vec_cls = SubprocVecEnv if n_envs > 1 else None
    vec_kwargs = {"start_method": "spawn"} if vec_cls is SubprocVecEnv else None

    return make_vec_env(
        "PalletStacking-v0",
        n_envs=n_envs,
        env_kwargs={"render_mode": render_mode},
        seed=seed,
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_kwargs,
    )


def load_basic_reward_net(venv, path: str):
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


def _resolve_reward_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    folder = os.path.dirname(path) or "."
    candidates = sorted(glob.glob(os.path.join(folder, "airl_reward_net*.pt")), key=os.path.getmtime)
    if candidates:
        return candidates[-1]
    return path


def calibrate_reward_scale(
    base_env,
    reward_net,
    device,
    stage_index: int,
    pick_stage_value: int,
    calibrate_pick: bool,
    steps: int = 20_000,
    target_std: float = 1.0,
):
    obs = base_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    samples = []
    for _ in range(steps):
        act = np.stack([base_env.action_space.sample() for _ in range(base_env.num_envs)], axis=0)
        next_obs, _r_env, dones, _infos = base_env.step(act)

        obs_t = obs
        done_t = dones.astype(np.float32)

        stage = np.rint(np.asarray(obs_t[:, stage_index], dtype=np.float32)).astype(np.int32)
        mask = (stage == int(pick_stage_value)) if calibrate_pick else (stage != int(pick_stage_value))

        with torch.no_grad():
            r = reward_net(
                torch.as_tensor(obs_t, dtype=torch.float32, device=device),
                torch.as_tensor(act, dtype=torch.float32, device=device),
                torch.as_tensor(next_obs, dtype=torch.float32, device=device),
                torch.as_tensor(done_t, dtype=torch.float32, device=device),
            ).detach().cpu().numpy().reshape(-1)

        if np.any(mask):
            samples.extend(r[mask].astype(np.float32).tolist())

        obs = next_obs
        if np.any(dones):
            obs = base_env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < 100:
        return 1.0

    r_std = float(samples.std())
    if r_std < 1e-6:
        return 1.0

    return float(target_std / r_std)


def build_or_resume_ppo(train_env):
    ckpt_pattern = os.path.join(CKPT_DIR, "ppo_stage2_*_steps.zip")
    ckpts = sorted(glob.glob(ckpt_pattern), key=os.path.getmtime)

    if ckpts:
        latest = ckpts[-1]
        model = PPO.load(latest, env=train_env, device=DEVICE, print_system_info=False)
        model.tensorboard_log = LOG_DIR
        apply_ppo_hparams(model)
        return model

    best_path = os.path.join(BEST_DIR, "best_model.zip")
    if os.path.isfile(best_path):
        model = PPO.load(best_path, env=train_env, device=DEVICE, print_system_info=False)
        model.tensorboard_log = LOG_DIR
        apply_ppo_hparams(model)
        return model

    policy_kwargs = dict(net_arch=PPO_POLICY_NET_ARCH)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        target_kl=PPO_TARGET_KL,
        policy_kwargs=policy_kwargs,
        seed=SEED,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=DEVICE,
    )
    apply_ppo_hparams(model)
    return model


if __name__ == "__main__":
    base_train_env = build_venv(TRAIN_RENDER_MODE, SEED, n_envs=TRAIN_N_ENVS)
    base_eval_env = build_venv("direct", SEED + 123, n_envs=EVAL_N_ENVS)

    pick_path = _resolve_reward_path(PICK_REWARD_NET_PATH)
    place_path = _resolve_reward_path(PLACE_REWARD_NET_PATH)

    if not os.path.isfile(pick_path):
        raise FileNotFoundError(f"Pick reward net not found at: {PICK_REWARD_NET_PATH}")
    if not os.path.isfile(place_path):
        raise FileNotFoundError(f"Place reward net not found at: {PLACE_REWARD_NET_PATH}")

    pick_reward_net = load_basic_reward_net(base_train_env, pick_path)
    place_reward_net = load_basic_reward_net(base_train_env, place_path)

    calib_env = build_venv("direct", SEED + 10, n_envs=CALIB_N_ENVS)
    pick_scale = calibrate_reward_scale(
        base_env=calib_env,
        reward_net=pick_reward_net,
        device=torch.device(DEVICE),
        stage_index=STAGE_INDEX,
        pick_stage_value=PICK_STAGE_VALUE,
        calibrate_pick=True,
        steps=CALIBRATION_STEPS,
        target_std=TARGET_REWARD_STD,
    )
    place_scale = calibrate_reward_scale(
        base_env=calib_env,
        reward_net=place_reward_net,
        device=torch.device(DEVICE),
        stage_index=STAGE_INDEX,
        pick_stage_value=PICK_STAGE_VALUE,
        calibrate_pick=False,
        steps=CALIBRATION_STEPS,
        target_std=TARGET_REWARD_STD,
    )

    train_env = VecPickPlaceThreeReward(
        base_train_env,
        pick_reward_net=pick_reward_net,
        place_reward_net=place_reward_net,
        device=DEVICE,
        pick_scale=pick_scale,
        place_scale=place_scale,
        w_airl=W_AIRL,
        w_env=W_ENV,
        reward_clip=REWARD_CLIP,
        stage_index=STAGE_INDEX,
        pick_stage_value=PICK_STAGE_VALUE,
    )
    eval_env = VecPickPlaceThreeReward(
        base_eval_env,
        pick_reward_net=pick_reward_net,
        place_reward_net=place_reward_net,
        device=DEVICE,
        pick_scale=pick_scale,
        place_scale=place_scale,
        w_airl=W_AIRL,
        w_env=W_ENV,
        reward_clip=REWARD_CLIP,
        stage_index=STAGE_INDEX,
        pick_stage_value=PICK_STAGE_VALUE,
    )

    train_env = VecMonitor(train_env)
    eval_env = VecMonitor(eval_env)

    model = build_or_resume_ppo(train_env)

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

    reward_tb_cb = RewardComponentsTBCallback()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[ckpt_cb, eval_cb, reward_tb_cb],
        reset_num_timesteps=False,
        tb_log_name="stage2",
    )

    final_path = os.path.join(BEST_DIR, "ppo_stage2_final.zip")
    model.save(final_path)
    print(f"[stage2] Saved final PPO model to: {final_path}")
