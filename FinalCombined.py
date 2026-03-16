
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


import Env


def extract_stage(obs, info, stage_index: int) -> float:

    if isinstance(info, dict):
        for k in ("stage", "Stage", "stage_id", "stage_idx"):
            if k in info:
                try:
                    return float(info[k])
                except Exception:
                    pass

    if isinstance(obs, dict):
        for k in ("stage", "Stage", "stage_id", "stage_idx"):
            if k in obs:
                try:
                    v = obs[k]
                    # handle scalar / array
                    return float(np.asarray(v).item())
                except Exception:
                    pass
        raise ValueError(
            "Obs is a dict but no stage key was found. Add it or update extract_stage()."
        )

    obs_arr = np.asarray(obs)
    if obs_arr.ndim != 1:
        # For demo purposes we expect a single env (1D obs). If you use VecEnv, set n_envs=1
        # and SB3 will still often give (obs_dim,) here, but handle just in case:
        if obs_arr.ndim == 2 and obs_arr.shape[0] == 1:
            obs_arr = obs_arr[0]
        else:
            raise ValueError(f"Unexpected obs shape {obs_arr.shape}. Use n_envs=1 for demo.")

    if stage_index < 0 or stage_index >= obs_arr.shape[0]:
        raise IndexError(f"STAGE_INDEX={stage_index} out of bounds for obs_dim={obs_arr.shape[0]}")

    return float(obs_arr[stage_index])


class StageSwitchPolicy:
    def __init__(
        self,
        pick_model: PPO,
        place_model: PPO,
        stage_index: int,
        pick_stage_value: float,
        threshold: float = 0.5,
        verbose: bool = True,
    ):
        self.pick_model = pick_model
        self.place_model = place_model
        self.stage_index = stage_index
        self.pick_stage_value = float(pick_stage_value)
        self.threshold = float(threshold)
        self.verbose = verbose
        self._last_mode = None  # "pick" or "place"

    def _is_pick(self, stage: float) -> bool:
        return abs(stage - self.pick_stage_value) < self.threshold

    def predict(self, obs, state=None, episode_start=None, deterministic=False, info=None):
        stage = extract_stage(obs, info or {}, self.stage_index)
        mode = "pick" if self._is_pick(stage) else "place"
        model = self.pick_model if mode == "pick" else self.place_model

        if self.verbose and mode != self._last_mode:
            print(f"[SWITCH] -> {mode.upper()} (stage={stage:.3f})")
            self._last_mode = mode

        return model.predict(obs, state=state, episode_start=episode_start, deterministic=deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PalletStacking-v0",
                        help="Your registered Gymnasium env id.")
    parser.add_argument("--pick", type=str, default="Final_PIck.zip",
                        help="Path to pick PPO .zip (SB3 save file).")
    parser.add_argument("--place", type=str, default="Final_Place.zip",
                        help="Path to place PPO .zip (SB3 save file).")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-index", type=int, default=3,
                        help="Index in observation vector that stores stage.")
    parser.add_argument("--pick-stage", type=float, default=0.0,
                        help="Value of stage that means 'pick'.")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--no-render", action="store_true", help="Disable rendering.")
    args = parser.parse_args()

    # Load models (your uploaded files are already SB3-format zips)
    pick_model = PPO.load(args.pick, device="auto")
    place_model = PPO.load(args.place, device="auto")

    # Create env (for PyBullet GUI, you usually want render_mode="human")
    render_mode = None if args.no_render else "human"
    env = gym.make(args.env_id, render_mode=render_mode)

    combined = StageSwitchPolicy(
        pick_model=pick_model,
        place_model=place_model,
        stage_index=args.stage_index,
        pick_stage_value=args.pick_stage,
        threshold=0.5,
        verbose=True,
    )

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        terminated = truncated = False
        ep_ret = 0.0
        steps = 0

        while not (terminated or truncated):
            action, _ = combined.predict(
                obs,
                deterministic=args.deterministic,
                info=info,  # lets us read info["stage"] if your env provides it
            )

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            steps += 1

        print(f"[EP {ep}] return={ep_ret:.3f} steps={steps} terminated={terminated} truncated={truncated}")

    env.close()


if __name__ == "__main__":
    main()
