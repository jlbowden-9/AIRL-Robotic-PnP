import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

import pybullet as p
from collections import deque

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
                    return float(np.asarray(v).item())
                except Exception:
                    pass
        raise ValueError("Obs is a dict but no stage key was found. Update extract_stage().")

    obs_arr = np.asarray(obs)
    if obs_arr.ndim != 1:
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


def _bool_from_info(info: dict, keys) -> bool:
    """Robustly read a boolean-ish flag from info using a list of possible keys."""
    for k in keys:
        if k in info:
            v = info[k]
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, np.integer)):
                return bool(int(v))
            if isinstance(v, (float, np.floating)) and v in (0.0, 1.0):
                return bool(int(v))
    return False


class EndEffectorTrail:
    """
    Draws a persistent line trail for the end-effector in PyBullet GUI using addUserDebugLine.
    Requires the env to expose: client, robot_id, end_effector_index.
    """
    def __init__(self, unwrapped_env, enabled=True, max_segments=2000, line_width=2.0):
        self.env = unwrapped_env
        self.enabled = enabled
        self.max_segments = int(max_segments)
        self.line_width = float(line_width)
        self._ids = deque()
        self._prev = None
        self._ok = all(hasattr(self.env, a) for a in ("client", "robot_id", "end_effector_index"))

        if enabled and not self._ok:
            missing = [a for a in ("client", "robot_id", "end_effector_index") if not hasattr(self.env, a)]
            print(f"[TRAIL] Disabled (env missing attrs: {missing}).")

    def clear(self):
        if not self.enabled or not self._ok:
            self._prev = None
            self._ids.clear()
            return
        while self._ids:
            p.removeUserDebugItem(self._ids.popleft(), physicsClientId=self.env.client)
        self._prev = None

    def update(self):
        if not self.enabled or not self._ok:
            return

        ee_pos = p.getLinkState(
            self.env.robot_id,
            self.env.end_effector_index,
            physicsClientId=self.env.client
        )[4]

        if self._prev is not None:
            uid = p.addUserDebugLine(
                self._prev,
                ee_pos,
                lineWidth=self.line_width,
                lifeTime=0,  # persistent
                physicsClientId=self.env.client
            )
            self._ids.append(uid)

            while len(self._ids) > self.max_segments:
                p.removeUserDebugItem(self._ids.popleft(), physicsClientId=self.env.client)

        self._prev = ee_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="PalletStacking-v0",
                        help="Your registered Gymnasium env id.")
    parser.add_argument("--pick", type=str, default="Final_Pick.zip",
                        help="Path to pick PPO .zip (SB3 save file).")
    parser.add_argument("--place", type=str, default="Final_Place.zip",
                        help="Path to place PPO .zip (SB3 save file).")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-index", type=int, default=3,
                        help="Index in observation vector that stores stage.")
    parser.add_argument("--pick-stage", type=float, default=0.0,
                        help="Value of stage that means 'pick'.")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--no-render", action="store_true", help="Disable rendering.")

    # Trail controls
    parser.add_argument("--trail", action="store_true", help="Draw end-effector trail (GUI only).")
    parser.add_argument("--trail-max", type=int, default=2000, help="Max trail segments to keep.")

    args = parser.parse_args()

    pick_model = PPO.load(args.pick, device="auto")
    place_model = PPO.load(args.place, device="auto")

    render_mode = None #if args.no_render else "human"
    env = gym.make(args.env_id, render_mode=render_mode)

    combined = StageSwitchPolicy(
        pick_model=pick_model,
        place_model=place_model,
        stage_index=args.stage_index,
        pick_stage_value=args.pick_stage,
        threshold=0.5,
        verbose=True,
    )

    # Trail (uses unwrapped env for PyBullet IDs)
    trail = EndEffectorTrail(env.unwrapped, enabled=1, max_segments=args.trail_max)

    # Completion statistics
    total_eps = 0
    pick_success_eps = 0
    place_success_eps = 0
    seated_success_eps = 0
    full_success_eps = 0  # both pick and place within episode

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        trail.clear()

        terminated = truncated = False
        ep_ret = 0.0
        steps = 0

        # Episode flags (set True once we observe success in info)
        did_pick = False
        did_place = False
        is_seated = False

        while not (terminated or truncated):
            action, _ = combined.predict(
                obs,
                deterministic=args.deterministic,
                info=info,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            steps += 1

            # Update trail after stepping
            trail.update()
            # Try to infer success flags from info (adapt keys to your env if needed)
            did_pick = did_pick or _bool_from_info(info, keys=("picked",))
            did_place = did_place or _bool_from_info(info, keys=("is_success",))
            is_seated = is_seated or _bool_from_info(info, keys=("seated",))
            #print(did_pick, did_place, is_seated)
        total_eps += 1
        pick_success_eps += int(did_pick)
        place_success_eps += int(did_place)
        seated_success_eps += int(is_seated)
        full_success_eps += int(did_pick and did_place)

        print(
            f"[EP {ep}] return={ep_ret:.3f} steps={steps} "
            f"pick={did_pick} place={did_place} terminated={terminated} truncated={truncated}"
        )

    env.close()

    # Final stats
    def pct(x): return 100.0 * x / max(1, total_eps)

    print("\n=== FINAL COMPLETION STATS ===")
    print(f"Episodes: {total_eps}")
    print(f"Pick success:  {pick_success_eps}/{total_eps} ({pct(pick_success_eps):.1f}%)")
    print(f"Place success: {place_success_eps}/{total_eps} ({pct(place_success_eps):.1f}%)")
    print(f"Seated success: {place_success_eps}/{total_eps} ({pct(place_success_eps):.1f}%)")
    print(f"Full PnP:      {full_success_eps}/{total_eps} ({pct(full_success_eps):.1f}%)")


if __name__ == "__main__":
    main()