import os
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
from imitation.data.types import Trajectory
from stable_baselines3.common.env_util import make_vec_env
from imitation.rewards import reward_nets
from imitation.util.networks import RunningNorm
import pandas as pd

import Env

DEVICE = torch.device("cpu")

REWARD_PATH = "checkpoints/airl_reward_net.pt"
FINAL_REWARD_PATH = "airl_reward_net_final.pt"
POLICY_PATH = "checkpoints/ppo_policy.zip"

def build_env(n_envs=1, render_mode="human", seed=42):
    # NOTE: Do NOT wrap with RolloutInfoWrapper (it’s for single envs, not VecEnv)
    venv = make_vec_env(
        "PalletStacking-v0",
        n_envs=n_envs,
        env_kwargs={"render_mode": render_mode},
        seed=seed,
    )
    return venv

def build_reward_net(venv):
    return reward_nets.BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

def load_airl_reward_net_for_eval(venv, path):
    rn = build_reward_net(venv)
    state = torch.load(path, map_location=DEVICE)
    rn.load_state_dict(state)
    rn.to(DEVICE)
    rn.eval()                           # freeze RunningNorm + dropout/bn if any
    for p in rn.parameters():
        p.requires_grad_(False)
    return rn

def rollout_airl_reward(venv, reward_net, policy=None, n_episodes=3, deterministic=True):
    """
    Runs a rollout in a VecEnv, computes AIRL reward per step, and returns per-episode sums.
    If policy is None, uses random actions from env.action_space.
    """
    # Gymnasium VecEnv reset may return obs or (obs, info)
    reset_out = venv.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    n_envs = venv.num_envs

    ep_airl_sums = np.zeros(n_envs, dtype=np.float64)
    ep_lengths   = np.zeros(n_envs, dtype=np.int32)
    finished_airl = []
    per_step_preview = []  # keep a few lines to print

    episodes_done = 0
    while episodes_done < n_episodes:
        if policy is None:
            actions = np.stack([venv.action_space.sample() for _ in range(n_envs)], axis=0)
        else:
            actions, _ = policy.predict(obs, deterministic=deterministic)

        next_obs, _, dones, infos = venv.step(actions)

        with torch.no_grad():
            t_obs      = torch.as_tensor(obs,      dtype=torch.float32, device=DEVICE)
            t_actions  = torch.as_tensor(actions,  dtype=torch.float32, device=DEVICE)
            t_next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE)
            t_dones    = torch.as_tensor(dones,    dtype=torch.bool,    device=DEVICE)
            airl_r = reward_net(t_obs, t_actions, t_next_obs, t_dones).cpu().numpy()

        ep_airl_sums += airl_r
        ep_lengths   += 1

        # store a few lines to preview
        if len(per_step_preview) < 10:  # first ~10 steps
            per_step_preview.append(dict(airl_reward=float(airl_r[0]), done=bool(dones[0])))

        for i, d in enumerate(dones):
            if d:
                finished_airl.append(float(ep_airl_sums[i]))
                ep_airl_sums[i] = 0.0
                ep_lengths[i]   = 0
                episodes_done  += 1

        obs = next_obs

    return finished_airl, per_step_preview

def airl_reward_on_expert_trajs(expert_trajs, reward_net):
    totals = []
    with torch.no_grad():
        for traj in expert_trajs:
            obs = np.asarray(traj.obs)
            acts = np.asarray(traj.acts)
            T = acts.shape[0]
            o      = torch.as_tensor(obs[:-1], dtype=torch.float32, device=DEVICE)
            a      = torch.as_tensor(acts,     dtype=torch.float32, device=DEVICE)
            next_o = torch.as_tensor(obs[1:],  dtype=torch.float32, device=DEVICE)
            dones  = torch.zeros((T,), dtype=torch.bool, device=DEVICE)
            dones[-1] = True
            r = reward_net(o, a, next_o, dones).cpu().numpy()
            totals.append(float(np.sum(r)))
    return totals

def reward_gradients(reward_net, obs, act, next_obs, done):
    # obs/act/next_obs are 1xD numpy arrays from a real step
    o  = torch.as_tensor(obs,      dtype=torch.float32, requires_grad=True)
    a  = torch.as_tensor(act,      dtype=torch.float32, requires_grad=True)
    o2 = torch.as_tensor(next_obs, dtype=torch.float32, requires_grad=True)
    d  = torch.as_tensor(done,     dtype=torch.bool)

    r = reward_net(o[None], a[None], o2[None], d[None]).squeeze()  # scalar
    r.backward()

    g_obs  = o.grad.detach().cpu().numpy()
    g_act  = a.grad.detach().cpu().numpy()
    g_o2   = o2.grad.detach().cpu().numpy()

    return float(r.item()), g_obs, g_act, g_o2
def export_airl_rollout_to_excel(
    venv,
    reward_net,
    policy=None,
    n_episodes=10,
    deterministic=True,
    out_csv="airl_rewards_log.csv",
    out_xlsx="airl_rewards_log.xlsx",
    extract_fn=None,):

    reset_out = venv.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    n_envs = venv.num_envs
    ep_counters = np.zeros(n_envs, dtype=int)
    step_counters = np.zeros(n_envs, dtype=int)

    rows = []
    episodes_done = 0

    reward_net.eval()
    with torch.no_grad():
        while episodes_done < n_episodes:
            if policy is None:
                actions = np.stack([venv.action_space.sample() for _ in range(n_envs)], axis=0)
            else:
                actions, _ = policy.predict(obs, deterministic=deterministic)

            next_obs, _, dones, infos = venv.step(actions)

            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
            t_actions = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)
            t_next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE)
            t_dones = torch.as_tensor(dones, dtype=torch.bool, device=DEVICE)

            r_airl = reward_net(t_obs, t_actions, t_next_obs, t_dones).cpu().numpy()

            # Log each env separately
            for i in range(n_envs):
                base = {
                    "episode": int(ep_counters[i]),
                    "step": int(step_counters[i]),
                    "env": i,
                    "r_airl": float(r_airl[i]),
                    "done": bool(dones[i]),
                }
                # Optional extras (distance-to-pick, etc.)
                if extract_fn is not None:
                    try:
                        extra = extract_fn(obs[i], actions[i], next_obs[i], infos[i])
                        if extra is not None:
                            base.update({k: float(v) if np.isscalar(v) else v for k, v in extra.items()})
                    except Exception as e:
                        base["extract_error"] = str(e)

                # (Optional) save a few raw obs/action elements for debugging
                # Edit indices to suit your layout if you want them in Excel:
                # base["ee_x"] = float(obs[i, 0]); base["ee_y"] = float(obs[i, 1]); ...

                rows.append(base)
                step_counters[i] += 1

            # Handle episode ends
            for i, d in enumerate(dones):
                if d:
                    ep_counters[i] += 1
                    step_counters[i] = 0
                    episodes_done += 1

            obs = next_obs

    # Build DataFrame and write
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    try:
        df.to_excel(out_xlsx, index=False)  # requires 'openpyxl' or 'xlsxwriter'
    except Exception as e:
        print(f"Could not write ({e}). CSV is still saved at {out_csv}.")
        out_xlsx = None

    # Quick episode summary table (useful in Excel too)
    ep_summary = df.groupby(["env", "episode"])["r_airl"].sum().reset_index(name="airl_return")
    ep_summary.to_csv(out_csv.replace(".csv", "_episodes.csv"), index=False)
    if out_xlsx:
        with pd.ExcelWriter(out_xlsx, mode="a", engine="openpyxl", if_sheet_exists="replace") as xlw:
            df.to_excel(xlw, sheet_name="steps", index=False)
            ep_summary.to_excel(xlw, sheet_name="episodes", index=False)

    print(f"Wrote per-step log → {out_csv}")
    if out_xlsx:
        print(f" Wrote Excel workbook → {out_xlsx}")
    print(f"Episode summary (CSV) → {out_csv.replace('.csv', '_episodes.csv')}")
    return df
if __name__ == "__main__":
    venv = build_env()
    reward_file = REWARD_PATH
    reward_net = load_airl_reward_net_for_eval(venv, reward_file)

    policy = None
    if os.path.isfile(POLICY_PATH):
        policy = PPO.load(POLICY_PATH, env=venv, print_system_info=False)
        print("PPO policy.")

    airl_returns, preview = rollout_airl_reward(venv, reward_net, policy=policy, n_episodes=3)
    print("Per-episode AIRL returns:", np.round(airl_returns, 2))
    print("average return:", float(np.mean(airl_returns)))

    for exp in ["expert_trajs.pkl", "expert_traj.pkl"]:
        if os.path.isfile(exp):
            with open(exp, "rb") as f:
                demos = pickle.load(f)
            if isinstance(demos, Trajectory):
                demos = [demos]
            expert_totals = airl_reward_on_expert_trajs(demos, reward_net)
            print(f"EXPERT! AIRL total reward per traj from {exp}:", np.round(expert_totals, 3))
            print("EXPERT! Mean:", float(np.mean(expert_totals)))
            break
    export_airl_rollout_to_excel(
        venv,
        reward_net,
        policy=None,
        n_episodes=10,
        deterministic=True,
        out_csv="airl_rewards_log.csv",
        out_xlsx="airl_rewards_log.xlsx",
        extract_fn=None, )