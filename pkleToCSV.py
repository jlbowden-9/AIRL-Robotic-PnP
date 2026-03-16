import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path

# --- helpers ---
def _to_2d(a):
    a = np.asarray(a)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a[:, None]
    return a

def _guess_key(d, candidates):
    return next((k for k in candidates if k in d), None)

def _traj_like_to_rows(traj, ep_id):
    """Return list[dict] rows for a single trajectory (dict-like or imitation Trajectory)."""
    # Try imitation Trajectory
    try:
        from imitation.data.types import Trajectory as ImitTrajectory
        if isinstance(traj, ImitTrajectory):
            obs = _to_2d(traj.obs)
            acts = _to_2d(traj.acts)
            infos = getattr(traj, "infos", None)
            T = min(len(obs), len(acts))
            rows = []
            for t in range(T):
                row = {"episode": ep_id, "t": t}
                for j, v in enumerate(np.atleast_1d(obs[t])):
                    row[f"obs_{j}"] = float(v)
                for j, v in enumerate(np.atleast_1d(acts[t])):
                    row[f"act_{j}"] = float(v)
                if isinstance(infos, list) and t < len(infos) and isinstance(infos[t], dict):
                    for k, v in infos[t].items():
                        row[f"info.{k}"] = v
                rows.append(row)
            return rows
    except Exception:
        pass

    # Dict-like (common: {"obs": (T,d), "acts": (T,m), "rews": (T,), "dones": (T,), "infos": list})
    d = dict(traj) if isinstance(traj, dict) else {k: getattr(traj, k) for k in dir(traj) if not k.startswith("_")}
    obs_k  = _guess_key(d, ["obs", "observations", "states", "observation"])
    act_k  = _guess_key(d, ["acts", "actions", "action"])
    rew_k  = _guess_key(d, ["rews", "rewards", "reward"])
    done_k = _guess_key(d, ["dones", "terminals", "done"])
    info_k = "infos" if "infos" in d else None

    obs  = _to_2d(d[obs_k]) if obs_k and d[obs_k] is not None else None
    acts = _to_2d(d[act_k]) if act_k and d[act_k] is not None else None
    rews = np.asarray(d[rew_k]).reshape(-1) if rew_k and d[rew_k] is not None else None
    dones = np.asarray(d[done_k]).reshape(-1) if done_k and d[done_k] is not None else None
    infos = d.get(info_k, None)

    arrays = [x for x in [obs, acts, rews, dones] if x is not None]
    if not arrays:
        return [{"episode": ep_id, "t": 0}]  # fallback

    T = max(getattr(x, "shape", (len(x),))[0] for x in arrays)
    rows = []
    for t in range(T):
        row = {"episode": ep_id, "t": t}
        if obs is not None and t < len(obs):
            for j, v in enumerate(np.atleast_1d(obs[t])):
                row[f"obs_{j}"] = float(v)
        if acts is not None and t < len(acts):
            for j, v in enumerate(np.atleast_1d(acts[t])):
                row[f"act_{j}"] = float(v)
        if rews is not None and t < len(rews):
            row["reward"] = float(rews[t])
        if dones is not None and t < len(dones):
            row["done"] = bool(dones[t])
        if isinstance(infos, list) and t < len(infos) and isinstance(infos[t], dict):
            for k, v in infos[t].items():
                row[f"info.{k}"] = v
        rows.append(row)
    return rows

def trajs_to_df(obj):
    # Normalize "episodes" container to a list
    if isinstance(obj, dict):
        # Sometimes data is dict of lists (key -> list per episode)
        first_val = next(iter(obj.values()))
        if isinstance(first_val, list) and all(len(obj[k]) == len(first_val) for k in obj):
            episodes = len(first_val)
            trajs = [{k: obj[k][i] for k in obj} for i in range(episodes)]
        else:
            trajs = [obj]
    elif isinstance(obj, (list, tuple)):
        trajs = list(obj)
    else:
        trajs = [obj]

    all_rows = []
    for i, traj in enumerate(trajs):
        all_rows.extend(_traj_like_to_rows(traj, i))
    return pd.DataFrame(all_rows).sort_values(["episode", "t"]).reset_index(drop=True)

# --- run ---
pkl_path = Path("expert_trajs_Pick.pkl")
with pkl_path.open("rb") as f:
    obj = pkl.load(f)

df = trajs_to_df(obj)
out = "expert_trajs_Pick.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} with shape {df.shape}")
