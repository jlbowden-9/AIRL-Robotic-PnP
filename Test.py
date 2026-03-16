#!/usr/bin/env python3
"""
test_bc_policy_gym.py

Runs a trained PyTorch imitation model on your Gymnasium PyBullet env (PalletStackEnv).

Assumptions (matches your env code):
- obs is a flat np.float32 vector (shape = (obs_dim,))
- action is 6 absolute joint position targets (shape = (6,))
- env.step(action) applies POSITION_CONTROL to joints [1..6]

Edit ENV_IMPORT below if your env lives in a different module name.
"""
#!/usr/bin/env python3
"""
test_bc_policy_gym.py

Runs a trained PyTorch imitation model on your Gymnasium PyBullet env (PalletStackEnv).

Assumptions (matches your env code):
- obs is a flat np.float32 vector (shape = (obs_dim,))
- action is 6 absolute joint position targets (shape = (6,))
- env.step(action) applies POSITION_CONTROL to joints [1..6]

Edit ENV_IMPORT below if your env lives in a different module name.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path




# =========================
# CONFIG
# =========================
ENV_IMPORT = "Env"
MODEL_PATH = Path(__file__).resolve().parent / "checkpoints" / "best.pth"
RENDER_MODE = "human"             # direct
N_EPISODES = 10
MAX_STEPS_PER_EP = 512
SLEEP_DT = 1.0 / 240.0
USE_SLEEP = True                   # set False to run faster
# =========================


# --- Same network architecture you trained ---
class ImitationNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def infer_io_dims_from_state_dict(state_dict):
    # First layer: (64, input_dim)
    w0 = state_dict["net.0.weight"]
    input_dim = int(w0.shape[1])
    # Last layer: (output_dim, 64)
    w_last = state_dict["net.4.weight"]
    output_dim = int(w_last.shape[0])
    return input_dim, output_dim


def load_model_and_norm(model_path):
    ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        X_mean = ckpt.get("X_mean", None)
        X_std = ckpt.get("X_std", None)
    else:
        state_dict = ckpt
        X_mean, X_std = None, None

    in_dim, out_dim = infer_io_dims_from_state_dict(state_dict)
    model = ImitationNetwork(in_dim, out_dim)
    model.load_state_dict(state_dict)
    model.eval()

    if X_mean is not None and not torch.is_tensor(X_mean):
        X_mean = torch.tensor(X_mean, dtype=torch.float32)
    if X_std is not None and not torch.is_tensor(X_std):
        X_std = torch.tensor(X_std, dtype=torch.float32)

    return model, in_dim, out_dim, X_mean, X_std


def main():
    # Import Enviroment
    env_mod = __import__(ENV_IMPORT, fromlist=["PalletStackEnv"])
    PalletStackEnv = getattr(env_mod, "PalletStackEnv")
    env = PalletStackEnv(render_mode=RENDER_MODE)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    model, input_dim, output_dim, X_mean, X_std = load_model_and_norm(MODEL_PATH)
    print(f"[OK] Loaded model: input_dim={input_dim}, output_dim={output_dim}")

    # Sanity-check with env spaces
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    print(f"[OK] Env obs_dim={obs_dim}, act_dim={act_dim}")

    if input_dim != obs_dim:
        print(f"[WARN] Model expects obs_dim={input_dim} but env provides {obs_dim}. "
              f"If you trained on a different observation vector, you must match it here.")
    if output_dim != act_dim:
        print(f"[WARN] Model outputs act_dim={output_dim} but env expects {act_dim}. "
              f"Actions will be truncated/padded.")

    successes = 0

    for ep in range(1, N_EPISODES + 1):
        # NOTE: your env.reset() calls p.removeConstraint(self.cid) unconditionally.
        # If you ever get a PyBullet error here, add a guard in your env:
        #   if self.cid != 0: p.removeConstraint(self.cid)
        obs, info = env.reset()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        ep_success = False
        last_error = None

        for step in range(MAX_STEPS_PER_EP):
            x = torch.from_numpy(obs).float().unsqueeze(0)  # (1, obs_dim)

            # Apply training normalization if present
            if X_mean is not None and X_std is not None:
                x = (x - X_mean.reshape(1, -1)) / (X_std.reshape(1, -1) + 1e-8)

            with torch.no_grad():
                a = model(x).squeeze(0).cpu().numpy().astype(np.float32)



            # Clip to env action bounds (important)
            a = np.clip(a, env.action_space.low, env.action_space.high).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(a)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)

            # Your obs packs "error" right after grip_success:
            # q(6) + grip_success(1) + error(1) + ...
            if obs.size >= 8:
                last_error = float(obs[7])

            if info.get("is_success", False):
                ep_success = True

            if terminated or truncated:
                break

            if USE_SLEEP and RENDER_MODE == "human":
                time.sleep(SLEEP_DT)

        successes += int(ep_success)
        print(f"Episode {ep}/{N_EPISODES} | success={ep_success} | steps={step+1} | last_error={last_error}")

    print(f"\nSuccess rate: {successes}/{N_EPISODES} = {successes / max(1, N_EPISODES):.2%}")
    env.close()


if __name__ == "__main__":
    main()
