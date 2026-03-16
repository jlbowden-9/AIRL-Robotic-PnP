import os, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from imitation.data import rollout
from imitation.data.types import Trajectory

# =======================
# CONFIG
# =======================
PKL_PATH = "expert_trajs_Place.pkl"

NORMALIZE_OBS = True          # True means we are using normilize
BATCH_SIZE = 1024
LR = 1e-3
EPOCHS = 100000

PRINT_EVERY = 500              # print every N epochs
SAVE_EVERY = 1000             # save periodic checkpoints
CKPT_DIR = Path("checkpoints")
RESUME_FROM_LAST = False       # resume from checkpoints/last.pth if present

EARLY_STOP_PATIENCE = 5000    # epochs without improvement before stopping
MIN_DELTA = 1e-9              # required improvement in val loss
# =======================

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LAST_CKPT_PATH = CKPT_DIR / "last.pth"
BEST_CKPT_PATH = CKPT_DIR / "best.pth"

# ---------- Load expert trajectories ----------
if not os.path.isfile(PKL_PATH):
    raise FileNotFoundError(f"Could not find {PKL_PATH}")

with open(PKL_PATH, "rb") as f:
    obj = pickle.load(f)

if isinstance(obj, Trajectory):
    expert_trajs = [obj]
elif isinstance(obj, (list, tuple)):
    expert_trajs = list(obj)
else:
    raise TypeError(f"Unexpected type in pkl: {type(obj)}")

# ---------- Flatten to (obs, acts) pairs ----------
transitions = rollout.flatten_trajectories(expert_trajs)
X = np.asarray(transitions.obs, dtype=np.float32)
y = np.asarray(transitions.acts, dtype=np.float32)

if X.ndim == 1: X = X[:, None]
if y.ndim == 1: y = y[:, None]

# ---------- Normalization stats ----------
# (store as float32 so inference doesn't accidentally become float64)
X_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
X_std  = (X.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

if NORMALIZE_OBS:
    X = (X - X_mean) / X_std

X_t = torch.from_numpy(X)
y_t = torch.from_numpy(y)

# ---------- Model ----------
class ImitationNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    def forward(self, x):
        return self.net(x)

input_dim = X_t.shape[1]
output_dim = y_t.shape[1]

model = ImitationNetwork(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ---------- DataLoader ----------
dataset = TensorDataset(X_t, y_t)
n_train = int(0.8 * len(dataset))
n_val = len(dataset) - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Checkpoint helpers ----------
def save_ckpt(path: Path, epoch: int, best_val: float):
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "normalize_obs": NORMALIZE_OBS,
        "X_mean": torch.from_numpy(X_mean),
        "X_std": torch.from_numpy(X_std),
    }, str(path))

start_epoch = 1
best_val = float("inf")
epochs_no_improve = 0

# ---------- Resume ----------
if RESUME_FROM_LAST and LAST_CKPT_PATH.exists():
    ckpt = torch.load(str(LAST_CKPT_PATH), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    print(f"[RESUME] from {LAST_CKPT_PATH} at epoch {start_epoch}, best_val={best_val:.6g}")

# ---------- Train ----------
try:
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss /= max(1, len(val_ds))

        # Print sometimes
        if epoch == start_epoch or epoch % PRINT_EVERY == 0:
            print(f"Epoch {epoch} | train {train_loss:.6f} | val {val_loss:.6f}")

        # Always save "last"
        save_ckpt(LAST_CKPT_PATH, epoch, best_val)

        # Periodic snapshot
        if epoch % SAVE_EVERY == 0:
            save_ckpt(CKPT_DIR / f"epoch_{epoch}.pth", epoch, best_val)

        # Best checkpoint + early stopping
        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            epochs_no_improve = 0
            save_ckpt(BEST_CKPT_PATH, epoch, best_val)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"[EARLY STOP] No val improvement for {EARLY_STOP_PATIENCE} epochs. "
                      f"Best val={best_val:.6f}.")
                break

except KeyboardInterrupt:
    print("\n[INTERRUPT] Saving last checkpoint and exiting...")
    save_ckpt(LAST_CKPT_PATH, epoch, best_val)

print(f"[DONE] Best checkpoint: {BEST_CKPT_PATH}")
print(f"[DONE] Last checkpoint: {LAST_CKPT_PATH}")
