# activities.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from temporalio import activity

# =====================
# Dataset paths
# =====================
X_PATH = "/home/exouser/02_triangular_mesh_autoencoder/Dataset/LDC/skelneton_lid_driven_cavity_X.npz"
Y_PATH = "/home/exouser/02_triangular_mesh_autoencoder/Dataset/LDC/skelneton_lid_driven_cavity_Y.npz"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Dataset
# =====================
class LDCDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])

# =====================
# Model (UNCHANGED)
# =====================
class CNO_LReLu(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bicubic", antialias=True)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=0.5, mode="bicubic", antialias=True)
        return x

class CNOBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.leaky_relu(self.c1(x)))

class CNO2d(nn.Module):
    def __init__(self, in_dim=3, out_dim=4, width=32):
        super().__init__()
        self.lift = nn.Conv2d(in_dim, width, 1)
        self.res1 = ResidualBlock(width)
        self.res2 = ResidualBlock(width)
        self.project = nn.Conv2d(width, out_dim, 1)

    def forward(self, x):
        x = self.lift(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.project(x)

# =====================
# Training Activity
# =====================
@activity.defn
def train_cno_model(epochs: int = 50, batch_size: int = 16):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.load(X_PATH)["data"].astype(np.float32)
    Y = np.load(Y_PATH)["data"].astype(np.float32)

    n_train = int(0.8 * len(X))
    train_ds = LDCDataset(X[:n_train], Y[:n_train])
    test_ds  = LDCDataset(X[n_train:], Y[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    model = CNO2d().to(device)
    opt = AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                test_loss += loss_fn(model(xb), yb).item() * xb.size(0)
        test_loss /= len(test_ds)

        print(f"[Epoch {epoch+1}/{epochs}] Train: {train_loss:.6f} | Test: {test_loss:.6f}")

    # =====================
    # Save GT vs Prediction
    # =====================
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X[0:1]).to(device)
        pred = model(x).cpu().numpy()[0]

    fig, axs = plt.subplots(2, 4, figsize=(16, 6))
    for i in range(4):
        axs[0, i].imshow(Y[0, i], cmap="jet")
        axs[0, i].set_title(f"GT {i}")
        axs[1, i].imshow(pred[i], cmap="jet")
        axs[1, i].set_title(f"Pred {i}")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "ldc_groundtruth_vs_prediction.png")
    plt.savefig(save_path)
    plt.close()

    return save_path
