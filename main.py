from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
from models import JEPA
import glob
from torch import nn
from torch.nn import functional as F

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "D:\\ProgramData\\DL25SP"

    jepa_train_ds = create_wall_dataloader(
        data_path=f"{data_path}\\train",
        probing=False,
        device=device,
        train=True,
    )

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}\\probe_normal\\train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}\\probe_normal\\val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}\\probe_wall\\val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }

    return jepa_train_ds, probe_train_ds, probe_val_ds


def load_model(device):
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    model = JEPA(embedding_dim=786, action_dim=2).to(device)

    return model

def vicreg_loss(p, t):

    var_w=1.0
    cov_w=0.01
    gamma=1.0

    """p,t: [B,T-1,D]"""
    B, Tm1, D = p.shape
    # 1. alignment (MSE)
    align = F.mse_loss(p, t)

    # 2. variance regulariser  (each dim std ≥ γ)
    z = p.reshape(B*Tm1, D)                 #  grads OK
    std = torch.sqrt(z.var(dim=0) + 1e-04)
    var = torch.mean(F.relu(gamma - std))

    # 3. covariance regulariser (off‑diag → 0)
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (B*Tm1 - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov = (off_diag ** 2).sum() / D

    loss = align + var_w * var + cov_w * cov
    return loss, {"align": align, "var": var, "cov": cov}

def train_model(device, model, dataloader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            states = batch.states.to(device)      # [B, T, 3, 64, 64]
            actions = batch.actions.to(device)    # [B, T-1, 2]

            pred, target = model(states, actions) # [B, T-1, D]
            loss, metrics = vicreg_loss(pred, target) # [B, T-1, D]
            # loss = criterion(pred, target)
            # print({k:f"{v.item():.3f}" for k,v in metrics.items()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target()  # Update target encoder
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    model = load_model(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    jepa_train_ds, probe_train_ds, probe_val_ds = load_data(device)
    # train_model(device, model, jepa_train_ds)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
