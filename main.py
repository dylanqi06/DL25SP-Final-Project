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
    data_path = "/scratch/DL25SP"

    jepa_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/train",
        probing=False,
        device=device,
        train=True,
    )

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
    }

    return jepa_train_ds, probe_train_ds, probe_val_ds


def load_model(device, input_shape):
    """Initialize JEPA model with dynamic input shape."""
    print(f"[load_model] input_shape={input_shape}")
    model = JEPA(input_shape=input_shape, embedding_dim=256, action_dim=2)
    # print(f"[load_model] encoder.in_channels={model.encoder.cnn[0].in_channels}")
    model.to(device)
    return model

def vicreg_loss(p, t):
    var_w = 1.0
    cov_w = 0.01
    gamma = 1.0
    B, Tm1, D = p.shape
    align = F.mse_loss(p, t)
    z = p.reshape(B * Tm1, D)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(gamma - std))
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (B * Tm1 - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov = (off_diag ** 2).sum() / D
    loss = align + var_w * var + cov_w * cov
    return loss, {"align": align, "var": var, "cov": cov}

def train_model(device, model, dataloader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    print("[DEBUG] Entering training loop")
    for epoch in range(epochs):
        print(f"[DEBUG] Starting epoch {epoch + 1}")
        total_loss = 0
        for batch in dataloader:
            print("[DEBUG] Got batch")
            states = batch.states.to(device)
            actions = batch.actions.to(device)
            pred, target = model(states, actions)
            loss, metrics = vicreg_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target()
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
    jepa_train_ds, probe_train_ds, probe_val_ds = load_data(device)

    # Inspect a batch to get channel and spatial dimensions
    sample_batch = next(iter(probe_train_ds))
    C = sample_batch.states.size(2)
    H = sample_batch.states.size(3)
    W = sample_batch.states.size(4)
    print(f"[main] Detected state shape: {sample_batch.states.shape}")
    model = load_model(device, (C, H, W))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    train_model(device, model, jepa_train_ds)
    model.eval() 
    evaluate_model(device, model, probe_train_ds, probe_val_ds)