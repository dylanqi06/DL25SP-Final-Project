from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import JEPA
import glob
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import copy


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
    # return loss, {"align": align, "var": var, "cov": cov}
    return loss


def train_model(device, model, dataloader, epochs=50, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            states = batch.states.to(device)
            actions = batch.actions.to(device)
            pred, target = model(states, actions)
            loss, metrics = vicreg_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model weights based on training loss.")


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
    model = load_model(device, (C, H, W))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    train_model(device, model, jepa_train_ds)
    model.eval() 
    evaluate_model(device, model, probe_train_ds, probe_val_ds)