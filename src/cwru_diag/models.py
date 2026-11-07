"""模型：经典基线（SVM / 随机森林）与 1D-CNN。"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset


def make_svm(C: float = 10.0, gamma: str | float = "scale", seed: int = 0,
             class_weight: str | dict | None = "balanced") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, gamma=gamma, kernel="rbf", random_state=seed,
                    class_weight=class_weight)),
    ])


def make_rf(n_estimators: int = 400, max_depth: int | None = None,
            min_samples_leaf: int = 1, seed: int = 0,
            class_weight: str | dict | None = "balanced") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, n_jobs=-1, random_state=seed,
            class_weight=class_weight,
        )),
    ])


class BearingCNN(nn.Module):
    """轻量 1D-CNN：3 个卷积块 + 全局平均池化。输入 (B, 1, L)。"""

    def __init__(self, n_classes: int = 4, width: int = 32, dropout: float = 0.3):
        super().__init__()

        def block(cin: int, cout: int, k: int = 5, pool: int = 2) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool),
            )

        self.features = nn.Sequential(
            block(1, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _standardize_per_window(X: np.ndarray) -> np.ndarray:
    """逐窗标准化：去掉载荷/量程带来的整体幅值差异，只保留波形形态。"""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mu) / sd


def augment_noise(X: np.ndarray, snr_choices: list, seed: int = 0) -> np.ndarray:
    """噪声增广：每个窗口独立随机抽一个 SNR（dB），None 表示保持干净。

    噪声功率按该窗口自身的平均功率换算（与 08 脚本的 add_noise 同口径），
    返回 dtype 与输入一致。
    """
    if not snr_choices:
        raise ValueError("snr_choices 不能为空")
    rng = np.random.default_rng(seed)
    Xf = X.astype(np.float64, copy=True)
    picks = rng.integers(0, len(snr_choices), size=len(Xf))
    p_sig = (Xf ** 2).mean(axis=1, keepdims=True)
    for i, snr in enumerate(snr_choices):
        mask = picks == i
        if snr is None or not mask.any():
            continue
        p_noise = p_sig[mask] / (10 ** (float(snr) / 10.0))
        Xf[mask] += rng.normal(size=(int(mask.sum()), X.shape[1])) * np.sqrt(p_noise)
    return Xf.astype(X.dtype if np.issubdtype(X.dtype, np.floating) else np.float32)


@dataclass
class CNNResult:
    model: BearingCNN
    history: dict[str, list[float]]
    best_epoch: int
    best_val_acc: float


def train_cnn(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              n_classes: int = 4, epochs: int = 30, batch_size: int = 128,
              lr: float = 1e-3, weight_decay: float = 1e-4, dropout: float = 0.3,
              seed: int = 42, device: str | None = None,
              patience: int = 8, verbose: bool = True,
              aug_snrs: list | None = None) -> CNNResult:
    """训练 1D-CNN，按验证集准确率早停并回滚到最优权重。

    aug_snrs：若给出（如 [None, 10, 6, 2, 0, -4]），每个 epoch 对训练窗口
    重新抽 SNR 加噪（噪声增广）；验证集始终干净。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = torch.tensor(_standardize_per_window(X_train)[:, None, :], dtype=torch.float32)
    Xva = torch.tensor(_standardize_per_window(X_val)[:, None, :], dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    yva = torch.tensor(y_val, dtype=torch.long)

    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=512, shuffle=False)

    model = BearingCNN(n_classes=n_classes, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_state, best_acc, best_epoch, bad = None, -1.0, -1, 0

    for epoch in range(1, epochs + 1):
        if aug_snrs:
            # 每个 epoch 重新加噪，避免模型记住固定的噪声实现
            X_ep = augment_noise(X_train, aug_snrs, seed=seed * 10007 + epoch)
            Xtr_ep = torch.tensor(_standardize_per_window(X_ep)[:, None, :], dtype=torch.float32)
        else:
            Xtr_ep = Xtr
        train_loader = DataLoader(TensorDataset(Xtr_ep, ytr), batch_size=batch_size, shuffle=True)

        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        sched.step()

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()

        tr_loss = running / len(Xtr)
        va_loss = val_loss / len(Xva)
        va_acc = correct / len(Xva)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    epoch {epoch:02d}  train_loss={tr_loss:.4f}  "
                  f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")

        if va_acc > best_acc + 1e-6:
            best_acc, best_epoch, bad = va_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"    早停于 epoch {epoch}，最优 epoch {best_epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return CNNResult(model=model, history=history, best_epoch=best_epoch, best_val_acc=best_acc)


@torch.no_grad()
def predict_cnn(model: BearingCNN, X: np.ndarray, batch_size: int = 1024,
                device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    xt = torch.tensor(_standardize_per_window(X)[:, None, :], dtype=torch.float32)
    preds = []
    for i in range(0, len(xt), batch_size):
        logits = model(xt[i:i + batch_size].to(device))
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)
