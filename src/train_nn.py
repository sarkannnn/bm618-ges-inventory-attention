"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Derin Öğrenme Modelleri Eğitim Boru Hattı (src/train_nn.py)

EĞİTİM VE DONDURULMUŞ PROTOKOL ŞARTLARI:
- Optimizer: AdamW (weight_decay=1e-4)
- Batch Size: 16
- Kayıp Fonksiyonu: MSE (log1p hedefi üzerinde)
- Erken Durdurma / Model Seçimi: İç doğrulamada tesis dengeli MAE (kWp, ters dönüşümle)
- Sabır (Patience): 15 epoch, Maksimum: 150 epoch
- D1, D2, D3 aynı sözlüğü, aynı hedefi, aynı bölmeleri kullanır.
- Yalnız in_main_regression=True (151 ana envanter) kayıtları kullanılır.
"""

import os
import sys
import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_prep import (
    load_audited_data,
    prepare_tokenized_dataset,
    log1p_to_power,
    power_to_log1p
)
from src.metrics import (
    compute_facility_balanced_mae,
    compute_all_metrics
)


class GESTokenDataset(Dataset):
    """PyTorch Dataset kapsayıcısı."""
    def __init__(self, dataset_dict: Dict):
        self.token_ids = dataset_dict["token_ids"]
        self.padding_mask = dataset_dict["key_padding_mask"]
        self.targets_z = dataset_dict["targets_z"]
        self.y_true = dataset_dict["y_true"]
        self.record_ids = dataset_dict["record_ids"]
        self.entity_groups = dataset_dict["entity_groups"]

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "token_ids": self.token_ids[idx],
            "padding_mask": self.padding_mask[idx],
            "target_z": self.targets_z[idx] if self.targets_z is not None else torch.tensor(0.0),
            "y_true": self.y_true[idx] if self.y_true is not None else 0.0,
            "record_id": self.record_ids[idx],
            "entity_group": self.entity_groups[idx]
        }


def collate_ges_batch(batch: List[Dict]) -> Dict:
    """Batch elemanlarını tensörlere dönüştürür."""
    token_ids = torch.stack([item["token_ids"] for item in batch])
    padding_mask = torch.stack([item["padding_mask"] for item in batch])
    targets_z = torch.stack([item["target_z"] for item in batch])
    y_true = np.array([item["y_true"] for item in batch], dtype=np.float64)
    record_ids = [item["record_id"] for item in batch]
    entity_groups = [item["entity_group"] for item in batch]

    return {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
        "targets_z": targets_z,
        "y_true": y_true,
        "record_ids": record_ids,
        "entity_groups": entity_groups
    }


def train_neural_net(
    model: nn.Module,
    train_data: Dict,
    val_data: Optional[Dict] = None,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 150,
    patience: int = 15,
    seed: int = 42,
    verbose: bool = False
) -> Dict:
    """
    Verilen derin öğrenme modelini (D1/D2/D3) dondurulmuş protokole göre eğitir.
    - Tohum sabitlenir (ağırlık ilklendirme ve DataLoader karıştırma).
    - Her epoch sonunda doğrulama kümesinde tesis dengeli MAE (kWp) hesaplanır.
    - En düşük iç doğrulama tesis dengeli MAE'sine ulaşılan en iyi epoch ve ağırlıklar saklanır.
    """
    # Sabit tohum
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = GESTokenDataset(train_data)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_ges_batch
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "val_mae": []
    }

    best_val_mae = float("inf")
    best_epoch = 1
    best_state_dict = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            b_ids = batch["token_ids"]
            b_mask = batch["padding_mask"]
            b_targets = batch["targets_z"]

            preds = model(b_ids, padding_mask=b_mask)
            loss = criterion(preds, b_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(1, num_batches)
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)

        # Doğrulama kümesi değerlendirmesi
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                v_ids = val_data["token_ids"]
                v_mask = val_data["key_padding_mask"]
                v_preds_z = model(v_ids, padding_mask=v_mask)

                # Ters dönüşüm: log1p -> kWp
                v_preds_kwp = log1p_to_power(v_preds_z.cpu().numpy())
                v_true_kwp = val_data["y_true"]
                v_groups = val_data["entity_groups"]

                # Birincil metrik: Tesis dengeli MAE
                val_mae = compute_facility_balanced_mae(v_true_kwp, v_preds_kwp, v_groups)

            history["val_mae"].append(val_mae)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f} | Val Tesis Dengeli MAE: {val_mae:.2f} kWp")

            # Erken durdurma kontrolü (Tesis dengeli MAE'ye göre)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Erken durdurma tetiklendi! Epoch: {epoch}, En iyi epoch: {best_epoch} (Val MAE: {best_val_mae:.2f} kWp)")
                    break
        else:
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"  Epoch {epoch:03d} | Train MSE: {avg_train_loss:.4f}")

    # En iyi ağırlıkları modele geri yükle
    if val_data is not None and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_state_dict": best_state_dict,
        "history": history
    }


def train_fixed_epochs(
    model: nn.Module,
    train_data: Dict,
    num_epochs: int,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    seed: int = 42
) -> nn.Module:
    """
    Seçilen sabit epoch sayısı kadar modeli dış eğitim kümesinde baştan eğitir
    (Epoch taşıma kuralı için kullanılır).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = GESTokenDataset(train_data)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_ges_batch
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            b_ids = batch["token_ids"]
            b_mask = batch["padding_mask"]
            b_targets = batch["targets_z"]

            preds = model(b_ids, padding_mask=b_mask)
            loss = criterion(preds, b_targets)
            loss.backward()
            optimizer.step()

    return model
