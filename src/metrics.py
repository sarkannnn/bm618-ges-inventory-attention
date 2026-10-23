"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Değerlendirme Metrikleri ve Eşleştirilmiş Bootstrap Modülü (src/metrics.py)

KURALLAR VE METRİKLER:
- Birincil Metrik: Tesis Dengeli MAE (Facility-Balanced MAE) (kWp):
  Her tesis grubunun (entity_group_id) kayıt hatalarının ortalaması alınır,
  ardından tüm tesis grupları eşit ağırlıkla ortalanır.
- İkincil Metrikler:
  * Kayıt MAE (Record MAE)
  * Kök Ortalama Kare Hata (RMSE)
  * Belirlilik Katsayısı (R²)
  * Log-ölçek Hata (log1p mutlak farklarının MAE'si)
- Eşleştirilmiş Bootstrap Fonksiyonu:
  Tesis gruplarını 1000 kez eşleştirilmiş bootstrap ile örnekleyip iki modelin
  tesis dengeli MAE farkı (Model A - Model B) için %95 güven aralığı hesaplar (sabit tohum).
"""

import os
import sys
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def compute_facility_balanced_mae(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    entity_groups: Union[np.ndarray, List[str]]
) -> float:
    """
    Tesis dengeli MAE (kWp):
    Her tesis grubunun kayıt hatalarının aritmetik ortalaması alınır,
    ardından tüm tesis grupları eşit ağırlıkla ortalanır.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    groups = np.asarray(entity_groups)

    abs_errors = np.abs(y_t - y_p)
    df = pd.DataFrame({"group": groups, "abs_error": abs_errors})
    group_maes = df.groupby("group")["abs_error"].mean()
    return float(group_maes.mean())


def compute_record_mae(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> float:
    """Kayıt bazında standart MAE (kWp)"""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def compute_rmse(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> float:
    """Kök Ortalama Kare Hata (RMSE) (kWp)"""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def compute_r2(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> float:
    """Belirlilik Katsayısı (R²)"""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot == 0.0:
        return 0.0 if ss_res > 0.0 else 1.0
    return float(1.0 - (ss_res / ss_tot))


def compute_log_mae(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> float:
    """Log-ölçek hata (log1p değerlerinin mutlak farklarının MAE'si)"""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    y_p_clipped = np.maximum(0.0, y_p)
    log_t = np.log1p(y_t)
    log_p = np.log1p(y_p_clipped)
    return float(np.mean(np.abs(log_t - log_p)))


def compute_signed_error(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> float:
    """Ortalama işaretli hata (y_hat - y, kWp)"""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(y_p - y_t))


def compute_log_signed_error(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]]
) -> float:
    """Log-ölçek ortalama işaretli hata (log1p(y_hat) - log1p(y))"""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    y_p_clipped = np.maximum(0.0, y_p)
    return float(np.mean(np.log1p(y_p_clipped) - np.log1p(y_t)))


def compute_all_metrics(
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    entity_groups: Union[np.ndarray, List[str]]
) -> Dict[str, float]:
    """Tüm metrikleri tek bir sözlük olarak hesaplar ve döndürür."""
    return {
        "facility_balanced_mae": compute_facility_balanced_mae(y_true, y_pred, entity_groups),
        "record_mae": compute_record_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "log_mae": compute_log_mae(y_true, y_pred),
        "signed_error": compute_signed_error(y_true, y_pred),
        "log_signed_error": compute_log_signed_error(y_true, y_pred)
    }


def paired_bootstrap_facility_mae_ci(
    y_true: Union[np.ndarray, List[float]],
    y_pred_a: Union[np.ndarray, List[float]],
    y_pred_b: Union[np.ndarray, List[float]],
    entity_groups: Union[np.ndarray, List[str]],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Tesis gruplarını n_boot (varsayılan 1000) kez eşleştirilmiş bootstrap ile
    örnekleyip iki modelin (Model A - Model B) tesis dengeli MAE farkı için
    %(1-alpha)*100 (varsayılan %95) güven aralığı hesaplar.

    Döndürür: (ci_low, ci_high, mean_diff)
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_a = np.asarray(y_pred_a, dtype=np.float64)
    y_b = np.asarray(y_pred_b, dtype=np.float64)
    groups = np.asarray(entity_groups)

    abs_err_a = np.abs(y_t - y_a)
    abs_err_b = np.abs(y_t - y_b)

    df = pd.DataFrame({
        "group": groups,
        "err_a": abs_err_a,
        "err_b": abs_err_b
    })

    # Her grup için ortalama hatalar
    grouped = df.groupby("group").mean()
    group_maes_a = grouped["err_a"].to_numpy()
    group_maes_b = grouped["err_b"].to_numpy()

    # Grup bazında hata farkı: d_g = MAE_g(A) - MAE_g(B)
    group_diffs = group_maes_a - group_maes_b
    mean_diff = float(np.mean(group_diffs))

    G = len(group_diffs)
    rng = np.random.default_rng(seed)

    # 1000 yineleme ile grupların yerine koyarak örneklenmesi
    boot_indices = rng.integers(0, G, size=(n_boot, G))
    boot_means = np.mean(group_diffs[boot_indices], axis=1)

    ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return ci_low, ci_high, mean_diff
