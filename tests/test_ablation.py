"""
BM618: Ablation ve tanı testleri (tests/test_ablation.py)

(a) ablation tahminleri: her varyant için 151 kayıt, karantina yok, her kayıt tam bir kez ve doğru dış katta.
(b) rastgele bölmede karantina kayıtları (REC_2024_R13, REC_2024_R15) eğitim/test dışında.
(c) ortak tesis oranı 0-1 arasında (ve kayıtlı CSV ile tutarlı).
(d) token silme PAD ile değiştirme değil, diziden ÇIKARMA ile yapılıyor.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_prep import load_audited_data, PAD_IDX
from src.diagnostics import make_random_splits, shared_facility_ratio, delete_token

sys.stdout.reconfigure(encoding="utf-8")
QUARANTINE = {"REC_2024_R13", "REC_2024_R15"}


def test_a_ablation_predictions():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "ablation_predictions.csv"))
    df_main, _, _ = load_audited_data()
    fold_of = dict(zip(df_main["record_id"], df_main["outer_test_fold"]))
    assert not df["record_id"].isin(QUARANTINE).any(), "ablation tahminlerinde karantina kaydı var!"
    assert set(df["variant"].unique()) == {"D2_kontrol", "D2_esit_agirlik", "D3_kontrol", "D3_konumsuz", "D3_yilsiz"}
    for v, d in df.groupby("variant"):
        assert len(d) == 151 and d["record_id"].nunique() == 151, f"{v}: 151 benzersiz kayıt değil"
        assert (d["outer_fold"] == d["record_id"].map(fold_of)).all(), f"{v}: kat uyuşmazlığı"
        assert d["y_pred"].notna().all() and np.isfinite(d["y_pred"]).all(), f"{v}: NaN/Inf tahmin"
        assert (d["seed"] == 42).all()
    print("Test (a) (Ablation: 5 varyant x 151 kayıt, karantina yok, kat uyumlu, NaN yok) BAŞARILI!")


def test_b_random_split_excludes_quarantine():
    df_main, _, _ = load_audited_data()
    df_main = df_main.reset_index(drop=True)
    assert not df_main["record_id"].isin(QUARANTINE).any()
    folds = make_random_splits(df_main)
    assert len(folds) == 151 and set(folds.tolist()) == {0, 1, 2, 3, 4}
    for k in range(5):
        te_ids = set(df_main.loc[folds == k, "record_id"])
        tr_ids = set(df_main.loc[folds != k, "record_id"])
        assert not (te_ids | tr_ids) & QUARANTINE, "rastgele bölmede karantina kaydı var!"
        assert not te_ids & tr_ids, "eğitim/test kesişiyor!"
    lk = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "leakage_predictions.csv"))
    assert len(lk) == 151 and not lk["record_id"].isin(QUARANTINE).any()
    assert (lk["etiket"] == "SIZINTI TANISI, geçerli skor değil").all()
    print("Test (b) (Rastgele bölme: karantina kayıtları eğitim/test dışında, etiketli) BAŞARILI!")


def test_c_shared_ratio_in_unit_interval():
    df_main, _, _ = load_audited_data()
    df_main = df_main.reset_index(drop=True)
    g = df_main["entity_group_id"].to_numpy()
    r_rand, per = shared_facility_ratio(g, make_random_splits(df_main))
    r_grp, _ = shared_facility_ratio(g, df_main["outer_test_fold"].to_numpy())
    assert 0.0 <= r_rand <= 1.0 and all(0.0 <= x <= 1.0 for x in per)
    assert r_grp == 0.0, "gruplu bölmede ortak tesis oranı 0 olmalı"
    tab = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "leakage_diagnostic.csv"))
    assert abs(tab["ortak_tesis_orani"].iloc[1] - r_rand) < 1e-3
    print(f"Test (c) (Ortak tesis oranı: rastgele {r_rand:.3f}, gruplu {r_grp:.3f}; ikisi de 0-1) BAŞARILI!")


def test_d_deletion_removes_not_pads():
    ids = torch.tensor([2, 3, 10, 4, 11, 5, 12, 13, PAD_IDX, PAD_IDX])
    mask = torch.tensor([False] * 8 + [True] * 2)
    new_ids, new_mask = delete_token(ids, mask, 6)          # 12 çıkarılır
    assert new_ids.tolist() == [2, 3, 10, 4, 11, 5, 13, PAD_IDX, PAD_IDX, PAD_IDX], "token çıkarılmadı/kaydırılmadı"
    assert int((~new_mask).sum()) == 7, "gerçek token sayısı 1 azalmalı"
    pad_replaced = ids.clone(); pad_replaced[6] = PAD_IDX    # yanlış yöntem (PAD ile değiştirme)
    assert new_ids.tolist() != pad_replaced.tolist(), "silme PAD ile değiştirmeye indirgenmiş!"
    assert 12 not in new_ids.tolist() and new_ids[6].item() == 13, "sonraki token sola kaymalı"
    print("Test (d) (Token silme: diziden ÇIKARMA, PAD ile değiştirme değil) BAŞARILI!")


def run_all():
    print("=" * 80)
    print("BM618: ABLATION VE TANI TESTLERİ BAŞLADI")
    print("=" * 80)
    test_a_ablation_predictions()
    test_b_random_split_excludes_quarantine()
    test_c_shared_ratio_in_unit_interval()
    test_d_deletion_removes_not_pads()
    print("=" * 80)
    print("[TÜM ABLATION/TANI TESTLERİ EKSİKSİZ BAŞARIYLA GEÇTİ]")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
