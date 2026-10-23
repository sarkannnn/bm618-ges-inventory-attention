"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Ana Dış Deney Yürütücüsü (src/run_main_experiment.py)

DENEY PROTOKOLÜ:
- Modeller: D1, D2, D3
- Dış Katlar: 0, 1, 2, 3, 4
- İç Arama: 4 aday (lr in {0.001, 0.0003} x dropout in {0.0, 0.2})
  * Her aday 3 iç katta eğitilir, iç tohum: 42
  * Sözlük ve max_len HER iç eğitim kümesinden bağımsız öğrenilir
  * Aday skoru = 3 iç katın doğrulama tesis dengeli MAE (kWp) ortalaması; en düşük seçilir
  * Epoch = seçilen adayın 3 iç kattaki en iyi epoch ortalaması (en yakın tam sayıya yuvarlanmış)
- Dış Eğitim:
  * Tüm dış eğitim kümesi üzerinde (sözlük ve max_len dış eğitimden)
  * Sabit epoch sayısı
  * Tohumlar: 17, 42, 2026
  * Dış testte YALNIZCA BİR KEZ tahmin
- Yeniden Başlatılabilirlik:
  * Biten adımları results/ dosyalarından kontrol eder ve kaldığı yerden devam eder.
- Çıktılar:
  * results/nn_inner_search.csv (model, outer_fold, lr, dropout, inner_fold, best_epoch, val_mae, duration_sec)
  * results/nn_selected.csv (model, outer_fold, lr, dropout, epochs)
  * results/nn_predictions.csv (record_id, outer_fold, model, seed, y_true, y_pred)
  * results/nn_metrics.csv (model, seed, fold, facility_balanced_mae, record_mae, rmse, r2, log_mae, signed_error, log_signed_error)
"""

import os
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_prep import (
    load_audited_data,
    prepare_tokenized_dataset,
    log1p_to_power
)
from src.models import (
    ModelD1,
    ModelD2,
    ModelD3
)
from src.train_nn import (
    train_neural_net,
    train_fixed_epochs
)
from src.metrics import (
    compute_all_metrics
)

# Aday hiperparametre ızgarası (Sabit 4 aday)
CANDIDATES = [
    (0.001, 0.0),
    (0.001, 0.2),
    (0.0003, 0.0),
    (0.0003, 0.2)
]

OUTER_SEEDS = [17, 42, 2026]
INNER_SEARCH_SEED = 42
MODELS = ["D1", "D2", "D3"]


def instantiate_model(model_name: str, vocab_size: int, max_len: int, dropout: float) -> torch.nn.Module:
    """Verilen model adına göre taze model nesnesi oluşturur."""
    if model_name == "D1":
        return ModelD1(vocab_size=vocab_size, d_model=32, dropout=dropout)
    elif model_name == "D2":
        return ModelD2(vocab_size=vocab_size, d_model=32, dropout=dropout)
    elif model_name == "D3":
        # D3 için konum embedding tablosu protokol kararı gereği sabit 64 x 32
        return ModelD3(vocab_size=vocab_size, max_len=64, d_model=32, num_heads=4, d_ff=64, dropout=dropout)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")


def run_main_experiment():
    total_start_time = time.time()
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    inner_search_file = os.path.join(results_dir, "nn_inner_search.csv")
    selected_file = os.path.join(results_dir, "nn_selected.csv")
    predictions_file = os.path.join(results_dir, "nn_predictions.csv")
    metrics_file = os.path.join(results_dir, "nn_metrics.csv")

    df_main, _, df_inner = load_audited_data()
    outer_folds = sorted(df_main["outer_test_fold"].unique())

    # -------------------------------------------------------------------------
    # 1. İÇ ARAMA (INNER SEARCH) VE HİPERPARAMETRE SEÇİMİ
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("1. İÇ DOĞRULAMA HİPERPARAMETRE ARAMASI BAŞLIYOR...")
    print("=" * 80)

    # Mevcut iç arama kayıtlarını yükle (Yeniden başlatılabilirlik)
    completed_inner_search = set()
    inner_search_records = []
    if os.path.exists(inner_search_file):
        try:
            df_exist = pd.read_csv(inner_search_file)
            for _, r in df_exist.iterrows():
                key = (r["model"], int(r["outer_fold"]), float(r["lr"]), float(r["dropout"]), int(r["inner_fold"]))
                completed_inner_search.add(key)
                inner_search_records.append(r.to_dict())
            print(f"Mevcut {len(completed_inner_search)} iç arama adımı okundu ve atlanacak.")
        except Exception:
            pass

    for outer_fold in outer_folds:
        outer_train = df_main[df_main["outer_test_fold"] != outer_fold].copy()
        inner_info = df_inner[df_inner["outer_fold"] == outer_fold]
        train_with_inner = outer_train.merge(inner_info[["record_id", "inner_fold"]], on="record_id", how="inner")

        for inner_fold in [0, 1, 2]:
            in_train_df = train_with_inner[train_with_inner["inner_fold"] != inner_fold].copy()
            in_val_df = train_with_inner[train_with_inner["inner_fold"] == inner_fold].copy()

            # Sözlük ve max_len HER iç eğitim kümesinden bağımsız öğrenilir!
            in_train_data = prepare_tokenized_dataset(in_train_df, fit_vocab=True)
            in_vocab = in_train_data["vocab"]
            in_max_len = in_train_data["max_len"]
            in_vocab_size = len(in_vocab)

            # Doğrulama kümesi aynı sözlükle hazırlanır
            in_val_data = prepare_tokenized_dataset(
                in_val_df,
                vocab=in_vocab,
                max_len=in_max_len,
                fit_vocab=False
            )

            for model_name in MODELS:
                for lr, dropout in CANDIDATES:
                    key = (model_name, int(outer_fold), float(lr), float(dropout), int(inner_fold))
                    if key in completed_inner_search:
                        continue

                    t0 = time.time()
                    # Adım 8b: Tohum model nesnesi oluşturulmadan ÖNCE kurulur (ilklendirme ve karıştırma tohuma bağlı)
                    torch.manual_seed(INNER_SEARCH_SEED)
                    np.random.seed(INNER_SEARCH_SEED)
                    model = instantiate_model(model_name, in_vocab_size, in_max_len, dropout)

                    train_res = train_neural_net(
                        model,
                        in_train_data,
                        val_data=in_val_data,
                        lr=lr,
                        weight_decay=1e-4,
                        batch_size=16,
                        max_epochs=150,
                        patience=15,
                        seed=INNER_SEARCH_SEED,
                        verbose=False
                    )

                    duration = time.time() - t0
                    rec = {
                        "model": model_name,
                        "outer_fold": int(outer_fold),
                        "lr": float(lr),
                        "dropout": float(dropout),
                        "inner_fold": int(inner_fold),
                        "best_epoch": int(train_res["best_epoch"]),
                        "val_mae": float(train_res["best_val_mae"]),
                        "duration_sec": round(duration, 2)
                    }
                    inner_search_records.append(rec)
                    completed_inner_search.add(key)

                    # Anlık CSV'ye yaz (güvenli kaydetme)
                    pd.DataFrame(inner_search_records).to_csv(inner_search_file, index=False, encoding="utf-8")
                    print(f"  [İç Arama] {model_name} | Kat {outer_fold} | İç Kat {inner_fold} | lr={lr}, drop={dropout} -> En İyi Epoch: {rec['best_epoch']}, Val MAE: {rec['val_mae']:.2f} kWp ({duration:.2f}s)")

    df_inner_all = pd.DataFrame(inner_search_records)
    df_inner_all.to_csv(inner_search_file, index=False, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 2. EN İYİ YAPILANDIRMA VE EPOCH SEÇİMİ
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. EN İYİ HİPERPARAMETRE VE EPOCH SEÇİMİ HESAPLANIYOR...")
    print("=" * 80)

    selected_records = []
    for model_name in MODELS:
        for outer_fold in outer_folds:
            sub = df_inner_all[(df_inner_all["model"] == model_name) & (df_inner_all["outer_fold"] == outer_fold)]

            candidate_scores = []
            for lr, dropout in CANDIDATES:
                c_sub = sub[(sub["lr"] == lr) & (sub["dropout"] == dropout)]
                assert len(c_sub) == 3, f"3 iç kat tamamlanmamış: {model_name}, kat {outer_fold}, lr {lr}, drop {dropout}"
                mean_val_mae = float(c_sub["val_mae"].mean())
                mean_epochs = float(c_sub["best_epoch"].mean())
                candidate_scores.append({
                    "lr": lr,
                    "dropout": dropout,
                    "mean_val_mae": mean_val_mae,
                    "mean_epochs": mean_epochs
                })

            # En düşük MAE'yi seç (eşitlik durumunda CANDIDATES sırasındaki ilk)
            best_cand = min(candidate_scores, key=lambda x: x["mean_val_mae"])
            best_epoch_int = max(1, int(np.round(best_cand["mean_epochs"])))

            selected_records.append({
                "model": model_name,
                "outer_fold": int(outer_fold),
                "lr": float(best_cand["lr"]),
                "dropout": float(best_cand["dropout"]),
                "epochs": int(best_epoch_int)
            })
            print(f"  [Seçildi] {model_name:2s} | Dış Kat {outer_fold} -> lr={best_cand['lr']}, drop={best_cand['dropout']}, Sabit Epoch={best_epoch_int} (İç CV MAE={best_cand['mean_val_mae']:.2f} kWp)")

    df_selected = pd.DataFrame(selected_records)
    df_selected.to_csv(selected_file, index=False, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 3. DIŞ EĞİTİM VE TEST TAHMİNLERİ (3 TOHUM: 17, 42, 2026)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. DIŞ EĞİTİM VE DIŞ TEST TAHMİNLERİ BAŞLIYOR (Tohumlar: 17, 42, 2026)...")
    print("=" * 80)

    completed_predictions = set()
    predictions_records = []
    if os.path.exists(predictions_file):
        try:
            df_pred_exist = pd.read_csv(predictions_file)
            for _, r in df_pred_exist.iterrows():
                key = (r["model"], int(r["outer_fold"]), int(r["seed"]))
                completed_predictions.add(key)
                predictions_records.append(r.to_dict())
            print(f"Mevcut {len(completed_predictions)} dış eğitim/tahmin seti okundu ve atlanacak.")
        except Exception:
            pass

    for outer_fold in outer_folds:
        outer_train_df = df_main[df_main["outer_test_fold"] != outer_fold].copy()
        outer_test_df = df_main[df_main["outer_test_fold"] == outer_fold].copy()

        # Sözlük ve max_len TÜM DIŞ EĞİTİMDEN öğrenilir
        out_train_data = prepare_tokenized_dataset(outer_train_df, fit_vocab=True)
        out_vocab = out_train_data["vocab"]
        out_max_len = out_train_data["max_len"]
        out_vocab_size = len(out_vocab)

        out_test_data = prepare_tokenized_dataset(
            outer_test_df,
            vocab=out_vocab,
            max_len=out_max_len,
            fit_vocab=False
        )

        for model_name in MODELS:
            sel_row = df_selected[(df_selected["model"] == model_name) & (df_selected["outer_fold"] == outer_fold)].iloc[0]
            lr = float(sel_row["lr"])
            dropout = float(sel_row["dropout"])
            epochs = int(sel_row["epochs"])

            for seed in OUTER_SEEDS:
                key = (model_name, int(outer_fold), int(seed))
                if key in completed_predictions:
                    continue

                t0 = time.time()
                # Adım 8b: Tohum model nesnesi oluşturulmadan ÖNCE kurulur (ilklendirme ve karıştırma tohuma bağlı)
                torch.manual_seed(seed)
                np.random.seed(seed)
                # Taze model ilklendirme
                model = instantiate_model(model_name, out_vocab_size, out_max_len, dropout)

                # Dış eğitim kümesinde tam epochs sayısı kadar eğit
                train_fixed_epochs(
                    model,
                    out_train_data,
                    num_epochs=epochs,
                    lr=lr,
                    weight_decay=1e-4,
                    batch_size=16,
                    seed=seed
                )

                # Adım 8b (3): Dış eğitilen model ağırlıklarını kaydet (results/weights/model_fold_seed.pt)
                weights_dir = os.path.join(results_dir, "weights")
                os.makedirs(weights_dir, exist_ok=True)
                weight_path = os.path.join(weights_dir, f"{model_name}_fold{outer_fold}_seed{seed}.pt")
                torch.save(model.state_dict(), weight_path)

                # Dış testte YALNIZ BİR KEZ tahmin üret
                model.eval()
                with torch.no_grad():
                    z_hat = model(out_test_data["token_ids"], padding_mask=out_test_data["key_padding_mask"]).cpu().numpy()
                    y_hat = log1p_to_power(z_hat)

                duration = time.time() - t0
                for (_, row), yp in zip(outer_test_df.iterrows(), y_hat):
                    predictions_records.append({
                        "record_id": row["record_id"],
                        "outer_fold": int(outer_fold),
                        "model": model_name,
                        "seed": int(seed),
                        "y_true": float(row["power_kwp"]),
                        "y_pred": float(yp)
                    })

                completed_predictions.add(key)
                pd.DataFrame(predictions_records).to_csv(predictions_file, index=False, encoding="utf-8")
                print(f"  [Dış Tahmin] {model_name:2s} | Kat {outer_fold} | Tohum {seed} | Sabit Epoch={epochs} tamamlandı ({duration:.2f}s)")

    df_preds_all = pd.DataFrame(predictions_records)
    # Sütun sırasını garantiye al
    pred_cols = ["record_id", "outer_fold", "model", "seed", "y_true", "y_pred"]
    df_preds_all[pred_cols].to_csv(predictions_file, index=False, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 4. METRİKLERİN HESAPLANMASI (results/nn_metrics.csv)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. METRİKLER HESAPLANIYOR VE KAYDEDİLİYOR...")
    print("=" * 80)

    # entity_group_id bilgisini merge et
    entity_map = dict(zip(df_main["record_id"], df_main["entity_group_id"]))
    df_preds_all["entity_group_id"] = df_preds_all["record_id"].map(entity_map)

    metrics_list = []
    for model_name in MODELS:
        m_sub = df_preds_all[df_preds_all["model"] == model_name]
        for seed in OUTER_SEEDS:
            s_sub = m_sub[m_sub["seed"] == seed]

            # Kat bazında
            for outer_fold in outer_folds:
                f_sub = s_sub[s_sub["outer_fold"] == outer_fold]
                met = compute_all_metrics(f_sub["y_true"], f_sub["y_pred"], f_sub["entity_group_id"])
                met["model"] = model_name
                met["seed"] = seed
                met["fold"] = str(outer_fold)
                metrics_list.append(met)

            # Birleştirilmiş (pooled)
            pooled_met = compute_all_metrics(s_sub["y_true"], s_sub["y_pred"], s_sub["entity_group_id"])
            pooled_met["model"] = model_name
            pooled_met["seed"] = seed
            pooled_met["fold"] = "pooled"
            metrics_list.append(pooled_met)

    cols_order = [
        "model", "seed", "fold", "facility_balanced_mae", "record_mae",
        "rmse", "r2", "log_mae", "signed_error", "log_signed_error"
    ]
    df_metrics = pd.DataFrame(metrics_list)[cols_order]
    df_metrics.to_csv(metrics_file, index=False, encoding="utf-8")

    total_duration = time.time() - total_start_time
    print(f"\nAna dış deney tamamlandı! Toplam süre: {total_duration:.2f} saniye ({total_duration/60.0:.2f} dakika).")
    return df_inner_all, df_selected, df_preds_all, df_metrics, total_duration


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run_main_experiment()
