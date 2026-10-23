"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Model Karşılaştırma, Parametre Analizi, Eşleştirilmiş Bootstrap ve H1 Karar Modülü (src/compare.py)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_prep import load_audited_data, prepare_tokenized_dataset, extract_features, power_to_log1p
from src.baselines import build_b1_pipeline
from src.models import (
    ModelD1,
    ModelD2,
    ModelD3,
    count_parameters_detailed
)
from src.metrics import (
    compute_facility_balanced_mae,
    compute_record_mae,
    compute_rmse,
    compute_r2,
    compute_log_mae,
    compute_signed_error,
    compute_log_signed_error
)

MODELS_ORDER = ["B0", "B1", "D1", "D2", "D3"]


def load_all_predictions() -> pd.DataFrame:
    """B0, B1 ve D1, D2, D3 tahminlerini birleştirir."""
    results_dir = os.path.join(PROJECT_ROOT, "results")
    b_path = os.path.join(results_dir, "baselines_predictions.csv")
    nn_path = os.path.join(results_dir, "nn_predictions.csv")

    assert os.path.exists(b_path), f"Baselines tahminleri bulunamadı: {b_path}"
    assert os.path.exists(nn_path), f"NN tahminleri bulunamadı: {nn_path}"

    df_b = pd.read_csv(b_path)
    # B0 ve B1 için tohum sütunu yoksa tek tohum (0) ekle
    if "seed" not in df_b.columns:
        df_b["seed"] = 0

    df_nn = pd.read_csv(nn_path)

    # entity_group_id bilgisini ana envanterden ekle
    df_main, _, _ = load_audited_data()
    entity_map = dict(zip(df_main["record_id"], df_main["entity_group_id"]))

    df_all = pd.concat([df_b, df_nn], ignore_index=True)
    df_all["entity_group_id"] = df_all["record_id"].map(entity_map)

    return df_all


def generate_comparison_table(df_preds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pooled ve kat bazında karşılaştırma tablolarını üretir.
    D1-D3 için 3 tohum üzerinden ortalama ± standart sapma hesaplanır.
    """
    folds = sorted(df_preds["outer_fold"].unique())
    pooled_rows = []
    fold_rows = []

    for model in MODELS_ORDER:
        m_df = df_preds[df_preds["model"] == model]
        seeds = sorted(m_df["seed"].unique())

        # 1. POOLED METRİKLER (Her tohum için pooled, sonra tohumlar ortalaması ve std)
        seed_pooled_metrics = []
        for s in seeds:
            s_df = m_df[m_df["seed"] == s]
            y_t = s_df["y_true"].to_numpy()
            y_p = s_df["y_pred"].to_numpy()
            groups = s_df["entity_group_id"].to_numpy()

            seed_pooled_metrics.append({
                "facility_mae": compute_facility_balanced_mae(y_t, y_p, groups),
                "record_mae": compute_record_mae(y_t, y_p),
                "rmse": compute_rmse(y_t, y_p),
                "r2": compute_r2(y_t, y_p),
                "log_mae": compute_log_mae(y_t, y_p),
                "signed_error": compute_signed_error(y_t, y_p),
                "log_signed_error": compute_log_signed_error(y_t, y_p)
            })

        df_spm = pd.DataFrame(seed_pooled_metrics)
        # B0 ve B1 deterministiktir (std = 0)
        p_row = {"Model": model}
        for col in ["facility_mae", "record_mae", "rmse", "r2", "log_mae", "signed_error", "log_signed_error"]:
            m_val = df_spm[col].mean()
            s_val = df_spm[col].std() if len(df_spm) > 1 else 0.0
            if len(df_spm) > 1 and s_val > 0.001:
                p_row[col] = f"{m_val:.2f} ± {s_val:.2f}"
            else:
                p_row[col] = f"{m_val:.2f}"
        pooled_rows.append(p_row)

        # 2. KAT BAZINDA TESİS DENGELİ MAE
        f_row = {"Model": model}
        for f in folds:
            f_vals = []
            for s in seeds:
                f_sub = m_df[(m_df["outer_fold"] == f) & (m_df["seed"] == s)]
                f_mae = compute_facility_balanced_mae(
                    f_sub["y_true"].to_numpy(),
                    f_sub["y_pred"].to_numpy(),
                    f_sub["entity_group_id"].to_numpy()
                )
                f_vals.append(f_mae)
            f_row[f"Kat {f}"] = f"{np.mean(f_vals):.2f}"
        fold_rows.append(f_row)

    df_comp_pooled = pd.DataFrame(pooled_rows)
    df_comp_folds = pd.DataFrame(fold_rows)
    return df_comp_pooled, df_comp_folds


def generate_fold_parameter_table() -> pd.DataFrame:
    """
    Her dış eğitim kümesi için sözlük boyutu, max_len ve D1/D2/D3 toplam ve etkin parametre sayıları.
    Sözlük ve max_len dış eğitimden öğrenildiği için parametre sayısı kata göre değişir
    (embedding = sözlük x 32). Etkin parametre: D3'te konum tablosundan yalnız fiilen kullanılan
    max_len konumu (max_len x 32) sayılır; D1/D2'de etkin = toplam.
    B1 parametresi = uygulanmış Ridge'in katsayı sayısı + 1 (kesim).
    """
    df_main, _, _ = load_audited_data()
    rows = []
    for f in sorted(df_main["outer_test_fold"].unique()):
        tr = df_main[df_main["outer_test_fold"] != f]
        data = prepare_tokenized_dataset(tr, fit_vocab=True)
        v = len(data["vocab"])
        ml = data["max_len"]
        c1 = count_parameters_detailed(ModelD1(vocab_size=v, d_model=32), effective_len=ml)
        c2 = count_parameters_detailed(ModelD2(vocab_size=v, d_model=32), effective_len=ml)
        c3 = count_parameters_detailed(ModelD3(vocab_size=v, max_len=64, d_model=32), effective_len=ml)
        pipe = build_b1_pipeline(alpha=1.0)
        pipe.fit(extract_features(tr), power_to_log1p(tr["power_kwp"]))
        b1 = int(pipe.named_steps["regressor"].coef_.size + 1)
        rows.append({
            "Dış Kat": int(f), "Eğitim Tesis": int(tr["entity_group_id"].nunique()),
            "Sözlük": v, "max_len": ml, "B1": b1,
            "D1 Toplam": c1["total"], "D1 Etkin": c1["effective_total"],
            "D2 Toplam": c2["total"], "D2 Etkin": c2["effective_total"],
            "D3 Toplam": c3["total"], "D3 Etkin": c3["effective_total"]
        })
    return pd.DataFrame(rows)


def generate_parameter_analysis(df_fold_params: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Parametre sayısı (toplam ve etkin) ve parametre/tesis oranları tablosunu üretir.
    Dış eğitim tesis sayısı ~61, toplam tesis sayısı 76.
    """
    if df_fold_params is None:
        df_fold_params = generate_fold_parameter_table()
    n_total = 76
    n_train = float(df_fold_params["Eğitim Tesis"].mean())
    m = df_fold_params.mean(numeric_only=True)

    rows = [
        {"Model": "B0 (Medyan)", "Toplam Parametre": 1.0, "Etkin Parametre": 1.0},
        {"Model": "B1 (Ridge TF-IDF)", "Toplam Parametre": m["B1"], "Etkin Parametre": m["B1"]},
        {"Model": "D1 (Maskeli Ortalama)", "Toplam Parametre": m["D1 Toplam"], "Etkin Parametre": m["D1 Etkin"]},
        {"Model": "D2 (Attention Havuzlama)", "Toplam Parametre": m["D2 Toplam"], "Etkin Parametre": m["D2 Etkin"]},
        {"Model": "D3 (Pre-LN Transformer)", "Toplam Parametre": m["D3 Toplam"], "Etkin Parametre": m["D3 Etkin"]},
    ]
    df_p = pd.DataFrame(rows)
    df_p["Toplam Parametre"] = df_p["Toplam Parametre"].round(0).astype(int)
    df_p["Etkin Parametre"] = df_p["Etkin Parametre"].round(0).astype(int)
    df_p[f"Etkin parametre / tesis (eğitim ort. {n_train:.1f} tesis)"] = (df_p["Etkin Parametre"] / n_train).round(1)
    df_p["Etkin parametre / tesis (toplam 76)"] = (df_p["Etkin Parametre"] / n_total).round(1)
    return df_p


def run_paired_bootstrap_tests(df_preds: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    1000 tekrarlı eşleştirilmiş bootstrap analizi.
    Kayıt hatası = kaydın 3 tohumdaki mutlak hatalarının ortalaması.
    Tesis grupları yerine koyarak örneklenir.
    Negatif fark = İlk model daha iyi (daha düşük hata).
    """
    df_main, _, _ = load_audited_data()
    unique_groups = sorted(df_main["entity_group_id"].unique())
    G = len(unique_groups)
    assert G == 76, f"Toplam tesis grubu 76 olmalıdır, bulunan: {G}"

    # Her model için her kaydın tohum ortalamalı mutlak hatası
    record_errors = {}
    for model in MODELS_ORDER:
        m_df = df_preds[df_preds["model"] == model]
        # Her kayıt için tohumlar üzerinden ortalama mutlak hata
        m_df = m_df.copy()
        m_df["abs_err"] = np.abs(m_df["y_true"] - m_df["y_pred"])
        rec_mean_err = m_df.groupby("record_id")["abs_err"].mean()
        record_errors[model] = rec_mean_err

    # Grup düzeyinde hata: her tesis grubundaki kayıtların ortalama hatası
    df_eval = df_main[["record_id", "entity_group_id"]].copy()
    group_errors = {}
    for model in MODELS_ORDER:
        df_eval[f"err_{model}"] = df_eval["record_id"].map(record_errors[model])
        g_err = df_eval.groupby("entity_group_id")[f"err_{model}"].mean()
        group_errors[model] = g_err.reindex(unique_groups).to_numpy()

    # Karşılaştırılacak çiftler: (Model A, Model B) -> Fark: Model A - Model B
    comparison_pairs = [
        ("B1", "B0"),
        ("D1", "B1"),
        ("D2", "B1"),
        ("D3", "B1"),
        ("D2", "D1"),
        ("D3", "D1"),
        ("D3", "D2")
    ]

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, G, size=(n_boot, G))

    boot_results = []
    boot_dict = {}

    for mod_a, mod_b in comparison_pairs:
        # Grup düzeyinde fark vektörü
        diff_vec = group_errors[mod_a] - group_errors[mod_b]
        mean_diff = float(np.mean(diff_vec))

        # Bootstrap örneklemeleri
        boot_diffs = np.mean(diff_vec[boot_idx], axis=1)
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))

        # Sıfırı içeriyor mu?
        includes_zero = (ci_low <= 0.0 <= ci_high)

        # Karar / Üstünlük
        if ci_high < 0:
            conclusion = f"{mod_a} anlamlı üstün (negatif fark)"
        elif ci_low > 0:
            conclusion = f"{mod_b} anlamlı üstün (pozitif fark)"
        else:
            conclusion = "Fark ayırt edilemedi (aralık 0'ı içeriyor)"

        row = {
            "Karşılaştırma": f"{mod_a} − {mod_b}",
            "Tesis Dengeli MAE Farkı (kWp)": f"{mean_diff:+.2f}",
            "%95 Güven Aralığı": f"[{ci_low:+.2f}, {ci_high:+.2f}]",
            "Sıfırı İçeriyor mu?": "Evet" if includes_zero else "Hayır",
            "Sonuç": conclusion
        }
        boot_results.append(row)
        boot_dict[f"{mod_a}-{mod_b}"] = {
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "includes_zero": includes_zero
        }

    df_boot = pd.DataFrame(boot_results)
    return df_boot, boot_dict


def evaluate_h1_decision(df_comp_pooled: pd.DataFrame, boot_dict: Dict) -> Tuple[str, str]:
    """
    Protokol H1 koşulu: D3'ün tesis dengeli MAE'si hem B1'den hem D1'den düşük değilse,
    ya da D3−B1 ve D3−D1 farklarının eşleştirilmiş bootstrap %95 aralığı sıfırı içeriyorsa H1 reddedilir.
    D2 için aynı koşul ayrıca yazılır. 'D1'i geçemedi' yerine, MAE sayısal olarak düşük ama
    aralık sıfırı içeriyorsa 'fark ayırt edilemedi' denir.
    """
    mae = {}
    for _, r in df_comp_pooled.iterrows():
        mae[r["Model"]] = float(r["facility_mae"].split()[0])

    def decide(name: str) -> str:
        m, b1, d1 = mae[name], mae["B1"], mae["D1"]
        vb, vd = boot_dict[f"{name}-B1"], boot_dict[f"{name}-D1"]
        rejected = False
        parts = []
        # Belirleyici koşul: MAE(model) >= MAE(B1)
        if m >= b1:
            rejected = True
            parts.append(f"BELİRLEYİCİ KOŞUL: MAE({name}) = {m:.2f} >= MAE(B1) = {b1:.2f} ({name}−B1 = {vb['mean_diff']:+.2f} kWp; aralık [{vb['ci_low']:+.2f}, {vb['ci_high']:+.2f}])")
        if vb["includes_zero"]:
            rejected = True
            parts.append(f"{name}−B1 aralığı sıfırı içeriyor (ikinci koşul da sağlanıyor)")
        # D1 karşılaştırması
        if m >= d1:
            rejected = True
            parts.append(f"MAE({name}) = {m:.2f} >= MAE(D1) = {d1:.2f}")
        elif vd["includes_zero"]:
            parts.append(f"MAE({name}) = {m:.2f} < MAE(D1) = {d1:.2f} sayısal olarak düşük, ancak {name}−D1 = {vd['mean_diff']:+.2f} kWp, aralık [{vd['ci_low']:+.2f}, {vd['ci_high']:+.2f}] sıfırı içerdiğinden fark ayırt edilemedi")
        if vd["includes_zero"] and m >= d1:
            parts.append(f"{name}−D1 aralığı sıfırı içeriyor")
        head = "REDDEDİLDİ" if rejected else "REDDEDİLMEDİ"
        return f"{head}. " + "; ".join(parts) + "."

    return decide("D3"), decide("D2")


def run_full_comparison():
    print("=" * 80)
    print("BM618: B0, B1, D1, D2, D3 MODEL KARŞILAŞTIRMA VE HİPOTEZ RAPORU")
    print("=" * 80)

    df_preds = load_all_predictions()
    df_comp_pooled, df_comp_folds = generate_comparison_table(df_preds)
    df_fold_params = generate_fold_parameter_table()
    df_params = generate_parameter_analysis(df_fold_params)
    df_boot, boot_dict = run_paired_bootstrap_tests(df_preds)
    h1_dec, d2_dec = evaluate_h1_decision(df_comp_pooled, boot_dict)

    print("\n1. BİRLEŞTİRİLMİŞ (POOLED) METRİKLER (Tohum Ortalaması ± Std):")
    print(df_comp_pooled.to_string(index=False))

    print("\n2. KAT BAZINDA TESİS DENGELİ MAE (kWp):")
    print(df_comp_folds.to_string(index=False))

    print("\n3a. KAT BAZINDA SÖZLÜK, max_len VE PARAMETRE SAYISI (sözlük/max_len dış eğitimden öğrenilir):")
    print(df_fold_params.to_string(index=False))
    print("\n3b. PARAMETRE SAYISI VE ETKİN PARAMETRE / TESİS (5 kat ortalaması):")
    print(df_params.to_string(index=False))

    print("\n4. EŞLEŞTİRİLMİŞ BOOTSTRAP FARKLARI (1000 Tekrar, Tohum 42):")
    print("   [Not: Negatif fark = İlk model daha iyi / daha düşük hata]")
    print(df_boot.to_string(index=False))
    print("   * ÖNEMLİ METODOLOJİK NOT: Bu güven aralığı yalnızca sabit bölmeler ve eğitim")
    print("     prosedürü koşulundaki belirsizliği yansıtır; 76 bağımsız tesisin oluşturduğu")
    print("     küçük örneklem boyutu nedeniyle sonuçlar genellenemez.")

    print("\n5. PROTOKOL HİPOTEZ KARARLARI:")
    print(f"   H1 Kararı (D3 vs B1, D1): {h1_dec}")
    print(f"   D2 Kararı (D2 vs B1, D1): {d2_dec}")
    print("=" * 80)

    # Karşılaştırma özetini CSV olarak da kaydet
    out_dir = os.path.join(PROJECT_ROOT, "results")
    df_comp_pooled.to_csv(os.path.join(out_dir, "comparison_pooled.csv"), index=False, encoding="utf-8")
    df_boot.to_csv(os.path.join(out_dir, "comparison_bootstrap.csv"), index=False, encoding="utf-8")

    return df_comp_pooled, df_comp_folds, df_params, df_boot, h1_dec, d2_dec


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run_full_comparison()
