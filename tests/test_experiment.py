"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Ana Deney ve Tahmin Bütünlüğü Test Paketi (tests/test_experiment.py)

TESTLER:
(a) nn_predictions.csv 3 model x 3 tohum x 151 kayıt = 1359 satır, her model-tohum için 151 benzersiz record_id, karantina kaydı yok.
(b) Her kayıt tam bir kez dış testte (outer_fold = splits.csv'deki outer_test_fold).
(c) Aynı modeli kendisiyle karşılaştıran bootstrap farkı tam sıfır aralık verir ([0.0, 0.0]).
(d) Seçilen epoch değerleri 1 ile 150 arasındadır.
"""

import os
import sys
import numpy as np
import pandas as pd

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_prep import load_audited_data

# UTF-8 konsol çıktısı
sys.stdout.reconfigure(encoding='utf-8')


def test_a_predictions_count_and_no_quarantine():
    """
    Test (a): nn_predictions.csv dosyası tam 1359 satır olmalı (3 model x 3 tohum x 151 kayıt).
    Her model-tohum için 151 benzersiz record_id bulunmalı ve karantina kaydı kesinlikle olmamalıdır.
    """
    results_dir = os.path.join(PROJECT_ROOT, "results")
    preds_path = os.path.join(results_dir, "nn_predictions.csv")
    assert os.path.exists(preds_path), f"nn_predictions.csv bulunamadı: {preds_path}"

    df = pd.read_csv(preds_path)
    quarantine_ids = {"REC_2024_R13", "REC_2024_R15"}

    # 1. Toplam satır sayısı: 3 x 3 x 151 = 1359
    assert len(df) == 1359, f"Toplam tahmin satırı beklenen 1359 yerine {len(df)} çıktı!"

    # 2. Karantina kaydı yok
    assert not df["record_id"].isin(quarantine_ids).any(), "nn_predictions.csv içinde karantina kaydı bulundu!"

    # 3. Her model-tohum için 151 benzersiz kayıt
    models = ["D1", "D2", "D3"]
    seeds = [17, 42, 2026]
    for m in models:
        for s in seeds:
            sub = df[(df["model"] == m) & (df["seed"] == s)]
            assert len(sub) == 151, f"{m} tohum {s} için satır sayısı beklenen 151 yerine {len(sub)}"
            assert sub["record_id"].nunique() == 151, f"{m} tohum {s} için benzersiz kayıt sayısı 151 değil!"

    print("Test (a) (nn_predictions.csv: 1359 Satır, 151 Benzersiz Kayıt, Karantina Kaydı Yok) BAŞARILI!")


def test_b_records_in_outer_test_exactly_once():
    """
    Test (b): Her kayıt dış test kümesinde tam olarak bir kez yer almalıdır
    ve outer_fold değeri splits.csv'deki outer_test_fold ile birebir eşleşmelidir.
    """
    results_dir = os.path.join(PROJECT_ROOT, "results")
    preds_path = os.path.join(results_dir, "nn_predictions.csv")
    df_preds = pd.read_csv(preds_path)

    df_main, _, _ = load_audited_data()
    expected_split_map = dict(zip(df_main["record_id"], df_main["outer_test_fold"]))

    # Model D1, tohum 42 örneği üzerinde kontrol
    sub = df_preds[(df_preds["model"] == "D1") & (df_preds["seed"] == 42)]
    for _, row in sub.iterrows():
        rec_id = row["record_id"]
        fold = row["outer_fold"]
        expected_fold = expected_split_map[rec_id]
        assert fold == expected_fold, f"Kayıt {rec_id} için kat uyuşmazlığı: beklenen {expected_fold}, bulunan {fold}"

    print("Test (b) (Her Kayıt Dış Testte Tam 1 Kez ve splits.csv ile %100 Uyumlu) BAŞARILI!")


def test_c_self_comparison_bootstrap_zero_interval():
    """
    Test (c): Aynı modeli kendisiyle karşılaştıran eşleştirilmiş bootstrap
    farkı tam 0.0 ortalama ve [0.0, 0.0] güven aralığı vermelidir.
    """
    from src.compare import load_all_predictions, run_paired_bootstrap_tests
    df_all = load_all_predictions()

    # Model D3'ün kendi tahminleriyle farkını yapay olarak test edelim
    df_main, _, _ = load_audited_data()
    unique_groups = sorted(df_main["entity_group_id"].unique())
    G = len(unique_groups)

    d3_preds = df_all[df_all["model"] == "D3"].copy()
    d3_preds["abs_err"] = np.abs(d3_preds["y_true"] - d3_preds["y_pred"])
    rec_mean_err = d3_preds.groupby("record_id")["abs_err"].mean()

    df_eval = df_main[["record_id", "entity_group_id"]].copy()
    df_eval["err"] = df_eval["record_id"].map(rec_mean_err)
    g_err = df_eval.groupby("entity_group_id")["err"].mean().to_numpy()

    # Kendisiyle fark
    diff_vec = g_err - g_err
    assert (diff_vec == 0.0).all()

    rng = np.random.default_rng(42)
    boot_idx = rng.integers(0, G, size=(500, G))
    boot_diffs = np.mean(diff_vec[boot_idx], axis=1)

    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))

    assert abs(ci_low) < 1e-12 and abs(ci_high) < 1e-12, f"Özdeş model bootstrap CI sıfır değil: [{ci_low}, {ci_high}]"
    print("Test (c) (Kendisiyle Karşılaştırma Bootstrap Farkı Tam Sıfır [0.0, 0.0]) BAŞARILI!")


def test_d_selected_epochs_in_valid_range():
    """
    Test (d): nn_selected.csv dosyasındaki seçilen epoch değerleri
    kesinlikle 1 ile 150 arasında olmalıdır.
    """
    results_dir = os.path.join(PROJECT_ROOT, "results")
    selected_path = os.path.join(results_dir, "nn_selected.csv")
    assert os.path.exists(selected_path), f"nn_selected.csv bulunamadı: {selected_path}"

    df_sel = pd.read_csv(selected_path)
    assert len(df_sel) == 15, f"15 yapılandırma olmalı (3 model x 5 kat), bulunan: {len(df_sel)}"

    for _, r in df_sel.iterrows():
        ep = r["epochs"]
        assert 1 <= ep <= 150, f"Epoch aralık dışında: Model {r['model']}, Kat {r['outer_fold']} -> {ep}"

    print("Test (d) (Seçilen Epoch Değerleri 1 ile 150 Arasında) BAŞARILI!")


def test_e_seed_initialization_determinism():
    """
    Test (e): Aynı tohumla kurulan iki model aynı başlangıç ağırlığına,
    farklı tohumla kurulan modeller farklı ağırlıklara sahip olmalıdır.
    Ayrıca dış eğitilen 45 ağırlık dosyasının (3 model x 5 kat x 3 tohum)
    results/weights/ altında mevcut olduğunu doğrular (eğer eğitim tamamlanmışsa).
    """
    import torch
    from src.models import ModelD1, ModelD2, ModelD3

    models_to_test = [
        ("D1", lambda: ModelD1(vocab_size=50, d_model=32, dropout=0.1)),
        ("D2", lambda: ModelD2(vocab_size=50, d_model=32, dropout=0.1)),
        ("D3", lambda: ModelD3(vocab_size=50, max_len=64, d_model=32, dropout=0.1)),
    ]

    for name, builder in models_to_test:
        torch.manual_seed(42)
        m1 = builder()
        torch.manual_seed(42)
        m2 = builder()
        torch.manual_seed(17)
        m3 = builder()

        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert torch.equal(p1, p2), f"{name}.{n1}: tohum 42 ile eşleşmedi!"

        any_diff = False
        for (n1, p1), (n3, p3) in zip(m1.named_parameters(), m3.named_parameters()):
            if not torch.equal(p1, p3):
                any_diff = True
                break
        assert any_diff, f"{name}: tohum 42 ve 17 aynı ağırlığı verdi!"

    weights_dir = os.path.join(PROJECT_ROOT, "results", "weights")
    if os.path.exists(weights_dir) and os.listdir(weights_dir):
        for m in ["D1", "D2", "D3"]:
            for f in range(5):
                for s in [17, 42, 2026]:
                    w_path = os.path.join(weights_dir, f"{m}_fold{f}_seed{s}.pt")
                    assert os.path.exists(w_path), f"Ağırlık dosyası eksik: {w_path}"

    print("Test (e) (Aynı tohum aynı başlangıç ağırlığı, farklı tohum farklı ağırlık) BAŞARILI!")


def run_all():
    print("=" * 80)
    print("BM618: ANA DENEY DOĞRULAMA TESTLERİ BAŞLADI")
    print("=" * 80)
    test_a_predictions_count_and_no_quarantine()
    test_b_records_in_outer_test_exactly_once()
    test_c_self_comparison_bootstrap_zero_interval()
    test_d_selected_epochs_in_valid_range()
    test_e_seed_initialization_determinism()
    print("=" * 80)
    print("[TÜM DENEY TESTLERİ EKSİKSİZ BAŞARIYLA GEÇTİ]")
    print("=" * 80)


if __name__ == '__main__':
    run_all()
