"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Metrikler ve Bootstrap Doğrulama Testleri (tests/test_metrics.py)
"""

import os
import sys
import numpy as np

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.metrics import (
    compute_facility_balanced_mae,
    compute_record_mae,
    compute_rmse,
    compute_r2,
    compute_log_mae,
    compute_all_metrics,
    paired_bootstrap_facility_mae_ci
)

# UTF-8 konsol çıktısı
sys.stdout.reconfigure(encoding='utf-8')


def test_facility_balanced_mae_hand_calculated():
    """
    Test 1: Elle hesaplanmış küçük örnekte tesis dengeli MAE doğrulaması.
    Grup büyüklükleri farklı olduğunda tesis dengeli MAE ile kayıt MAE'nin
    farklı sonuç verdiği kesin olarak gösterilir.
    
    Örnek:
    Grup 1: 1 kayıt -> y=100, y_hat=110 (Hata = 10, Grup 1 MAE = 10.0)
    Grup 2: 3 kayıt -> y=[50, 60, 70], y_hat=[50, 60, 70] (Hatalar = [0, 0, 0], Grup 2 MAE = 0.0)
    
    Beklenen Tesis Dengeli MAE = (10.0 + 0.0) / 2 = 5.0 kWp
    Beklenen Kayıt MAE = (10 + 0 + 0 + 0) / 4 = 2.5 kWp
    """
    y_true = [100.0, 50.0, 60.0, 70.0]
    y_pred = [110.0, 50.0, 60.0, 70.0]
    groups = ["ENT_01", "ENT_02", "ENT_02", "ENT_02"]
    
    fb_mae = compute_facility_balanced_mae(y_true, y_pred, groups)
    rec_mae = compute_record_mae(y_true, y_pred)
    
    assert abs(fb_mae - 5.0) < 1e-9, f"Tesis dengeli MAE beklenen 5.0 yerine {fb_mae} çıktı!"
    assert abs(rec_mae - 2.5) < 1e-9, f"Kayıt MAE beklenen 2.5 yerine {rec_mae} çıktı!"
    assert fb_mae != rec_mae, "Grup büyüklükleri farklıyken tesis dengeli MAE ve kayıt MAE eşit olamaz!"
    print(f"Test 1 (Elle Hesap Tesis Dengeli MAE = {fb_mae:.1f} vs Kayıt MAE = {rec_mae:.1f}) BAŞARILI!")


def test_secondary_metrics():
    """
    Test 2: İkincil metriklerin (RMSE, R2, Log MAE) doğrulanması.
    """
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 34.0])
    groups = ["G1", "G2", "G3"]
    
    # Hatalar: [2.0, -2.0, 4.0]
    # Kareler: [4.0, 4.0, 16.0] -> Ort: 24/3 = 8.0 -> RMSE = sqrt(8) ~ 2.828427
    expected_rmse = np.sqrt(8.0)
    actual_rmse = compute_rmse(y_true, y_pred)
    assert abs(actual_rmse - expected_rmse) < 1e-6, "RMSE hesabı hatalı!"
    
    # R2 hesabı: y_mean = 20.0, ss_tot = (-10)^2 + 0^2 + 10^2 = 200.0, ss_res = 24.0 -> R2 = 1 - 24/200 = 0.88
    expected_r2 = 0.88
    actual_r2 = compute_r2(y_true, y_pred)
    assert abs(actual_r2 - expected_r2) < 1e-6, "R2 hesabı hatalı!"
    
    # Log MAE hesabı:
    # log1p(y_true) = [log(11), log(21), log(31)]
    # log1p(y_pred) = [log(13), log(19), log(35)]
    expected_log_mae = np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)))
    actual_log_mae = compute_log_mae(y_true, y_pred)
    assert abs(actual_log_mae - expected_log_mae) < 1e-6, "Log MAE hesabı hatalı!"
    
    all_m = compute_all_metrics(y_true, y_pred, groups)
    assert "facility_balanced_mae" in all_m and "rmse" in all_m and "r2" in all_m
    print("Test 2 (İkincil Metrikler: RMSE, R2, Log MAE) BAŞARILI!")


def test_paired_bootstrap():
    """
    Test 3: Eşleştirilmiş Bootstrap fonksiyonu kontrolü.
    Özdeş modellerde fark tam 0.0 ve %95 güven aralığı [0.0, 0.0] olmalıdır.
    Model A bariz daha kötüyse fark ve güven aralığı pozitif olmalıdır.
    """
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred_a = np.array([12.0, 22.0, 32.0, 42.0])  # Her biri +2 hata
    y_pred_b = np.array([12.0, 22.0, 32.0, 42.0])  # Birebir aynı
    groups = ["G1", "G2", "G3", "G4"]
    
    # Özdeş tahminler
    ci_low, ci_high, mean_diff = paired_bootstrap_facility_mae_ci(
        y_true, y_pred_a, y_pred_b, groups, n_boot=500, seed=42
    )
    assert abs(mean_diff) < 1e-9, "Özdeş tahminlerde mean_diff 0 olmalıdır!"
    assert abs(ci_low) < 1e-9 and abs(ci_high) < 1e-9, "Özdeş tahminlerde CI [0, 0] olmalıdır!"
    
    # Model A daha kötü tahmin yaparken
    y_pred_bad = np.array([30.0, 40.0, 50.0, 60.0])  # Her biri +20 hata
    ci_low2, ci_high2, mean_diff2 = paired_bootstrap_facility_mae_ci(
        y_true, y_pred_bad, y_pred_b, groups, n_boot=500, seed=42
    )
    assert mean_diff2 > 0, "Daha kötü modelin farkı pozitif olmalıdır!"
    assert ci_low2 > 0, "Model A anlamlı derecede kötüyse CI alt sınırı 0'dan büyük olmalıdır!"
    print(f"Test 3 (Eşleştirilmiş Bootstrap %95 GA: [{ci_low2:.2f}, {ci_high2:.2f}]) BAŞARILI!")


def run_all():
    print("=" * 80)
    print("BM618: METRİK VE BOOTSTRAP DOĞRULAMA TESTLERİ BAŞLADI")
    print("=" * 80)
    test_facility_balanced_mae_hand_calculated()
    test_secondary_metrics()
    test_paired_bootstrap()
    print("=" * 80)
    print("[TÜM METRİK TESTLERİ EKSİKSİZ BAŞARIYLA GEÇTİ]")
    print("=" * 80)


if __name__ == '__main__':
    run_all()
