"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Referans Modeller (Baselines B0 ve B1) Modülü (src/baselines.py)

MODELLER:
- B0: Eğitim kümesinin güç medyanı (kWp). Tüm test kayıtlarına aynı sabit kestirim.
- B1: TF-IDF (tesis adı) + One-Hot (ilçe ve yıl, handle_unknown='ignore') + Ridge regresyon.
  Hedef z = log1p(power_kwp). Ters dönüşüm y_hat = max(0, expm1(z_hat)).
  Hiperparametre kümesi: alpha in {0.1, 1.0, 10.0, 100.0}.

PROTOKOL KURALLARI:
- Her dış katta (0-4), dış eğitim verisi inner_splits.csv ile 3 iç kata bölünür.
- Her alpha adayı için 3 katlı iç doğrulama tesis dengeli MAE'si (kWp) hesaplanır.
- En düşük iç doğrulama tesis dengeli MAE'sine sahip alpha seçilir.
- Seçilen alpha ile dış eğitimin TAMAMI üzerinde pipeline yeniden eğitilir.
- Dış test kümesinde YALNIZ BİR KEZ tahmin yapılır.
- Dış test sonucuna göre model veya hiperparametre seçimi KESİNLİKLE YAPILMAZ.
- Çıktılar:
  * results/baselines_predictions.csv (record_id, outer_fold, model, y_true, y_pred)
  * results/baselines_metrics.csv (model, fold, facility_balanced_mae, record_mae, rmse, r2, log_mae)
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.normalize import turkish_lower
from src.data_prep import (
    load_audited_data,
    power_to_log1p,
    log1p_to_power,
    extract_features
)
from src.metrics import (
    compute_facility_balanced_mae,
    compute_all_metrics
)

CANDIDATE_ALPHAS = [0.1, 1.0, 10.0, 100.0]


def build_b1_pipeline(alpha: float) -> Pipeline:
    """
    B1 Referans Modeli Boru Hattı (Pipeline).
    TÜM dönüşümler (TF-IDF ve OneHot) pipeline içinde fit edilir (veri sızıntısı engellenir).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                'text',
                TfidfVectorizer(
                    preprocessor=turkish_lower,
                    token_pattern=r'(?u)\b\w+\b'
                ),
                'name_clean'
            ),
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore'),
                ['district_clean', 'file_year']
            )
        ]
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=alpha, random_state=42))
    ])


def run_b0(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    B0 Referans Modeli (Eğitim Kümesi Medyanı).
    Her dış katta, dış eğitimin güç medyanını test kayıtlarına tahmin eder.
    Karantina kayıtları (in_main_regression=False) kesinlikle hariç tutulur.
    """
    # Karantina filtreleme kontrolü
    assert len(df_main) == 151, f"df_main 151 ana kayıt içermelidir, bulunan: {len(df_main)}"
    assert not df_main['record_id'].isin({'REC_2024_R13', 'REC_2024_R15'}).any(), \
        "Karantina kayıtları B0 modeline sızamaz!"

    folds = sorted(df_main['outer_test_fold'].unique())
    preds = []
    expected_train_counts = {0: 120, 1: 120, 2: 122, 3: 121, 4: 121}

    for fold in folds:
        train_df = df_main[df_main['outer_test_fold'] != fold]
        test_df = df_main[df_main['outer_test_fold'] == fold]

        assert len(train_df) == expected_train_counts[fold], \
            f"Kat {fold} dış eğitim sayısı {expected_train_counts[fold]} olmalı, bulunan: {len(train_df)}"

        median_power = float(np.median(train_df['power_kwp']))

        for _, row in test_df.iterrows():
            preds.append({
                'record_id': row['record_id'],
                'outer_fold': int(fold),
                'model': 'B0',
                'y_true': float(row['power_kwp']),
                'y_pred': median_power,
                'entity_group_id': row['entity_group_id']
            })

    return pd.DataFrame(preds)


def run_b1(
    df_main: pd.DataFrame,
    df_inner: pd.DataFrame,
    candidate_alphas: List[float] = CANDIDATE_ALPHAS
) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """
    B1 Referans Modeli (TF-IDF + One-Hot + Ridge).
    İç çapraz doğrulama ile alpha seçimi, ardından dış test tahmini.
    Karantina kayıtları (in_main_regression=False) kesinlikle hariç tutulur.
    """
    # Karantina filtreleme kontrolleri
    assert len(df_main) == 151, f"df_main 151 ana kayıt içermelidir, bulunan: {len(df_main)}"
    assert not df_main['record_id'].isin({'REC_2024_R13', 'REC_2024_R15'}).any(), \
        "Karantina kayıtları B1 dış eğitim/test kümesine sızamaz!"
    assert not df_inner['record_id'].isin({'REC_2024_R13', 'REC_2024_R15'}).any(), \
        "Karantina kayıtları B1 iç katlarına sızamaz!"

    folds = sorted(df_main['outer_test_fold'].unique())
    preds = []
    tuning_log = {}
    expected_train_counts = {0: 120, 1: 120, 2: 122, 3: 121, 4: 121}

    for outer_fold in folds:
        outer_train = df_main[df_main['outer_test_fold'] != outer_fold].copy()
        outer_test = df_main[df_main['outer_test_fold'] == outer_fold].copy()

        assert len(outer_train) == expected_train_counts[outer_fold], \
            f"Kat {outer_fold} dış eğitim sayısı {expected_train_counts[outer_fold]} olmalı!"

        # İç kat atamaları
        inner_info = df_inner[df_inner['outer_fold'] == outer_fold]
        train_with_inner = outer_train.merge(
            inner_info[['record_id', 'inner_fold']],
            on='record_id',
            how='inner'
        )

        assert len(train_with_inner) == expected_train_counts[outer_fold], \
            f"Kat {outer_fold} iç katlar birleşim sayısı {expected_train_counts[outer_fold]} olmalı!"

        # İç çapraz doğrulama ile alpha seçimi
        alpha_cv_scores = {}
        for alpha in candidate_alphas:
            fold_maes = []
            for inner_f in [0, 1, 2]:
                in_train = train_with_inner[train_with_inner['inner_fold'] != inner_f]
                in_val = train_with_inner[train_with_inner['inner_fold'] == inner_f]

                pipe = build_b1_pipeline(alpha=alpha)
                X_tr = extract_features(in_train)
                z_tr = power_to_log1p(in_train['power_kwp'])
                pipe.fit(X_tr, z_tr)

                X_val = extract_features(in_val)
                z_hat_val = pipe.predict(X_val)
                y_hat_val = log1p_to_power(z_hat_val)

                val_mae = compute_facility_balanced_mae(
                    in_val['power_kwp'],
                    y_hat_val,
                    in_val['entity_group_id']
                )
                fold_maes.append(val_mae)

            alpha_cv_scores[alpha] = float(np.mean(fold_maes))

        # En iyi alpha seçimi (en düşük iç doğrulama tesis dengeli MAE)
        best_alpha = min(alpha_cv_scores, key=alpha_cv_scores.get)
        tuning_log[int(outer_fold)] = {
            'best_alpha': best_alpha,
            'cv_scores': alpha_cv_scores
        }

        # Seçilen alpha ile TÜM dış eğitimde yeniden eğit
        best_pipe = build_b1_pipeline(alpha=best_alpha)
        X_out_train = extract_features(outer_train)
        z_out_train = power_to_log1p(outer_train['power_kwp'])
        best_pipe.fit(X_out_train, z_out_train)

        # Dış testte YALNIZ BİR KEZ tahmin
        X_out_test = extract_features(outer_test)
        z_hat_test = best_pipe.predict(X_out_test)
        y_hat_test = log1p_to_power(z_hat_test)

        for (_, row), yp in zip(outer_test.iterrows(), y_hat_test):
            preds.append({
                'record_id': row['record_id'],
                'outer_fold': int(outer_fold),
                'model': 'B1',
                'y_true': float(row['power_kwp']),
                'y_pred': float(yp),
                'entity_group_id': row['entity_group_id']
            })

    return pd.DataFrame(preds), tuning_log


def compute_metrics_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Kat bazında ve birleştirilmiş (pooled) metrik tablosu üretir."""
    results = []
    models = pred_df['model'].unique()
    folds = sorted(pred_df['outer_fold'].unique())

    for model_name in models:
        m_df = pred_df[pred_df['model'] == model_name]

        # Kat bazında metrikler
        for f in folds:
            f_sub = m_df[m_df['outer_fold'] == f]
            met = compute_all_metrics(
                f_sub['y_true'],
                f_sub['y_pred'],
                f_sub['entity_group_id']
            )
            met['model'] = model_name
            met['fold'] = str(f)
            results.append(met)

        # Birleştirilmiş (pooled) metrikler
        pooled_met = compute_all_metrics(
            m_df['y_true'],
            m_df['y_pred'],
            m_df['entity_group_id']
        )
        pooled_met['model'] = model_name
        pooled_met['fold'] = 'pooled'
        results.append(pooled_met)

    cols_order = ['model', 'fold', 'facility_balanced_mae', 'record_mae', 'rmse', 'r2', 'log_mae']
    return pd.DataFrame(results)[cols_order]


def run_and_save_baselines() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """B0 ve B1 modellerini çalıştırır ve sonuçları results/ dizinine kaydeder."""
    df_main, _, df_inner = load_audited_data()

    preds_b0 = run_b0(df_main)
    preds_b1, tuning_log = run_b1(df_main, df_inner)

    combined_preds = pd.concat([preds_b0, preds_b1], ignore_index=True)

    # Dizinleri oluştur
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # results/baselines_predictions.csv (record_id, outer_fold, model, y_true, y_pred)
    save_pred_cols = ['record_id', 'outer_fold', 'model', 'y_true', 'y_pred']
    preds_path = os.path.join(results_dir, "baselines_predictions.csv")
    combined_preds[save_pred_cols].to_csv(preds_path, index=False, encoding='utf-8')

    # results/baselines_metrics.csv
    metrics_df = compute_metrics_table(combined_preds)
    metrics_path = os.path.join(results_dir, "baselines_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')

    return combined_preds, metrics_df, tuning_log


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 80)
    print("BM618: B0 VE B1 REFERANS MODELLERİ ÇALIŞTIRILIYOR...")
    print("=" * 80)
    preds, metrics, log = run_and_save_baselines()
    print("\nSeçilen Alpha Değerleri (İç Doğrulama İle):")
    for fold, info in log.items():
        print(f"  Kat {fold}: alpha={info['best_alpha']} | CV Skorları: {info['cv_scores']}")
    print("\nBirleştirilmiş ve Kat Bazında Metrik Tablosu:")
    print(metrics.to_string(index=False))
    print("=" * 80)
    print("Sonuçlar results/ klasörüne kaydedildi.")
