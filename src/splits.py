"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Görev 1.3 — Dondurulmuş Protokol ve İç/Dış Grup Bölme Hattı (src/splits.py)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')


def generate_splits(audited_csv_path: str, output_splits_path: str, output_inner_splits_path: str):
    """
    KURAL VE PROTOKOL İLKELERİ:
    --------------------------
    1. Grup Anahtarı: Yalnızca ve kesinlikle `entity_group_id`.
    2. Hedef/Durum Bağımsızlığı: Atama güce (power_kwp), duruma (status) veya herhangi bir
       hedef değişkene ASLA bakılmadan gerçekleştirilmiştir.
    3. Dış Katlama: 5 dış katlı GroupKFold. Gruplar yalnızca kayıt sayılarına (grup büyüklüğü)
       göre dengelenir.
    4. Karantina Uyumu: 2024 Çekmeköy Ömerli İtfaiye çelişkili kayıtları (REC_2024_R13 ve R15)
       in_main_regression=False olarak işaretlenmiştir ancak ait oldukları grup (ENT_036)
       diğer 2025 kaydıyla aynı dış ve iç katlarda tutulmuştur.
    5. Çıktı Biçimi:
       - data/splits.csv (153 satır): record_id, entity_group_id, outer_test_fold, in_main_regression
       - data/inner_splits.csv (612 satır): outer_fold, record_id, inner_fold (yalnız o dış katın eğitim kayıtları)
    6. İç Katlama: Her dış eğitim bölümü için 3 katlı GroupKFold uygulanmıştır.
       İç katlar kesinlikle ve sadece o dış katın eğitim gruplarından türetilmiştir.
    7. Deterministik Tekrarlanabilirlik: Sabit sıralama ve tohum garantisi.
    """
    if not os.path.exists(audited_csv_path):
        raise FileNotFoundError(f"Denetlenmiş envanter dosyası bulunamadı: {audited_csv_path}")

    df = pd.read_csv(audited_csv_path)
    
    # Deterministik sıralama (dosya yılı ve excel satırına göre sabit sıra)
    df = df.sort_values(by=['file_year', 'excel_row']).reset_index(drop=True)

    # Dış Katlama: 5-katlı GroupKFold
    outer_gkf = GroupKFold(n_splits=5)
    
    outer_test_folds = {}
    for fold_idx, (train_idx, test_idx) in enumerate(outer_gkf.split(df, groups=df['entity_group_id'])):
        test_groups = df.iloc[test_idx]['entity_group_id'].unique()
        for g in test_groups:
            outer_test_folds[g] = fold_idx

    df['outer_test_fold'] = df['entity_group_id'].map(outer_test_folds)

    # 1. data/splits.csv (Kayıt başına TEK satır: 153 satır)
    splits_df = df[['record_id', 'entity_group_id', 'outer_test_fold', 'in_main_regression']].copy()
    os.makedirs(os.path.dirname(output_splits_path), exist_ok=True)
    splits_df.to_csv(output_splits_path, index=False, encoding='utf-8')
    print(f"Dış bölme dosyası kaydedildi: {output_splits_path} (Toplam satır: {len(splits_df)})")

    # 2. data/inner_splits.csv (Yalnızca dış eğitim kayıtları: 612 satır)
    inner_rows = []
    inner_gkf = GroupKFold(n_splits=3)

    for outer_f in range(5):
        outer_train_df = df[df['outer_test_fold'] != outer_f].copy()
        outer_train_df = outer_train_df.sort_values(by=['file_year', 'excel_row']).reset_index(drop=True)
        
        inner_fold_map = {}
        for inner_f, (in_tr_idx, in_val_idx) in enumerate(inner_gkf.split(outer_train_df, groups=outer_train_df['entity_group_id'])):
            val_groups = outer_train_df.iloc[in_val_idx]['entity_group_id'].unique()
            for g in val_groups:
                inner_fold_map[g] = inner_f

        for idx, r in outer_train_df.iterrows():
            inner_rows.append({
                'outer_fold': outer_f,
                'record_id': r['record_id'],
                'inner_fold': int(inner_fold_map[r['entity_group_id']])
            })

    inner_splits_df = pd.DataFrame(inner_rows)
    os.makedirs(os.path.dirname(output_inner_splits_path), exist_ok=True)
    inner_splits_df.to_csv(output_inner_splits_path, index=False, encoding='utf-8')
    print(f"İç bölme dosyası kaydedildi: {output_inner_splits_path} (Toplam satır: {len(inner_splits_df)})")

    return splits_df, inner_splits_df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_path = os.path.join(base_dir, "data", "audited_inventory.csv")
    out_splits_path = os.path.join(base_dir, "data", "splits.csv")
    out_inner_splits_path = os.path.join(base_dir, "data", "inner_splits.csv")
    
    generate_splits(csv_path, out_splits_path, out_inner_splits_path)
