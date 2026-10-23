"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Bölme Manifestosu Doğrulama ve Bütünlük Testleri (tests/test_splits.py)
"""

import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')


def test_splits_integrity():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_path = os.path.join(base_dir, "data", "audited_inventory.csv")
    splits_path = os.path.join(base_dir, "data", "splits.csv")
    inner_splits_path = os.path.join(base_dir, "data", "inner_splits.csv")

    assert os.path.exists(csv_path), "audited_inventory.csv bulunamadı"
    assert os.path.exists(splits_path), "splits.csv bulunamadı"
    assert os.path.exists(inner_splits_path), "inner_splits.csv bulunamadı"

    df_inv = pd.read_csv(csv_path)
    df_splits = pd.read_csv(splits_path)
    df_inner = pd.read_csv(inner_splits_path)

    # 1. Toplam Boyut ve Şema Kontrolleri
    assert len(df_inv) == 153, f"Envanter satır sayısı 153 olmalı, alınan: {len(df_inv)}"
    assert len(df_splits) == 153, f"splits.csv satır sayısı 153 olmalı, alınan: {len(df_splits)}"
    assert len(df_inner) == 612, f"inner_splits.csv satır sayısı 612 olmalı, alınan: {len(df_inner)}"

    expected_splits_cols = ['record_id', 'entity_group_id', 'outer_test_fold', 'in_main_regression']
    assert list(df_splits.columns) == expected_splits_cols, f"splits.csv sütunları uyuşmuyor: {list(df_splits.columns)}"

    expected_inner_cols = ['outer_fold', 'record_id', 'inner_fold']
    assert list(df_inner.columns) == expected_inner_cols, f"inner_splits.csv sütunları uyuşmuyor: {list(df_inner.columns)}"

    # (a) Her grup yalnız tek dış katta testtedir
    group_test_folds = df_splits.groupby('entity_group_id')['outer_test_fold'].nunique()
    assert (group_test_folds == 1).all(), "Hata: Bazı entity grupları birden fazla dış test katında yer alıyor!"
    assert len(group_test_folds) == 76, f"Toplam 76 grup olmalı, bulunan: {len(group_test_folds)}"

    # (b) İç katlar yalnız o dış katın dış eğitim kayıtlarından oluşuyor
    rec_to_test_fold = dict(zip(df_splits['record_id'], df_splits['outer_test_fold']))
    for outer_f in range(5):
        inner_slice = df_inner[df_inner['outer_fold'] == outer_f]
        # Bu dış katın test kayıtları
        test_rec_ids = set(df_splits[df_splits['outer_test_fold'] == outer_f]['record_id'])
        train_rec_ids = set(inner_slice['record_id'])
        
        # Test ve train kayıtları kesinlikle kesişmemeli
        intersection = test_rec_ids.intersection(train_rec_ids)
        assert len(intersection) == 0, f"Dış Kat {outer_f}: Test ve eğitim kayıtları kesişiyor: {intersection}"
        
        # Eğitim kayıt sayısı 153 - test_kayıt_sayısı olmalı
        assert len(train_rec_ids) == 153 - len(test_rec_ids)
        
        # İç kat değerleri 0, 1, 2 olmalı
        inner_vals = set(inner_slice['inner_fold'].unique())
        assert inner_vals == {0, 1, 2}, f"Dış Kat {outer_f} iç kat değerleri eksik veya hatalı: {inner_vals}"

        # İç katlarda da aynı grubun tüm kayıtları aynı inner_fold'da olmalı
        inner_merged = inner_slice.merge(df_splits[['record_id', 'entity_group_id']], on='record_id')
        inner_group_folds = inner_merged.groupby('entity_group_id')['inner_fold'].nunique()
        assert (inner_group_folds == 1).all(), f"Dış Kat {outer_f}: Bazı gruplar iç katlar arasında bölünmüş!"

    # (c) Karantina grubu (ENT_036) tüm kayıtlarıyla aynı katta
    quar_recs = df_splits[df_splits['in_main_regression'] == False]['record_id'].tolist()
    assert len(quar_recs) == 2, f"Karantina kayıt sayısı 2 olmalı, alınan: {len(quar_recs)}"
    quar_sub = df_splits[df_splits['record_id'].isin(quar_recs)]
    assert quar_sub['outer_test_fold'].nunique() == 1, "Karantina kayıtları farklı dış test katlarına düşmüş!"
    
    # ENT_036'nın 2025 kaydıyla da aynı dış katta olduğu kontrolü
    ent_036_all = df_splits[df_splits['entity_group_id'] == 'ENT_036']
    assert ent_036_all['outer_test_fold'].nunique() == 1, "ENT_036 grubunun kayıtları farklı dış katlara bölünmüş!"
    assert ent_036_all['outer_test_fold'].iloc[0] == 2, f"ENT_036 Dış Kat 2'de olmalı, bulunan: {ent_036_all['outer_test_fold'].iloc[0]}"

    # (d) Her kayıt tam bir kez dış testte yer alır
    assert df_splits['outer_test_fold'].notna().all(), "Bazı kayıtların outer_test_fold değeri eksik!"
    assert df_splits['record_id'].nunique() == 153, "Kayıtlar tekil değil!"
    assert set(df_splits['outer_test_fold'].unique()) == {0, 1, 2, 3, 4}

    # (e) Düşük ve orta güvenli gruplar tek katta
    critical_groups = ['ENT_001', 'ENT_002', 'ENT_011', 'ENT_024']
    for cg in critical_groups:
        cg_tests = df_splits[df_splits['entity_group_id'] == cg]['outer_test_fold'].unique()
        assert len(cg_tests) == 1, f"Kritik grup {cg} birden fazla dış katta test edilmiş!"

    # (f) Kat başına grup ve kayıt sayıları dengeli
    fold_recs = df_splits.groupby('outer_test_fold').size()
    fold_groups = df_splits.groupby('outer_test_fold')['entity_group_id'].nunique()
    
    expected_recs = {0: 31, 1: 31, 2: 31, 3: 30, 4: 30}
    expected_groups = {0: 15, 1: 15, 2: 16, 3: 15, 4: 15}

    print("\n[TÜM KONTROLLER VE ASSERTION'LAR EKSİKSİZ BAŞARIYLA GEÇTİ]")
    for f in range(5):
        sub_main = df_splits[(df_splits['outer_test_fold'] == f) & (df_splits['in_main_regression'] == True)]
        sub_quar = df_splits[(df_splits['outer_test_fold'] == f) & (df_splits['in_main_regression'] == False)]
        assert fold_recs[f] == expected_recs[f], f"Kat {f} kayıt sayısı uyuşmuyor: {fold_recs[f]} != {expected_recs[f]}"
        assert fold_groups[f] == expected_groups[f], f"Kat {f} grup sayısı uyuşmuyor: {fold_groups[f]} != {expected_groups[f]}"
        print(f"  Outer Fold {f}: {fold_recs[f]} test kaydı (Ana: {len(sub_main)}, Karantina: {len(sub_quar)}), {fold_groups[f]} grup")


if __name__ == '__main__':
    test_splits_integrity()
