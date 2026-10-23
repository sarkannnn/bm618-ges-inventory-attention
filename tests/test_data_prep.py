"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Veri Hazırlama ve Model Girdisi Güvenlik Testleri (tests/test_data_prep.py)

TESTLER:
(a) Sözlük yalnız eğitimden öğreniliyor, yalnızca test kümesinde geçen bir sözcük [UNK]'a gidiyor.
(b) TF-IDF sözlüğü eğitim dışı sözcük içermiyor.
(c) Model girdi sütunlarında güç/durum/record_id/entity_group_id kesinlikle yok.
(d) Her dış kat ve her iç kat için eğitim ve doğrulama kümelerinde karantina kaydı
    (REC_2024_R13, REC_2024_R15) YOK; dış eğitim kayıt sayıları 120/120/122/121/121
    ve her dış katın iç katlarındaki toplam eğitim+doğrulama kaydı bu sayıyla aynı.
"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.normalize import turkish_lower
from src.data_prep import (
    Vocabulary,
    tokenize_text,
    format_sequence,
    load_audited_data,
    extract_features,
    prepare_tokenized_dataset,
    UNK_IDX,
    PAD_IDX,
    ALLOWED_FEATURE_COLS,
    FORBIDDEN_FEATURE_COLS
)

# UTF-8 konsol çıktısı
sys.stdout.reconfigure(encoding='utf-8')


def test_a_vocab_learned_only_from_train():
    """
    Test (a): Sözlük yalnız eğitimden öğreniliyor.
    Yalnızca test kümesinde geçen bir sözcük kesin olarak [UNK]'a (indeks 1) gitmelidir.
    """
    train_texts = [
        format_sequence("Bakırköy", 2022, "İBB Florya Kent Ormanı GES"),
        format_sequence("Fatih", 2023, "Fatih Spor Kompleksi")
    ]
    train_tokens = [tokenize_text(t) for t in train_texts]
    
    vocab = Vocabulary()
    vocab.fit(train_tokens)
    
    # Test metni: Eğitimde hiç geçmeyen sözcükler ("üsküdar", "yüzme", "havuzu", "validebağ")
    test_text = format_sequence("Üsküdar", 2025, "Validebağ Yüzme Havuzu")
    test_tokens = tokenize_text(test_text)
    encoded = vocab.encode(test_tokens)
    
    # Eğitimde olmayan kelimelerin UNK_IDX olduğunu doğrula
    novel_words = ["üsküdar", "validebağ", "yüzme", "havuzu"]
    for word in novel_words:
        assert word not in vocab.token_to_idx, f"'{word}' eğitim sözlüğüne sızmış!"
        word_encoded = vocab.encode([word])[0]
        assert word_encoded == UNK_IDX, f"'{word}' [UNK] yerine {word_encoded} kodlandı!"
        
    # Bilinen tokenların ise UNK olmadığını doğrula
    known_tokens = ["[CLS]", "[ILCE]", "[YIL]", "[MEKAN]"]
    for kt in known_tokens:
        assert vocab.encode([kt])[0] != UNK_IDX, f"Özel token {kt} UNK olamaz!"
        
    print("Test (a) (Sözlük Yalnız Eğitimden Öğreniliyor & Test Sözcükleri UNK'a Gidiyor) BAŞARILI!")


def test_b_tfidf_no_out_of_train_words():
    """
    Test (b): TF-IDF sözlüğü eğitim dışı sözcük içermiyor.
    Test verisinde geçen yeni bir kelime TF-IDF kelime dağarcığında yer almaz.
    """
    train_names = [
        "İstanbul Planlama Ajansı Florya",
        "Esenler Kemer Spor Kompleksi"
    ]
    test_names = [
        "Beylikdüzü Pazar Yeri Otoparkı"
    ]
    
    vec = TfidfVectorizer(preprocessor=turkish_lower, token_pattern=r'(?u)\b\w+\b')
    vec.fit(train_names)
    
    train_features = set(vec.get_feature_names_out())
    
    # Testte geçen "beylikdüzü", "pazar", "otoparkı" kelimelerinin train sözlüğünde olmadığını doğrula
    novel_words = ["beylikdüzü", "pazar", "yeri", "otoparkı"]
    for w in novel_words:
        assert w not in train_features, f"'{w}' eğitim TF-IDF sözlüğünde bulunamaz!"
        
    # Test verisini dönüştürdüğümüzde yeni kelimelerin sessizce yok sayıldığını doğrula
    test_vec = vec.transform(test_names)
    assert test_vec.nnz == 0, "Tamamen bilinmeyen kelimelerden oluşan test cümlesi sıfır vektör üretmelidir!"
    
    # Gerçek veri kümesi üzerinden de dış kat 0 için kontrol
    df_main, _, _ = load_audited_data()
    train_fold0 = df_main[df_main["outer_test_fold"] != 0]
    test_fold0 = df_main[df_main["outer_test_fold"] == 0]
    
    vec_fold0 = TfidfVectorizer(preprocessor=turkish_lower, token_pattern=r'(?u)\b\w+\b')
    vec_fold0.fit(train_fold0["name_clean"])
    fold0_vocab = set(vec_fold0.get_feature_names_out())
    
    # Sözlükteki tüm kelimelerin train_fold0 içinde geçtiğini teyit et
    all_train_text = " ".join([turkish_lower(s) for s in train_fold0["name_clean"]])
    for feat in fold0_vocab:
        assert feat in all_train_text, f"TF-IDF özelliği '{feat}' eğitim metinlerinde bulunamadı!"
        
    print("Test (b) (TF-IDF Sözlüğü Eğitim Dışı Sözcük İçermiyor) BAŞARILI!")


def test_c_forbidden_columns_not_in_model_inputs():
    """
    Test (c): Model girdi sütunlarında güç/durum/record_id/entity_group_id kesinlikle yok.
    Yalnızca district_clean, file_year, name_clean yer alabilir.
    """
    df_main, _, _ = load_audited_data()
    X = extract_features(df_main)
    
    # 1. Yalnızca izin verilen sütunlar var
    assert list(X.columns) == ALLOWED_FEATURE_COLS, \
        f"Girdi sütunları beklenen {ALLOWED_FEATURE_COLS} yerine {list(X.columns)} çıktı!"
        
    # 2. Yasaklı hiçbir sütun yok
    for col in FORBIDDEN_FEATURE_COLS:
        assert col not in X.columns, f"YASAKLI SÜTUN MODEL GİRDİSİNDE YER ALIYOR: {col}!"
        
    # 3. Karantina kayıtları kesinlikle df_main içinde yok
    assert (df_main["in_main_regression"] == True).all(), \
        "in_main_regression=False olan karantina kayıtları elenmemiş!"
    assert len(df_main) == 151, f"Ana regresyon veri sayısı 151 olmalı, bulunan: {len(df_main)}"
    
    print("Test (c) (Model Girdisinde Güç/Durum/ID/Grup Sütunları Kesinlikle Yok) BAŞARILI!")


def test_d_quarantine_records_completely_excluded():
    """
    Test (d): Her dış kat ve her iç kat için eğitim ve doğrulama kümelerinde
    karantina kaydı (REC_2024_R13, REC_2024_R15) KESİNLİKLE YOK;
    ayrıca dış eğitim kayıt sayıları 120/120/122/121/121 ve her dış katın
    iç katlarındaki toplam eğitim+doğrulama kaydı bu sayıyla tam olarak aynıdır.
    """
    df_main, df_splits, df_inner = load_audited_data()
    quarantine_ids = {"REC_2024_R13", "REC_2024_R15"}
    
    # 1. df_main ve df_inner içinde karantina kaydı olmamalıdır
    assert not df_main["record_id"].isin(quarantine_ids).any(), \
        "df_main içinde karantina kaydı bulundu!"
    assert not df_inner["record_id"].isin(quarantine_ids).any(), \
        "df_inner içinde karantina kaydı bulundu!"
    assert len(df_main) == 151, f"df_main 151 olmalı, bulunan: {len(df_main)}"

    # 2. Beklenen dış eğitim boyutları: 120, 120, 122, 121, 121
    expected_outer_train_counts = {
        0: 120,
        1: 120,
        2: 122,
        3: 121,
        4: 121
    }
    
    folds = sorted(df_main["outer_test_fold"].unique())
    assert folds == [0, 1, 2, 3, 4], f"Beklenen dış katlar 0-4, bulunan: {folds}"

    for fold in folds:
        outer_train = df_main[df_main["outer_test_fold"] != fold]
        outer_test = df_main[df_main["outer_test_fold"] == fold]
        expected_n = expected_outer_train_counts[fold]

        # Dış eğitim boyutu kontrolü (120/120/122/121/121)
        assert len(outer_train) == expected_n, \
            f"Kat {fold} dış eğitim sayısı {expected_n} olmalı, bulunan: {len(outer_train)}"

        # Dış eğitim ve testte karantina kaydı yok
        assert not outer_train["record_id"].isin(quarantine_ids).any(), \
            f"Kat {fold} dış eğitim kümesinde karantina kaydı saptandı!"
        assert not outer_test["record_id"].isin(quarantine_ids).any(), \
            f"Kat {fold} dış test kümesinde karantina kaydı saptandı!"

        # İç katların incelenmesi
        inner_info = df_inner[df_inner["outer_fold"] == fold]
        train_with_inner = outer_train.merge(
            inner_info[["record_id", "inner_fold"]],
            on="record_id",
            how="inner"
        )

        assert len(train_with_inner) == expected_n, \
            f"Kat {fold} iç katlar birleşim sayısı {expected_n} olmalı, bulunan: {len(train_with_inner)}"

        for inner_f in [0, 1, 2]:
            in_train = train_with_inner[train_with_inner["inner_fold"] != inner_f]
            in_val = train_with_inner[train_with_inner["inner_fold"] == inner_f]

            # Karantina kaydı kontrolü
            assert not in_train["record_id"].isin(quarantine_ids).any(), \
                f"Kat {fold} iç kat {inner_f} eğitim kümesinde karantina kaydı saptandı!"
            assert not in_val["record_id"].isin(quarantine_ids).any(), \
                f"Kat {fold} iç kat {inner_f} doğrulama kümesinde karantina kaydı saptandı!"

            # Toplam eğitim + doğrulama kaydı dış eğitim sayısıyla aynı
            total_inner = len(in_train) + len(in_val)
            assert total_inner == expected_n, \
                f"Kat {fold} iç kat {inner_f} toplamı {expected_n} olmalı, bulunan: {total_inner}"

    print("Test (d) (Karantina İzolasyonu & 120/120/122/121/121 Kayıt Sayısı Kontrolü) BAŞARILI!")


def run_all():
    print("=" * 80)
    print("BM618: VERİ HAZIRLAMA VE GİRDİ GÜVENLİK TESTLERİ BAŞLADI")
    print("=" * 80)
    test_a_vocab_learned_only_from_train()
    test_b_tfidf_no_out_of_train_words()
    test_c_forbidden_columns_not_in_model_inputs()
    test_d_quarantine_records_completely_excluded()
    print("=" * 80)
    print("[TÜM VERİ HAZIRLAMA TESTLERİ EKSİKSİZ BAŞARIYLA GEÇTİ]")
    print("=" * 80)


if __name__ == '__main__':
    run_all()
