"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Veri Hazırlama, Tokenizasyon ve Sözlük Modülü (src/data_prep.py)

KURALLAR:
- Karantina kayıtları (Ömerli 2024, REC_2024_R13 ve REC_2024_R15) eğitim, doğrulama ve testten
  kesinlikle hariç tutulur (yalnız in_main_regression=True kayıtları kullanılır).
- Model girdisi YALNIZ: district_clean (ilçe), file_year (dosya yılı/kaynak sürümü) ve name_clean (ad).
  power_kwp, status_clean, record_id, excel_row, entity_group_id MODEL GİRDİSİNE GİREMEZ.
- entity_group_id yalnız metrik hesabında ve grup doğrulamalarında kullanılır.
- Metin biçimi: "[CLS] [ILCE] <ilçe> [YIL] <yıl> [MEKAN] <ad>"
- Türkçe İ/ı kurallarına dikkat edilir, Türkçe karakterler ve etap numaraları korunur.
- Sözlük ve TF-IDF YALNIZ eğitim kümesinden öğrenilir.
- Hedef: z = log1p(power_kwp); ters dönüşüm: y_hat = max(0, expm1(z_hat)).
"""

import os
import sys
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.normalize import turkish_lower

# İzin verilen ve yasaklı model girdi sütunları
ALLOWED_FEATURE_COLS = ["district_clean", "file_year", "name_clean"]
FORBIDDEN_FEATURE_COLS = [
    "power_kwp", "status_raw", "status_clean", "record_id",
    "excel_row", "entity_group_id", "source_file", "sheet",
    "quality_flag", "in_main_regression", "match_confidence"
]

# Özel tokenlar
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
ILCE_TOKEN = "[ILCE]"
YIL_TOKEN = "[YIL]"
MEKAN_TOKEN = "[MEKAN]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, ILCE_TOKEN, YIL_TOKEN, MEKAN_TOKEN]
PAD_IDX = 0
UNK_IDX = 1
CLS_IDX = 2
ILCE_IDX = 3
YIL_IDX = 4
MEKAN_IDX = 5


def format_sequence(district: str, year: int or str, name: str, include_year: bool = True) -> str:
    """
    Girdi alanlarını birleştirerek model metin formatına çevirir:
    "[CLS] [ILCE] <ilçe> [YIL] <yıl> [MEKAN] <ad>"
    Türkçe İ/ı ve karakterleri korur.
    """
    d_clean = turkish_lower(str(district))
    y_clean = str(year).strip()
    n_clean = turkish_lower(str(name))
    if not include_year:
        # Ablation 3: dosya yılı bilgisi ([YIL] etiketi ve yıl değeri) girdiden çıkarılır
        return f"{CLS_TOKEN} {ILCE_TOKEN} {d_clean} {MEKAN_TOKEN} {n_clean}"
    return f"{CLS_TOKEN} {ILCE_TOKEN} {d_clean} {YIL_TOKEN} {y_clean} {MEKAN_TOKEN} {n_clean}"


def tokenize_text(text: str) -> List[str]:
    """
    Metni özel etiketleri ([CLS], [ILCE] vb.), kelimeleri ve noktalama/alt birim işaretlerini
    koruyarak ayrıştırır.
    """
    pattern = r'\[[A-Z]+\]|[\wçğıöşü]+|[^\w\s]'
    tokens = re.findall(pattern, text)
    return tokens


class Vocabulary:
    """
    Yalnızca eğitim verisinden öğrenilen sözlük sınıfı.
    Özel tokenlar: [PAD]: 0, [UNK]: 1, [CLS]: 2, [ILCE]: 3, [YIL]: 4, [MEKAN]: 5.
    Eğitim kümesinde yer almayan tüm sözcükler otomatik olarak [UNK] (1) indeksine yönlendirilir.
    """
    def __init__(self, special_tokens: Optional[List[str]] = None):
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        for idx, tok in enumerate(special_tokens):
            self.token_to_idx[tok] = idx
            self.idx_to_token[idx] = tok

    def fit(self, tokenized_texts: List[List[str]]) -> "Vocabulary":
        """Eğitim dizilerinden sözlüğü oluşturur."""
        for tokens in tokenized_texts:
            for token in tokens:
                if token not in self.token_to_idx:
                    new_idx = len(self.token_to_idx)
                    self.token_to_idx[token] = new_idx
                    self.idx_to_token[new_idx] = token
        return self

    def encode(self, tokens: List[str]) -> List[int]:
        """Token listesini sayısal indekslere çevirir (Bilinmeyenler UNK_IDX)."""
        return [self.token_to_idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Sayısal indeksleri token listesine çevirir."""
        return [self.idx_to_token.get(i, UNK_TOKEN) for i in indices]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)


def power_to_log1p(power_kwp: np.ndarray or float) -> np.ndarray or float:
    """Hedef değişken log1p dönüşümü: z = log(1 + power_kwp)"""
    return np.log1p(np.asarray(power_kwp, dtype=np.float64))


def log1p_to_power(z: np.ndarray or float) -> np.ndarray or float:
    """Ters dönüşüm: y_hat = max(0, exp(z) - 1)"""
    return np.maximum(0.0, np.expm1(np.asarray(z, dtype=np.float64)))


def load_audited_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Audited envanter, dış bölmeler ve iç bölmeler tablolarını yükler.
    Karantina kayıtlarını (in_main_regression=False) kesin olarak eler.
    """
    inv_path = os.path.join(PROJECT_ROOT, "data", "audited_inventory.csv")
    splits_path = os.path.join(PROJECT_ROOT, "data", "splits.csv")
    inner_splits_path = os.path.join(PROJECT_ROOT, "data", "inner_splits.csv")

    df_inv = pd.read_csv(inv_path)
    df_splits = pd.read_csv(splits_path)
    df_inner = pd.read_csv(inner_splits_path)

    # Karantina filtreleme (YALNIZ in_main_regression == True)
    df_inv_main = df_inv[df_inv["in_main_regression"] == True].copy()
    df_splits_main = df_splits[df_splits["in_main_regression"] == True].copy()
    
    # df_inner'da da yalnız in_main_regression olan record_id'ler kalsın
    valid_ids = set(df_inv_main["record_id"])
    df_inner_main = df_inner[df_inner["record_id"].isin(valid_ids)].copy()

    # Birleştirme (audited data ile dış kat ataması)
    df_main = df_inv_main.merge(
        df_splits_main[["record_id", "outer_test_fold"]],
        on="record_id",
        how="inner"
    )

    return df_main, df_splits_main, df_inner_main


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model için YALNIZCA izin verilen girdi özelliklerini döndürür.
    Yasaklı özelliklerin bulunmadığını kesinlikle kontrol eder (assert).
    """
    for forbidden in FORBIDDEN_FEATURE_COLS:
        # Eğer dışarıdan dataframe içinde gelmişse bile, model özellik setine alınmamalıdır
        pass
    
    # Model girdisi: YALNIZ district_clean, file_year, name_clean
    X = df[ALLOWED_FEATURE_COLS].copy()
    
    for forbidden in FORBIDDEN_FEATURE_COLS:
        assert forbidden not in X.columns, f"YASAKLI SÜTUN MODEL GİRDİSİNDE BULUNDU: {forbidden}"
        
    return X


def prepare_tokenized_dataset(
    df_subset: pd.DataFrame,
    vocab: Optional[Vocabulary] = None,
    max_len: Optional[int] = None,
    fit_vocab: bool = False,
    drop_year: bool = False
) -> Dict[str, any]:
    """
    Verilen DataFrame alt kümesi için metin dizilerini hazırlar, tokenlara ayırır ve tensörleri üretir.
    fit_vocab=True ise sözlük ve azami uzunluk bu eğitim kümesinden öğrenilir.
    """
    assert "in_main_regression" not in df_subset.columns or (df_subset["in_main_regression"] == True).all(), \
        "Karantina kayıtları veri hazırlamaya dahil edilemez!"

    formatted_texts = [
        format_sequence(row["district_clean"], row["file_year"], row["name_clean"], include_year=not drop_year)
        for _, row in df_subset.iterrows()
    ]
    tokenized_texts = [tokenize_text(text) for text in formatted_texts]

    # Sözlük oluşturma (yalnız eğitimde)
    if fit_vocab or vocab is None:
        vocab = Vocabulary()
        vocab.fit(tokenized_texts)

    # Azami uzunluk (üst sınır 64)
    raw_lengths = [len(t) for t in tokenized_texts]
    if max_len is None:
        max_len = min(64, max(raw_lengths) if raw_lengths else 64)

    # Kesilen kayıt sayısı raporlaması
    truncated_count = sum(1 for length in raw_lengths if length > max_len)

    # Tensörleri doldurma: token_ids ve key_padding_mask
    N = len(tokenized_texts)
    token_ids = np.zeros((N, max_len), dtype=np.int64)
    padding_mask = np.zeros((N, max_len), dtype=bool)  # True = maskeli (padding)

    for i, tokens in enumerate(tokenized_texts):
        encoded = vocab.encode(tokens)
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        length = len(encoded)
        token_ids[i, :length] = encoded
        token_ids[i, length:] = PAD_IDX
        padding_mask[i, length:] = True  # PAD olan pozisyonlar True

    # Hedef değerler
    power_kwp = df_subset["power_kwp"].to_numpy(dtype=np.float64) if "power_kwp" in df_subset.columns else None
    targets_z = power_to_log1p(power_kwp) if power_kwp is not None else None

    return {
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "key_padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        "targets_z": torch.tensor(targets_z, dtype=torch.float32) if targets_z is not None else None,
        "y_true": power_kwp,
        "vocab": vocab,
        "max_len": max_len,
        "truncated_count": truncated_count,
        "record_ids": df_subset["record_id"].tolist(),
        "entity_groups": df_subset["entity_group_id"].tolist()
    }
