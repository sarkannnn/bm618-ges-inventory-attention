"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Derin Öğrenme Modelleri ve M3 Doğrulama Testleri (tests/test_models.py)

TESTLER:
(a) D3 encoder blokları için toplam parametre tam 17.088 (M3 sayımı; son LayerNorm ayrı sayılır).
(b) D3 padding invariance: aynı kayda fazladan [PAD] eklenince kestirim değişmiyor (eval, dropout kapalı).
(c) D2 ağırlıkları PAD dışında toplam 1, PAD'de tam 0.
(d) Üç model NaN üretmiyor, çıktı şekli (B,).
(e) Üç modelin regresyon başlığı aynı yapıda (Linear(32,32) -> ReLU -> Dropout -> Linear(32,1)).
"""

import os
import sys
import torch
import numpy as np

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models import (
    ModelD1,
    ModelD2,
    ModelD3,
    SharedRegressionHead,
    count_parameters_detailed
)
from src.data_prep import PAD_IDX

# UTF-8 konsol çıktısı
sys.stdout.reconfigure(encoding='utf-8')


def test_a_d3_encoder_parameters_exact_17088():
    """
    Test (a): M3 Görevi Parametre Sayımı Doğrulaması.
    d=32, h=4, d_ff=64 olan iki Pre-LN encoder bloğunun toplam parametre sayısı
    KESİNLİKLE VE TAM OLARAK 17.088 olmalıdır.
    (Token embedding, konum embedding, son LayerNorm ve regresyon başlığı ayrı sayılır).
    """
    vocab_size = 100
    model_d3 = ModelD3(vocab_size=vocab_size, max_len=64, d_model=32, num_heads=4, d_ff=64)
    counts = count_parameters_detailed(model_d3)

    b1_params = counts["encoder_block1"]
    b2_params = counts["encoder_block2"]
    encoder_total = counts["encoder_blocks_total"]

    # Blok başına hesap:
    # MHSA (Q, K, V, O bias ile): 4 x (32x32 + 32) = 4224
    # LN1 (gamma, beta): 2 x 32 = 64
    # FFN (Linear1 + Linear2 bias ile): (32x64 + 64) + (64x32 + 32) = 2112 + 2080 = 4192
    # LN2 (gamma, beta): 2 x 32 = 64
    # Toplam 1 blok = 4224 + 64 + 4192 + 64 = 8544
    # Toplam 2 blok = 8544 x 2 = 17088
    assert b1_params == 8544, f"Blok 1 parametresi beklenen 8544 yerine {b1_params} çıktı!"
    assert b2_params == 8544, f"Blok 2 parametresi beklenen 8544 yerine {b2_params} çıktı!"
    assert encoder_total == 17088, f"D3 encoder blokları toplam parametresi beklenen 17.088 yerine {encoder_total} çıktı!"

    # Son LayerNorm ve regresyon başlığı ayrı sayılır
    assert counts["final_layernorm"] == 64, f"Son LayerNorm beklenen 64 yerine {counts['final_layernorm']} çıktı!"
    assert counts["regression_head"] == 1089, f"Regresyon başlığı beklenen 1089 yerine {counts['regression_head']} çıktı!"

    print(f"Test (a) (D3 Encoder Parametre Sayımı = {encoder_total} [M3: 17.088 TAM UYUMLU]) BAŞARILI!")


def test_b_d3_padding_invariance():
    """
    Test (b): D3 Padding Invariance.
    Aynı girdinin sonuna eklenen [PAD] tokenları çıktıyı değiştirmemelidir (eval modu, dropout=0).
    """
    torch.manual_seed(42)
    vocab_size = 50
    model_d3 = ModelD3(vocab_size=vocab_size, max_len=32, d_model=32, dropout=0.0)
    model_d3.eval()

    # 4 gerçek token (0. token [CLS])
    seq_real = torch.tensor([[2, 10, 15, 20]], dtype=torch.long)
    mask_real = torch.tensor([[False, False, False, False]], dtype=torch.bool)

    # Aynı sekansa 3 PAD eklenmiş hali
    seq_padded = torch.tensor([[2, 10, 15, 20, PAD_IDX, PAD_IDX, PAD_IDX]], dtype=torch.long)
    mask_padded = torch.tensor([[False, False, False, False, True, True, True]], dtype=torch.bool)

    with torch.no_grad():
        out_real = model_d3(seq_real, padding_mask=mask_real)
        out_padded = model_d3(seq_padded, padding_mask=mask_padded)

    diff = torch.abs(out_real - out_padded).max().item()
    assert diff < 1e-6, f"D3 padding invariance başarısız! Maksimum fark: {diff}"
    print(f"Test (b) (D3 Padding Invariance, Maksimum fark: {diff:.2e} < 1e-6) BAŞARILI!")


def test_c_d2_attention_weights_properties():
    """
    Test (c): D2 Bahdanau Attention Ağırlıkları.
    Ağırlıklar PAD dışındaki geçerli tokenlarda tam 1'e toplanmalı,
    PAD olan konumlarda kesinlikle 0.0 olmalıdır.
    """
    torch.manual_seed(42)
    vocab_size = 50
    model_d2 = ModelD2(vocab_size=vocab_size, d_model=32, dropout=0.0)
    model_d2.eval()

    # Batch boyutu 2: 1. örnek 3 token + 2 PAD, 2. örnek 4 token + 1 PAD
    x = torch.tensor([
        [2, 10, 15, PAD_IDX, PAD_IDX],
        [2, 10, 15, 20, PAD_IDX]
    ], dtype=torch.long)
    mask = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, False, True]
    ], dtype=torch.bool)

    with torch.no_grad():
        out, weights = model_d2(x, padding_mask=mask, return_attention_weights=True)

    # weights: (B, n)
    assert weights.shape == (2, 5), f"Ağırlık şekli (2, 5) olmalı, bulunan: {weights.shape}"

    # 1. PAD pozisyonları kesin 0.0 olmalı
    assert (weights[0, 3:] == 0.0).all(), "D2 1. örnek PAD ağırlıkları 0.0 değil!"
    assert (weights[1, 4:] == 0.0).all(), "D2 2. örnek PAD ağırlıkları 0.0 değil!"

    # 2. Geçerli pozisyonların toplamı 1.0 olmalı
    sum_valid_0 = weights[0, :3].sum().item()
    sum_valid_1 = weights[1, :4].sum().item()
    assert abs(sum_valid_0 - 1.0) < 1e-6, f"1. örnek ağırlık toplamı {sum_valid_0} != 1.0!"
    assert abs(sum_valid_1 - 1.0) < 1e-6, f"2. örnek ağırlık toplamı {sum_valid_1} != 1.0!"

    # 3. Eşit ağırlıklandırma seçeneği (Ablation 1)
    with torch.no_grad():
        out_eq, weights_eq = model_d2(x, padding_mask=mask, return_attention_weights=True, equal_weights=True)
    expected_w0 = torch.tensor([1/3, 1/3, 1/3, 0.0, 0.0])
    expected_w1 = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.0])
    assert torch.allclose(weights_eq[0], expected_w0, atol=1e-6), "Ablation 1 eşit ağırlıklar hatalı!"
    assert torch.allclose(weights_eq[1], expected_w1, atol=1e-6), "Ablation 1 eşit ağırlıklar hatalı!"

    print("Test (c) (D2 Attention Ağırlıkları: PAD=0, Geçerli Toplamı=1.0, Eşit Ağırlık Doğrulandı) BAŞARILI!")


def test_d_no_nans_and_output_shapes():
    """
    Test (d): Üç model (D1, D2, D3) hiçbir durumda NaN üretmez ve çıktı şekli tam (B,)'dir.
    """
    vocab_size = 60
    batch_size = 4
    seq_len = 10

    d1 = ModelD1(vocab_size=vocab_size, d_model=32)
    d2 = ModelD2(vocab_size=vocab_size, d_model=32)
    d3 = ModelD3(vocab_size=vocab_size, max_len=seq_len, d_model=32)

    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    mask[:, 7:] = True  # Son 3 token padding

    for name, model in [("D1", d1), ("D2", d2), ("D3", d3)]:
        model.eval()
        with torch.no_grad():
            out = model(x, padding_mask=mask)
        assert out.shape == (batch_size,), f"{name} çıktı şekli beklenen ({batch_size},) yerine {out.shape} çıktı!"
        assert not torch.isnan(out).any(), f"{name} çıktısında NaN tespit edildi!"
        assert not torch.isinf(out).any(), f"{name} çıktısında Inf tespit edildi!"

    print("Test (d) (D1, D2, D3 NaN Üretmiyor ve Çıktı Şekli (B,)) BAŞARILI!")


def test_e_identical_regression_heads():
    """
    Test (e): Üç modelin regresyon başlığı birebir aynı mimariye ve parametre sayısına sahiptir:
    Linear(32, 32) -> ReLU -> Dropout -> Linear(32, 1) -> Tam 1089 parametre.
    """
    vocab_size = 50
    d1 = ModelD1(vocab_size=vocab_size, d_model=32)
    d2 = ModelD2(vocab_size=vocab_size, d_model=32)
    d3 = ModelD3(vocab_size=vocab_size, d_model=32)

    # Başlık türü kontrolü
    assert isinstance(d1.head, SharedRegressionHead)
    assert isinstance(d2.head, SharedRegressionHead)
    assert isinstance(d3.head, SharedRegressionHead)

    # Parametre sayıları kontrolü
    p1 = sum(p.numel() for p in d1.head.parameters())
    p2 = sum(p.numel() for p in d2.head.parameters())
    p3 = sum(p.numel() for p in d3.head.parameters())

    expected_head_params = (32 * 32 + 32) + (32 * 1 + 1)  # 1056 + 33 = 1089
    assert p1 == expected_head_params, f"D1 başlık parametresi {p1} != 1089"
    assert p2 == expected_head_params, f"D2 başlık parametresi {p2} != 1089"
    assert p3 == expected_head_params, f"D3 başlık parametresi {p3} != 1089"

    print("Test (e) (Üç Modelin Regresyon Başlığı Birebir Özdeş: Tam 1089 Parametre) BAŞARILI!")


def run_all():
    print("=" * 80)
    print("BM618: DERİN ÖĞRENME MODELLERİ (D1, D2, D3) TESTLERİ BAŞLADI")
    print("=" * 80)
    test_a_d3_encoder_parameters_exact_17088()
    test_b_d3_padding_invariance()
    test_c_d2_attention_weights_properties()
    test_d_no_nans_and_output_shapes()
    test_e_identical_regression_heads()
    print("=" * 80)
    print("[TÜM MODEL TESTLERİ EKSİKSİZ BAŞARIYLA GEÇTİ]")
    print("=" * 80)


if __name__ == '__main__':
    run_all()
