"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Çok Başlıklı Self-Attention Doğrulama, M1 ve M2 Test Paketi (tests/test_attention.py)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.attention import MultiHeadSelfAttention

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')


def test_1_padding_invariance():
    """
    Test 1: Padding eklendiğinde geçerli tokenların çıktısı değişmemelidir.
    """
    torch.manual_seed(42)
    d_model = 16
    num_heads = 2
    
    m = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    m.eval()
    
    # 3 tokenlık gerçek dizi
    x_real = torch.randn(2, 3, d_model)
    out_real = m(x_real)
    
    # Sonuna 2 padding tokenı eklenmiş 5 tokenlık dizi
    x_pad = torch.cat([x_real, torch.randn(2, 2, d_model)], dim=1)
    mask = torch.tensor([
        [False, False, False, True, True],
        [False, False, False, True, True]
    ])
    out_padded = m(x_pad, key_padding_mask=mask)
    
    # İlk 3 tokenın çıktıları maskeli durumda da birebir aynı kalmalıdır
    diff = torch.abs(out_real - out_padded[:, :3, :]).max().item()
    assert diff < 1e-6, f"Padding invariance başarısız! Maksimum fark: {diff}"
    print(f"Test 1 (Padding Invariance) BAŞARILI! (Maksimum fark: {diff:.2e} < 1e-6)")


def test_2_row_sums_equal_one():
    """
    Test 2: Dropout kapalıyken (eval modu) attention satır toplamları geçerli konumlarda tam 1'dir.
    """
    torch.manual_seed(42)
    d_model = 32
    num_heads = 4
    
    m = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    m.eval()
    
    x = torch.randn(3, 6, d_model)
    mask = torch.tensor([
        [False, False, False, False, True, True],
        [False, False, False, True, True, True],
        [False, False, False, False, False, False]
    ])
    
    _, weights = m(x, key_padding_mask=mask, return_attention_weights=True)
    # weights: (B, h, n, n)
    row_sums = weights.sum(dim=-1) # (B, h, n)
    
    # Maskeli olmayan satırlar için satır toplamı 1.0 olmalıdır
    diff = torch.abs(row_sums - 1.0).max().item()
    assert diff < 1e-6, f"Satır toplamı 1 kontrolü başarısız! Maksimum sapma: {diff}"
    print(f"Test 2 (Satır Toplamları = 1) BAŞARILI! (Maksimum sapma: {diff:.2e} < 1e-6)")


def test_3_masked_weights_strictly_zero():
    """
    Test 3: Maskelenen pozisyonlara ait attention ağırlıkları kesin olarak sıfırdır (0.0).
    Ayrıca tüm anahtarların maskeli olduğu uç durumda NaN oluşmaz, sıfır çıktısı döner.
    """
    torch.manual_seed(42)
    d_model = 16
    num_heads = 2
    
    m = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    m.eval()
    
    x = torch.randn(2, 4, d_model)
    # 2. ve 3. pozisyonlar padding
    mask = torch.tensor([
        [False, False, True, True],
        [False, True, True, True]
    ])
    
    out, weights = m(x, key_padding_mask=mask, return_attention_weights=True)
    # weights: (B, h, n, n)
    # mask True olan yerlerde ağırlıklar kesin 0.0 olmalı
    masked_weights_1 = weights[0, :, :, 2:]
    masked_weights_2 = weights[1, :, :, 1:]
    
    assert (masked_weights_1 == 0.0).all(), "Maskelenen ağırlıklar 0.0 değil!"
    assert (masked_weights_2 == 0.0).all(), "Maskelenen ağırlıklar 0.0 değil!"
    
    # Uç durum: Tüm anahtarları maskeli örnek (All-keys-masked)
    x_all_masked = torch.randn(1, 3, d_model)
    all_mask = torch.tensor([[True, True, True]])
    out_zero, weights_zero = m(x_all_masked, key_padding_mask=all_mask, return_attention_weights=True)
    
    assert not torch.isnan(out_zero).any(), "Tüm anahtarlar maskeliyken çıktıda NaN oluştu!"
    assert not torch.isnan(weights_zero).any(), "Tüm anahtarlar maskeliyken ağırlıklarda NaN oluştu!"
    assert (weights_zero == 0.0).all(), "Tüm anahtarlar maskeliyken ağırlıklar 0.0 olmalıdır!"
    
    print("Test 3 (Maske Dışı Ağırlıklar Tam Sıfır ve NaN Koruması) BAŞARILI!")


def test_4_m2_numerical_verification():
    """
    Test 4: M2 Görevi Elle Hesap ve Sayısal Doğrulama Paketi
    X = [[1, 0], [0, 1], [1, 1]], W_Q = W_K = W_V = W_O = I_{2x2} (bias=False, float64).
    (4a) Maskesiz skor S, A ve O dört ondalıkla uyuşuyor.
    (4b) Son token padding iken anahtar maskesi dört ondalıkla uyuşuyor.
    (4c) Üst üçgeni -inf olan nedensel maske (causal mask) dört ondalıkla uyuşuyor.
    (4d) Analitik gradyan float64 sonlu farkla (eps=1e-5) uyuşuyor.
    (4e) PyTorch hazır nn.MultiheadAttention ile birebir çıktı eşleşmesi.
    """
    # Girdi: (B=1, n=3, d=2)
    X = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float64)
    
    m = MultiHeadSelfAttention(d_model=2, num_heads=1, dropout=0.0, bias=False)
    m.to(torch.float64)
    m.eval()
    
    # Ağırlıkları Birim Matrise (Identity) eşitleme
    with torch.no_grad():
        m.q_proj.weight.copy_(torch.eye(2, dtype=torch.float64))
        m.k_proj.weight.copy_(torch.eye(2, dtype=torch.float64))
        m.v_proj.weight.copy_(torch.eye(2, dtype=torch.float64))
        m.out_proj.weight.copy_(torch.eye(2, dtype=torch.float64))
        
    # -------------------------------------------------------------------------
    # 4a. Maskesiz Attention (Unmasked)
    # -------------------------------------------------------------------------
    out_unmasked, weights_unmasked = m(X, return_attention_weights=True)
    A_unmasked_np = weights_unmasked.squeeze().detach().numpy()
    O_unmasked_np = out_unmasked.squeeze().detach().numpy()
    
    expected_A_unmasked = np.array([
        [0.4011, 0.1978, 0.4011],
        [0.1978, 0.4011, 0.4011],
        [0.2483, 0.2483, 0.5035]
    ])
    expected_O_unmasked = np.array([
        [0.8022, 0.5989],
        [0.5989, 0.8022],
        [0.7517, 0.7517]
    ])
    
    np.testing.assert_almost_equal(A_unmasked_np, expected_A_unmasked, decimal=4)
    np.testing.assert_almost_equal(O_unmasked_np, expected_O_unmasked, decimal=4)
    print("  4a (M2 Maskesiz Doğrulama): 4 ondalıkla tam uyumlu! GEÇTİ.")

    # -------------------------------------------------------------------------
    # 4b. Anahtar Padding Maskesi (Son token padding)
    # -------------------------------------------------------------------------
    key_mask = torch.tensor([[False, False, True]])
    out_pad, weights_pad = m(X, key_padding_mask=key_mask, return_attention_weights=True)
    A_pad_np = weights_pad.squeeze().detach().numpy()
    O_pad_np = out_pad.squeeze().detach().numpy()
    
    expected_A_pad = np.array([
        [0.6698, 0.3302, 0.0000],
        [0.3302, 0.6698, 0.0000],
        [0.5000, 0.5000, 0.0000]
    ])
    expected_O_pad = np.array([
        [0.6698, 0.3302],
        [0.3302, 0.6698],
        [0.5000, 0.5000]
    ])
    
    np.testing.assert_almost_equal(A_pad_np, expected_A_pad, decimal=4)
    np.testing.assert_almost_equal(O_pad_np, expected_O_pad, decimal=4)
    print("  4b (M2 Son Token Padding Maskesi): 4 ondalıkla tam uyumlu! GEÇTİ.")

    # -------------------------------------------------------------------------
    # 4c. Nedensel Maske (Causal Mask - Üst üçgen maskeli)
    # -------------------------------------------------------------------------
    causal_mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    out_causal, weights_causal = m(X, attn_mask=causal_mask, return_attention_weights=True)
    A_causal_np = weights_causal.squeeze().detach().numpy()
    O_causal_np = out_causal.squeeze().detach().numpy()
    
    expected_A_causal = np.array([
        [1.0000, 0.0000, 0.0000],
        [0.3302, 0.6698, 0.0000],
        [0.2483, 0.2483, 0.5035]
    ])
    expected_O_causal = np.array([
        [1.0000, 0.0000],
        [0.3302, 0.6698],
        [0.7517, 0.7517]
    ])
    
    np.testing.assert_almost_equal(A_causal_np, expected_A_causal, decimal=4)
    np.testing.assert_almost_equal(O_causal_np, expected_O_causal, decimal=4)
    print("  4c (M2 Nedensel Maske): 4 ondalıkla tam uyumlu! GEÇTİ.")

    # -------------------------------------------------------------------------
    # 4d. Analitik Gradyan ve Float64 Sonlu Fark Doğrulaması (eps = 1e-5)
    # -------------------------------------------------------------------------
    scale = np.sqrt(2.0)
    K_const = X.squeeze(0).clone()
    V_const = X.squeeze(0).clone()
    
    def loss_fn(Q_tensor):
        scores = (Q_tensor @ K_const.T) / scale
        A = torch.softmax(scores, dim=-1)
        O = A @ V_const
        return O.sum()
        
    Q_var = X.squeeze(0).clone().requires_grad_(True)
    loss = loss_fn(Q_var)
    loss.backward()
    analytical_grad = Q_var.grad.clone()
    
    # İki yönlü sonlu fark hesabı: (f(x+eps) - f(x-eps)) / (2*eps)
    eps = 1e-5
    numerical_grad = torch.zeros_like(Q_var)
    for i in range(Q_var.shape[0]):
        for j in range(Q_var.shape[1]):
            Q_plus = X.squeeze(0).clone()
            Q_plus[i, j] += eps
            Q_minus = X.squeeze(0).clone()
            Q_minus[i, j] -= eps
            numerical_grad[i, j] = (loss_fn(Q_plus) - loss_fn(Q_minus)) / (2.0 * eps)
            
    grad_diff = torch.abs(analytical_grad - numerical_grad).max().item()
    assert grad_diff < 1e-7, f"Gradyan doğrulaması başarısız! Fark: {grad_diff}"
    print(f"  4d (M2 Sonlu Fark Gradyan Kontrolü): eps=1e-5 ile fark: {grad_diff:.2e} < 1e-7! GEÇTİ.")

    # -------------------------------------------------------------------------
    # 4e. PyTorch Hazır nn.MultiheadAttention Modülü ile Çıktı Karşılaştırması
    # -------------------------------------------------------------------------
    d_test = 16
    h_test = 2
    m_test_custom = MultiHeadSelfAttention(d_model=d_test, num_heads=h_test, bias=True)
    m_test_torch = nn.MultiheadAttention(embed_dim=d_test, num_heads=h_test, bias=True, batch_first=True)
    
    with torch.no_grad():
        in_proj_w = torch.cat([m_test_custom.q_proj.weight, m_test_custom.k_proj.weight, m_test_custom.v_proj.weight], dim=0)
        in_proj_b = torch.cat([m_test_custom.q_proj.bias, m_test_custom.k_proj.bias, m_test_custom.v_proj.bias], dim=0)
        m_test_torch.in_proj_weight.copy_(in_proj_w)
        m_test_torch.in_proj_bias.copy_(in_proj_b)
        m_test_torch.out_proj.weight.copy_(m_test_custom.out_proj.weight)
        m_test_torch.out_proj.bias.copy_(m_test_custom.out_proj.bias)
        
    m_test_custom.eval()
    m_test_torch.eval()
    
    x_test = torch.randn(2, 4, d_test)
    out_custom = m_test_custom(x_test)
    out_torch, _ = m_test_torch(x_test, x_test, x_test, need_weights=False)
    
    torch_diff = torch.abs(out_custom - out_torch).max().item()
    assert torch_diff < 1e-6, f"Hazır MultiheadAttention ile fark yüksek: {torch_diff}"
    print(f"  4e (Hazır nn.MultiheadAttention Karşılaştırması): Fark: {torch_diff:.2e} < 1e-6! GEÇTİ.")
    print("Test 4 (M2 Sayısal Doğrulama Paketi) EKSİKSİZ BAŞARILI!")


def test_5_m1_matrix_chain_gradients():
    """
    Test 5: M1 Matris Zinciri Gradyan Doğrulaması.
    Formüller:
      dA = dO @ V.T
      dV = A.T @ dO
      dS = A * (dA - (dA * A) @ 1)  (Softmax backward formülü)
      dQ = (dS @ K) / sqrt(d_k)
      dK = (dS.T @ Q) / sqrt(d_k)
    Elle kodlanan bu matris türevleri PyTorch autograd ile float64'te 1e-10 toleransla karşılaştırılır.
    """
    torch.manual_seed(42)
    n = 4
    d_k = 3
    scale = np.sqrt(d_k)

    # float64 tensörler
    Q = torch.randn(n, d_k, dtype=torch.float64, requires_grad=True)
    K = torch.randn(n, d_k, dtype=torch.float64, requires_grad=True)
    V = torch.randn(n, d_k, dtype=torch.float64, requires_grad=True)

    # İleri geçiş
    S = (Q @ K.T) / scale
    A = torch.softmax(S, dim=-1)
    O = A @ V

    # Keyfi bir üst akış gradyan matrisi dO
    dO = torch.randn(n, d_k, dtype=torch.float64)

    # Autograd ile analitik gradyan hesabı: Loss = sum(O * dO)
    loss = (O * dO).sum()
    loss.backward()

    # M1 Matris Zinciri ile Elle Gradyan Hesabı:
    # 1. dA = dO @ V^T
    dA = dO @ V.T

    # 2. dV = A^T @ dO
    dV = A.T @ dO

    # 3. dS = A * (dA - (dA * A) 1)
    # (dA * A) 1 ifadesi her satırdaki elemanların toplamını hesaplayıp sütunlara yayar
    row_sum = (dA * A).sum(dim=-1, keepdim=True)
    dS = A * (dA - row_sum)

    # 4. dQ = dS @ K / sqrt(d_k)
    dQ = (dS @ K) / scale

    # 5. dK = dS^T @ Q / sqrt(d_k)
    dK = (dS.T @ Q) / scale

    # Karşılaştırma (1e-10 tolerans)
    diff_Q = torch.abs(Q.grad - dQ).max().item()
    diff_K = torch.abs(K.grad - dK).max().item()
    diff_V = torch.abs(V.grad - dV).max().item()

    assert diff_Q < 1e-10, f"dQ autograd farkı yüksek: {diff_Q}"
    assert diff_K < 1e-10, f"dK autograd farkı yüksek: {diff_K}"
    assert diff_V < 1e-10, f"dV autograd farkı yüksek: {diff_V}"

    print(f"Test 5 (M1 Matris Zinciri: dQ={diff_Q:.2e}, dK={diff_K:.2e}, dV={diff_V:.2e} < 1e-10) BAŞARILI!")


def run_all_tests():
    print("=" * 80)
    print("BM618: ATTENTION MODÜLÜ, M1 VE M2 SAYISAL DOĞRULAMA TESTLERİ BAŞLADI")
    print("=" * 80)
    test_1_padding_invariance()
    test_2_row_sums_equal_one()
    test_3_masked_weights_strictly_zero()
    test_4_m2_numerical_verification()
    test_5_m1_matrix_chain_gradients()
    print("=" * 80)
    print("[TÜM TESTLER EKSİKSİZ BAŞARIYLA GEÇTİ]")
    print("=" * 80)


if __name__ == '__main__':
    run_all_tests()
