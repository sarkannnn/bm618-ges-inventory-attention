"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Çok Başlıklı Self-Attention Temel Tensör Uygulaması (src/attention.py)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

# Proje kök dizinini bağıl olarak belirleme (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class MultiHeadSelfAttention(nn.Module):
    """
    Çok Başlıklı Öz-Dikkat Mekanizması (Multi-Head Self-Attention).
    
    YÖNERGE VE KURAL GEREKSİNİMLERİ:
    ---------------------------------
    1. PyTorch'un hazır nn.MultiheadAttention modülü ana çözümde KESİNLİKLE KULLANILMAMIŞTIR.
    2. Projeksiyonlar için nn.Linear kullanılmış; başlıklara ayırma, ölçekli nokta çarpımı,
       maskeleme, softmax, dropout ve başlık birleştirme temel tensör işlemleriyle sıfırdan yazılmıştır.
    3. Tensör şekilleri her adımda açık yorum satırlarıyla belgelenmiştir.
    4. Maskelemede kararsızlığı önlemek için -inf yerine büyük negatif sayı (-1e9) kullanılmıştır.
    5. Tüm Anahtarların Maskeli Olduğu Durum (All-Keys-Masked / Empty Sequence):
       Bir sorgu konumu için tüm anahtarlar maskeli ise (örn. tamamen padding olan bir sekans),
       softmax işleminden sonra oluşabilecek NaN değerleri engellenir; bu satırların attention
       ağırlıkları doğrudan 0.0 yapılır ve bu konuma ait bağlam çıktısı sıfır tensörü (0.0) olarak üretilir.
    6. Attention haritalarının görselleştirilmesi ve hata analizleri için 'return_attention_weights' seçeneği sunulur.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) başlık sayısı num_heads ({num_heads}) ile tam bölünmelidir."
        )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / np.sqrt(self.d_k)
        
        # Q, K, V doğrusal izdüşümleri: (d_model -> d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Çıktı izdüşümü: (d_model -> d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Attention ağırlıkları üzerinde dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        İleri Geçiş (Forward Pass).
        
        Parametreler:
        -------------
        x: torch.Tensor
            Girdi tensörü. Şekil: (B, n, d)
            B: Mini-batch boyutu, n: Sekans uzunluğu (token sayısı), d: Model boyutu (d_model).
        key_padding_mask: Optional[torch.Tensor]
            Padding maskesi. Şekil: (B, n) - Boolean tensör.
            True olan konumlar padding kabul edilir ve maskelenir (dikkat edilmez).
        attn_mask: Optional[torch.Tensor]
            İki boyutlu veya yayınlanabilir (broadcastable) dikkat maskesi (örn. nedensel maske).
            Şekil: (n, n) veya (B, h, n, n) veya (1, 1, n, n).
            Boolean ise True olan yerler maskelenir; Float ise doğrudan skorlara eklenir.
        return_attention_weights: bool
            True ise (output, attention_weights) çifti döner; False ise yalnızca output döner.
            
        Dönüş:
        ------
        output: torch.Tensor
            Dikkat uygulanmış çıktı tensörü. Şekil: (B, n, d).
        attn_weights: torch.Tensor (Opsiyonel)
            Hesaplanan dikkat ağırlıkları matrisi. Şekil: (B, h, n, n).
        """
        # B: batch_size, n: seq_len, d: d_model
        B, n, d = x.shape
        
        # 1. Adım: Q, K, V Doğrusal İzdüşümleri
        # (B, n, d) -> (B, n, d)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. Adım: Başlıklara Ayırma ve Transpoz
        # (B, n, d) -> (B, n, h, d_k) -> (B, h, n, d_k)
        Q = Q.view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Adım: Ölçekli Noktasal Çarpım Skorları (Scaled Dot-Product)
        # Q: (B, h, n, d_k) @ K^T: (B, h, d_k, n) -> scores: (B, h, n, n)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 4. Adım: Maskeleme (Büyük negatif sayı: -1e9)
        large_neg = -1e9
        
        # Padding Anahtar Maskesi: (B, n) -> (B, 1, 1, n)
        all_keys_masked = None
        if key_padding_mask is not None:
            # key_padding_mask True olan pozisyonlar maskelenir
            pad_mask_expanded = key_padding_mask.view(B, 1, 1, n)
            scores = scores.masked_fill(pad_mask_expanded, large_neg)
            # Tüm anahtarları maskeli olan satırların tespiti: (B, 1, 1, 1)
            all_keys_masked = pad_mask_expanded.all(dim=-1, keepdim=True)
            
        # Ek Dikkat Maskesi (Örn: Nedensel Maske)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, large_neg)
            else:
                scores = scores + attn_mask
                
        # 5. Adım: Softmax (Son eksen olan anahtar boyutu üzerinde)
        # scores: (B, h, n, n) -> attn_weights: (B, h, n, n)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Tüm anahtarların maskeli olduğu durumlarda NaN yerine sıfır çıktısı garantisi:
        if all_keys_masked is not None and all_keys_masked.any():
            attn_weights = torch.where(all_keys_masked, torch.zeros_like(attn_weights), attn_weights)
            
        # Maskelenen pozisyonların ağırlığını kesin ve bit düzeyinde 0.0 yapma
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(pad_mask_expanded, 0.0)
            
        # 6. Adım: Dropout
        attn_weights_dropped = self.dropout_layer(attn_weights)
        
        # 7. Adım: Değer Tensörü (V) ile Ağırlıklı Toplam (Bağlam Hesabı)
        # attn_weights: (B, h, n, n) @ V: (B, h, n, d_k) -> context: (B, h, n, d_k)
        context = torch.matmul(attn_weights_dropped, V)
        
        # 8. Adım: Başlıkları Birleştirme
        # (B, h, n, d_k) -> (B, n, h, d_k) -> (B, n, d)
        context = context.transpose(1, 2).contiguous().view(B, n, d)
        
        # 9. Adım: Çıktı Projeksiyonu
        # (B, n, d) -> (B, n, d)
        output = self.out_proj(context)
        
        if return_attention_weights:
            return output, attn_weights
        return output
