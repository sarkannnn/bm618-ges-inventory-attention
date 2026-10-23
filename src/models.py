"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Derin Öğrenme Modelleri Modülü (src/models.py)

MODELLER:
- D1: Maskeli Ortalama Havuzlama (Masked Mean Pooling) -> Ortak Regresyon Başlığı
- D2: Bahdanau Attention Havuzlama (Additive Attention Pooling) -> Ortak Regresyon Başlığı
  e_i = vᵀ tanh(W h_i + b), W ∈ R^{32×32}, v ∈ R^{32}. Ağırlık ve eşit ağırlık (ablation 1) seçenekleri.
- D3: Pre-LN Transformer Encoder (d=32, h=4, d_ff=64, 2 blok) -> [CLS] -> Ortak Regresyon Başlığı
  Öğrenilen konum embedding'i açma/kapama (ablation 2) ve dosya yılı çıkarma (ablation 3) seçenekleri.
  PyTorch nn.MultiheadAttention YOKTUR; src/attention.py'deki temel tensör modülü kullanılır.
- Ortak Regresyon Başlığı: Linear(32,32) -> ReLU -> Dropout -> Linear(32,1)
"""

import os
import sys
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.attention import MultiHeadSelfAttention
from src.data_prep import PAD_IDX


class SharedRegressionHead(nn.Module):
    """
    D1, D2 ve D3 için dondurulmuş ortak regresyon başlığı:
    Linear(32, 32) -> ReLU -> Dropout(p) -> Linear(32, 1)
    """
    def __init__(self, d_in: int = 32, d_hidden: int = 32, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_hidden, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32) -> (B, 1) -> (B,)
        h = self.act(self.fc1(x))
        h = self.drop(h)
        out = self.fc2(h).squeeze(-1)
        return out


class ModelD1(nn.Module):
    """
    D1: Maskeli Ortalama Havuzlama Modeli.
    Embedding(vocab, 32, padding_idx=PAD) -> PAD hariç ortalama -> Ortak Başlık.
    Konum embedding'i KESİNLİKLE YOKTUR.
    """
    def __init__(self, vocab_size: int, d_model: int = 32, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.head = SharedRegressionHead(d_in=d_model, d_hidden=d_model, dropout=dropout)

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # token_ids: (B, n)
        # padding_mask: (B, n), True = PAD
        emb = self.embedding(token_ids)  # (B, n, 32)

        if padding_mask is not None:
            mask_valid = (~padding_mask).unsqueeze(-1).to(emb.dtype)  # (B, n, 1)
            sum_emb = torch.sum(emb * mask_valid, dim=1)  # (B, 32)
            valid_counts = torch.clamp(mask_valid.sum(dim=1), min=1.0)  # (B, 1)
            pooled = sum_emb / valid_counts  # (B, 32)
        else:
            pooled = torch.mean(emb, dim=1)  # (B, 32)

        out = self.head(pooled)  # (B,)
        return out


class ModelD2(nn.Module):
    """
    D2: Bahdanau Attention Havuzlama Modeli.
    e_i = vᵀ tanh(W h_i + b), W ∈ R^{32×32}, b ∈ R^{32}, v ∈ R^{32}.
    PAD konumları softmax öncesi -1e9 ile maskelenir.
    r = Σ α_i h_i -> Ortak Başlık.
    Konum embedding'i KESİNLİKLE YOKTUR.
    """
    def __init__(self, vocab_size: int, d_model: int = 32, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)

        # Bahdanau Attention Pooling Parametreleri: W (32x32) + b (32) + v (32)
        self.W_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, 1, bias=False)

        self.head = SharedRegressionHead(d_in=d_model, d_hidden=d_model, dropout=dropout)

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        equal_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # token_ids: (B, n)
        # padding_mask: (B, n), True = PAD
        h = self.embedding(token_ids)  # (B, n, 32)

        if equal_weights:
            # Ablation 1: Eşit ağırlıklandırma (uniform pooling)
            if padding_mask is not None:
                mask_valid = (~padding_mask).to(h.dtype)  # (B, n)
                valid_counts = torch.clamp(mask_valid.sum(dim=-1, keepdim=True), min=1.0)
                alpha = mask_valid / valid_counts  # (B, n)
            else:
                n = h.shape[1]
                alpha = torch.full((h.shape[0], n), 1.0 / n, device=h.device, dtype=h.dtype)
        else:
            # e_i = vᵀ tanh(W h_i + b)
            proj = torch.tanh(self.W_proj(h))  # (B, n, 32)
            e = self.v_proj(proj).squeeze(-1)  # (B, n)

            if padding_mask is not None:
                # Maskelenmiş pozisyonlara -1e9
                e = e.masked_fill(padding_mask, -1e9)
                # Tüm satır maskeli ise NaN koruması
                all_masked = padding_mask.all(dim=-1, keepdim=True)
                alpha = torch.softmax(e, dim=-1)  # (B, n)
                # Tamamen maskeli satırları sıfıra çek
                alpha = alpha.masked_fill(all_masked, 0.0)
                # PAD olan pozisyonları kesin 0 yap
                alpha = alpha.masked_fill(padding_mask, 0.0)
            else:
                alpha = torch.softmax(e, dim=-1)

        # r = Σ α_i h_i
        r = torch.sum(alpha.unsqueeze(-1) * h, dim=1)  # (B, 32)

        out = self.head(r)  # (B,)

        if return_attention_weights:
            return out, alpha
        return out


class PreLNEncoderBlock(nn.Module):
    """
    Pre-LN Transformer Encoder Bloğu:
    x' = x + Attn(LN1(x))
    y  = x' + FFN(LN2(x'))
    d=32, h=4, d_ff=64.
    """
    def __init__(self, d_model: int = 32, num_heads: int = 4, d_ff: int = 64, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout, bias=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True)
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, n, 32)
        norm_x = self.ln1(x)
        if return_attention_weights:
            attn_out, weights = self.attn(
                norm_x,
                key_padding_mask=key_padding_mask,
                return_attention_weights=True
            )
        else:
            attn_out = self.attn(norm_x, key_padding_mask=key_padding_mask)
            weights = None

        x = x + self.drop(attn_out)

        norm_x2 = self.ln2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.drop(ffn_out)

        if return_attention_weights:
            return x, weights
        return x


class ModelD3(nn.Module):
    """
    D3: İki Bloklu Pre-LN Transformer Encoder Modeli.
    - [CLS] konum 0
    - Öğrenilen konum embedding'i (isteğe bağlı açılıp kapanabilir, ablation 2)
    - İki Pre-LN encoder bloğu (d=32, h=4, d_ff=64)
    - Son LayerNorm
    - [CLS] çıktısı -> Ortak Regresyon Başlığı
    - Dosya yılı çıkarma seçeneği (ablation 3)
    """
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 64,
        d_model: int = 32,
        num_heads: int = 4,
        d_ff: int = 64,
        dropout: float = 0.0,
        use_pos_embedding: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.use_pos_embedding = use_pos_embedding

        # Ortak token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)

        # Öğrenilen konum embedding'i (D3'e özel, ablation 2'de kapatılabilir)
        self.pos_embedding = nn.Embedding(max_len, d_model) if use_pos_embedding else None

        # İki Pre-LN encoder bloğu (toplam tam 17.088 parametre)
        self.block1 = PreLNEncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
        self.block2 = PreLNEncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)

        # Son LayerNorm
        self.final_ln = nn.LayerNorm(d_model)

        # Ortak regresyon başlığı
        self.head = SharedRegressionHead(d_in=d_model, d_hidden=d_model, dropout=dropout)

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        # token_ids: (B, n)
        # padding_mask: (B, n), True = PAD
        B, n = token_ids.shape
        x = self.token_embedding(token_ids)  # (B, n, 32)

        if self.use_pos_embedding and self.pos_embedding is not None:
            # Konum indeksleri: 0 .. n-1
            positions = torch.arange(n, device=token_ids.device).unsqueeze(0).expand(B, n)
            x = x + self.pos_embedding(positions)

        # Blok 1
        if return_attention_weights:
            x, w1 = self.block1(x, key_padding_mask=padding_mask, return_attention_weights=True)
            x, w2 = self.block2(x, key_padding_mask=padding_mask, return_attention_weights=True)
        else:
            x = self.block1(x, key_padding_mask=padding_mask)
            x = self.block2(x, key_padding_mask=padding_mask)
            w1, w2 = None, None

        # Son LayerNorm
        x = self.final_ln(x)  # (B, n, 32)

        # [CLS] çıktısı (indeks 0)
        cls_rep = x[:, 0, :]  # (B, 32)

        out = self.head(cls_rep)  # (B,)

        if return_attention_weights:
            return out, (w1, w2)
        return out


def count_parameters_detailed(model: nn.Module, effective_len: Optional[int] = None) -> Dict[str, int]:
    """
    Modelin parametre sayısını bileşen bileşen sayar ve döndürür:
    (embedding, konum, encoder blokları, son LayerNorm, başlık, toplam ve etkin toplam).
    """
    counts = {}

    if isinstance(model, ModelD1):
        counts["embedding"] = sum(p.numel() for p in model.embedding.parameters())
        counts["pooling"] = 0
        counts["regression_head"] = sum(p.numel() for p in model.head.parameters())
        counts["total"] = sum(p.numel() for p in model.parameters())
        counts["effective_total"] = counts["total"]

    elif isinstance(model, ModelD2):
        counts["embedding"] = sum(p.numel() for p in model.embedding.parameters())
        counts["pooling"] = sum(p.numel() for p in model.W_proj.parameters()) + sum(p.numel() for p in model.v_proj.parameters())
        counts["regression_head"] = sum(p.numel() for p in model.head.parameters())
        counts["total"] = sum(p.numel() for p in model.parameters())
        counts["effective_total"] = counts["total"]

    elif isinstance(model, ModelD3):
        counts["embedding"] = sum(p.numel() for p in model.token_embedding.parameters())
        counts["pos_embedding"] = sum(p.numel() for p in model.pos_embedding.parameters()) if model.pos_embedding is not None else 0
        if effective_len is not None and model.pos_embedding is not None:
            eff_len = min(effective_len, model.max_len)
            counts["effective_pos_embedding"] = eff_len * model.d_model
        else:
            counts["effective_pos_embedding"] = counts["pos_embedding"]

        b1_params = sum(p.numel() for p in model.block1.parameters())
        b2_params = sum(p.numel() for p in model.block2.parameters())
        counts["encoder_block1"] = b1_params
        counts["encoder_block2"] = b2_params
        counts["encoder_blocks_total"] = b1_params + b2_params
        counts["final_layernorm"] = sum(p.numel() for p in model.final_ln.parameters())
        counts["regression_head"] = sum(p.numel() for p in model.head.parameters())
        counts["total"] = sum(p.numel() for p in model.parameters())
        counts["effective_total"] = (
            counts["embedding"] + counts["effective_pos_embedding"] +
            counts["encoder_blocks_total"] + counts["final_layernorm"] + counts["regression_head"]
        )
    else:
        counts["total"] = sum(p.numel() for p in model.parameters())
        counts["effective_total"] = counts["total"]

    return counts
