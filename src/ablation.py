"""
BM618: Ablation deneyleri (src/ablation.py)

Aynı dış bölmeler; her kat için Adım 7'de SEÇİLMİŞ (lr, dropout, epoch) kullanılır (results/nn_selected.csv),
tek sabit tohum 42. Izgara, protokol ve ölçütler DEĞİŞMEZ; results/nn_* ve baselines_* dosyalarına yazılmaz.

Varyantlar:
  (a) D2_esit_agirlik : D2'de öğrenilen attention ağırlıkları yerine eşit ağırlık
  (b) D3_konumsuz     : D3'te öğrenilen konum embedding'i yok
  (c) D3_yilsiz       : D3'ten dosya yılı ([YIL] etiketi + yıl değeri) çıkarıldı
  + KONTROL: D2_kontrol, D3_kontrol = ablation ile BİREBİR aynı prosedürle (tohumlu ilklendirme) yeniden eğitilmiş
    referanslar. Adım 7'de model ilklendirmesi tohumdan önce yapıldığı için (bkz. rapor notu) Adım 7 tohum-42 koşusu
    ile ablation arasındaki fark ilklendirme gürültüsünü de içerir; kontrol bunu ayırır.

Bootstrap: tesis grupları üzerinde 1000 tekrar, eşleştirilmiş, tohum 42. Fark = (ablation − referans);
pozitif = ablation daha kötü. H4: D3_yilsiz − D3(ref, Adım 7 tohum 42) aralığı sıfırı dışlıyorsa H4 reddedilir.
"""

import os
import sys
import time
from typing import Dict
import numpy as np
import pandas as pd
import torch

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_prep import load_audited_data, prepare_tokenized_dataset, log1p_to_power
from src.models import ModelD2, ModelD3
from src.train_nn import train_fixed_epochs
from src.metrics import compute_all_metrics, paired_bootstrap_facility_mae_ci

ABLATION_SEED = 42


class ModelD2EqualWeights(ModelD2):
    """D2'de öğrenilen attention yerine eşit ağırlıklı (PAD hariç) ortalama havuzlama (ablation a)."""
    def forward(self, token_ids, padding_mask=None, return_attention_weights=False, equal_weights=True):
        return super().forward(token_ids, padding_mask=padding_mask,
                               return_attention_weights=return_attention_weights, equal_weights=True)


# variant -> (taban model, drop_year, model üretici)
VARIANTS = {
    "D2_kontrol":      ("D2", False, lambda v, p: ModelD2(vocab_size=v, d_model=32, dropout=p)),
    "D2_esit_agirlik": ("D2", False, lambda v, p: ModelD2EqualWeights(vocab_size=v, d_model=32, dropout=p)),
    "D3_kontrol":      ("D3", False, lambda v, p: ModelD3(vocab_size=v, max_len=64, d_model=32, dropout=p)),
    "D3_konumsuz":     ("D3", False, lambda v, p: ModelD3(vocab_size=v, max_len=64, d_model=32, dropout=p, use_pos_embedding=False)),
    "D3_yilsiz":       ("D3", True,  lambda v, p: ModelD3(vocab_size=v, max_len=64, d_model=32, dropout=p)),
}
# Başlıkta konumlar: konumsuzda tablo yok; yılsızda dizi 2 token kısa (max_len eğitimden öğrenilir)


def fit_variant(variant: str, outer_fold: int, df_main: pd.DataFrame, sel_row: pd.Series, seed: int = ABLATION_SEED):
    """
    Bir varyantı bir dış katta eğitir (seçilmiş lr/dropout/epoch, sabit epoch) ve dış testte TEK kez tahmin eder.
    İlklendirme tohumdan ÖNCE değil, SONRA yapılır (tohum ilklendirmeyi de kontrol eder).
    Döndürür: model, outer_train_data, outer_test_data, test_df, y_hat
    """
    base, drop_year, builder = VARIANTS[variant]
    tr_df = df_main[df_main["outer_test_fold"] != outer_fold]
    te_df = df_main[df_main["outer_test_fold"] == outer_fold]

    tr = prepare_tokenized_dataset(tr_df, fit_vocab=True, drop_year=drop_year)
    te = prepare_tokenized_dataset(te_df, vocab=tr["vocab"], max_len=tr["max_len"], fit_vocab=False, drop_year=drop_year)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = builder(len(tr["vocab"]), float(sel_row["dropout"]))
    train_fixed_epochs(model, tr, num_epochs=int(sel_row["epochs"]), lr=float(sel_row["lr"]),
                       weight_decay=1e-4, batch_size=16, seed=seed)
    model.eval()
    with torch.no_grad():
        z_hat = model(te["token_ids"], padding_mask=te["key_padding_mask"]).cpu().numpy()
    return model, tr, te, te_df, log1p_to_power(z_hat)


def run_ablation() -> Dict[str, pd.DataFrame]:
    t_start = time.time()
    results_dir = os.path.join(PROJECT_ROOT, "results")
    df_main, _, _ = load_audited_data()
    sel = pd.read_csv(os.path.join(results_dir, "nn_selected.csv"))
    ent = dict(zip(df_main["record_id"], df_main["entity_group_id"]))

    rows = []
    for variant, (base, _, _) in VARIANTS.items():
        for f in sorted(df_main["outer_test_fold"].unique()):
            sel_row = sel[(sel["model"] == base) & (sel["outer_fold"] == f)].iloc[0]
            _, _, _, te_df, y_hat = fit_variant(variant, int(f), df_main, sel_row)
            for (_, r), yp in zip(te_df.iterrows(), y_hat):
                rows.append({"record_id": r["record_id"], "outer_fold": int(f), "variant": variant,
                             "seed": ABLATION_SEED, "y_true": float(r["power_kwp"]), "y_pred": float(yp)})
        print(f"  [Ablation] {variant} tamamlandı")
    df_pred = pd.DataFrame(rows)
    df_pred.to_csv(os.path.join(results_dir, "ablation_predictions.csv"), index=False, encoding="utf-8")

    # Adım 7 referansları (tohum 42) — yeniden hesaplanmaz, nn_predictions.csv'den okunur
    nn = pd.read_csv(os.path.join(results_dir, "nn_predictions.csv"))
    refs = {"D2_ref_adim7": nn[(nn["model"] == "D2") & (nn["seed"] == 42)].copy(),
            "D3_ref_adim7": nn[(nn["model"] == "D3") & (nn["seed"] == 42)].copy()}
    all_sets = {k: v for k, v in refs.items()}
    for v in VARIANTS:
        all_sets[v] = df_pred[df_pred["variant"] == v].copy()

    # Metrik tablosu (birleştirilmiş + kat bazında tesis dengeli MAE)
    met_rows = []
    for name, d in all_sets.items():
        d = d.copy()
        d["g"] = d["record_id"].map(ent)
        m = compute_all_metrics(d["y_true"], d["y_pred"], d["g"])
        row = {"model": name, **{k: round(v, 3) for k, v in m.items()}}
        for f in range(5):
            s = d[d["outer_fold"] == f]
            row[f"kat{f}_tesis_mae"] = round(compute_all_metrics(s["y_true"], s["y_pred"], s["g"])["facility_balanced_mae"], 2)
        met_rows.append(row)
    df_met = pd.DataFrame(met_rows)
    df_met.to_csv(os.path.join(results_dir, "ablation_metrics.csv"), index=False, encoding="utf-8")

    # Eşleştirilmiş bootstrap: (ablation − referans)
    def boot(a_name: str, b_name: str):
        a = all_sets[a_name].set_index("record_id")
        b = all_sets[b_name].set_index("record_id").loc[a.index]
        g = a.index.map(ent)
        lo, hi, md = paired_bootstrap_facility_mae_ci(a["y_true"].to_numpy(), a["y_pred"].to_numpy(),
                                                       b["y_pred"].to_numpy(), np.asarray(g), n_boot=1000, seed=42)
        return md, lo, hi

    pairs = [("D2_esit_agirlik", "D2_ref_adim7"), ("D3_konumsuz", "D3_ref_adim7"), ("D3_yilsiz", "D3_ref_adim7"),
             ("D2_esit_agirlik", "D2_kontrol"), ("D3_konumsuz", "D3_kontrol"), ("D3_yilsiz", "D3_kontrol"),
             ("D2_kontrol", "D2_ref_adim7"), ("D3_kontrol", "D3_ref_adim7")]
    b_rows = []
    for a, b in pairs:
        md, lo, hi = boot(a, b)
        b_rows.append({"fark (A − B)": f"{a} − {b}", "tesis_dengeli_MAE_farki": round(md, 2),
                       "aralik_alt": round(lo, 2), "aralik_ust": round(hi, 2),
                       "sifiri_icerir": bool(lo <= 0.0 <= hi)})
    df_boot = pd.DataFrame(b_rows)
    df_boot.to_csv(os.path.join(results_dir, "ablation_bootstrap.csv"), index=False, encoding="utf-8")

    # H4 kararı (protokol): D3'ten yıl çıkarılınca fark aralığı sıfırı DIŞLIYORSA reddedilir
    h4 = df_boot[df_boot["fark (A − B)"] == "D3_yilsiz − D3_ref_adim7"].iloc[0]
    h4c = df_boot[df_boot["fark (A − B)"] == "D3_yilsiz − D3_kontrol"].iloc[0]

    def verdict(r):
        return ("REDDEDİLDİ (aralık sıfırı dışlıyor)" if not r["sifiri_icerir"]
                else "REDDEDİLMEDİ (aralık sıfırı içeriyor; fark ayırt edilemedi)")
    h4_text = (f"H4 (protokol karşılaştırması D3_yilsiz − D3 Adım-7 tohum 42): fark {h4['tesis_dengeli_MAE_farki']:+.2f} kWp, "
               f"%95 aralık [{h4['aralik_alt']:+.2f}, {h4['aralik_ust']:+.2f}] -> {verdict(h4)}. "
               f"[Kontrol: D3_yilsiz − D3_kontrol {h4c['tesis_dengeli_MAE_farki']:+.2f} kWp, "
               f"[{h4c['aralik_alt']:+.2f}, {h4c['aralik_ust']:+.2f}] -> {verdict(h4c)}]")

    print(f"Ablation süresi: {time.time() - t_start:.1f} sn")
    return {"metrics": df_met, "bootstrap": df_boot, "h4": h4_text}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    out = run_ablation()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nABLATION METRİKLERİ (tohum 42; tesis dengeli MAE kWp):")
    print(out["metrics"].to_string(index=False))
    print("\nEŞLEŞTİRİLMİŞ BOOTSTRAP (1000 tekrar, tohum 42; fark = A − B, pozitif = A daha kötü):")
    print(out["bootstrap"].to_string(index=False))
    print("\n" + out["h4"])
    print("NOT: Aralık yalnızca sabit bölmeler ve eğitim prosedürü koşulundaki belirsizliği gösterir; örneklem küçüktür (76 tesis).")
