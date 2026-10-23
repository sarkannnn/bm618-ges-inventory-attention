"""
BM618: Tanı analizleri (src/diagnostics.py)

(2) SIZINTI TANISI (H2): B1'in rastgele satır bölmesiyle (5 kat, tohum 42) çalıştırılması. Sonuç "SIZINTI TANISI,
    geçerli skor değil"dir; yalnızca gruplu bölmeyle farkı ve ortak tesis oranını göstermek içindir.
(3) HATA ANALİZİ: B1, D2, D3 için tesis dengeli MAE'ye en çok katkı veren kayıtlar; en büyük 10 tesis kaydında sistematik
    düşük tahmin kontrolü.
(4) ATTENTION YORUMU: D2 ağırlıkları ve D3 [CLS] attention'ı ile token silme (token ÇIKARILIR, PAD ile değiştirilmez)
    sıra korelasyonu. Yüksek attention nedensel önem OLARAK YORUMLANMAZ.

Hiçbir seçim yapılmaz; results/baselines_* ve results/nn_* dosyaları yalnız okunur.
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

# Proje kökünü bağıl olarak bulma (mutlak yol kullanılmaz)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_prep import (load_audited_data, prepare_tokenized_dataset, extract_features,
                           power_to_log1p, log1p_to_power, PAD_IDX, UNK_IDX)
from src.baselines import build_b1_pipeline, CANDIDATE_ALPHAS
from src.metrics import compute_facility_balanced_mae, compute_record_mae
from src.models import ModelD2, ModelD3
from src.ablation import fit_variant

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LEAK_LABEL = "SIZINTI TANISI, geçerli skor değil"


# ----------------------------------------------------------------------------------------------
# (2) SIZINTI TANISI
# ----------------------------------------------------------------------------------------------
def make_random_splits(df_main: pd.DataFrame, seed: int = 42, n_splits: int = 5) -> np.ndarray:
    """Grup bilgisini YOK sayan rastgele satır bölmesi (5 kat, tohum 42). Yalnız in_main_regression=True kayıtlar."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df_main), -1, dtype=int)
    for k, (_, te) in enumerate(kf.split(np.arange(len(df_main)))):
        fold[te] = k
    return fold


def shared_facility_ratio(groups: np.ndarray, fold_ids: np.ndarray) -> Tuple[float, List[float]]:
    """Ortak tesis oranı: test kayıtlarının, eğitimde de bulunan bir gruba ait olma oranı (0-1)."""
    groups = np.asarray(groups)
    shared_total, per_fold = 0, []
    for k in sorted(set(fold_ids.tolist())):
        te = fold_ids == k
        train_groups = set(groups[~te].tolist())
        s = int(sum(g in train_groups for g in groups[te].tolist()))
        per_fold.append(s / int(te.sum()))
        shared_total += s
    return shared_total / len(groups), per_fold


def run_b1_on_split(df_main: pd.DataFrame, fold_ids: np.ndarray, inner_seed: int = 42) -> pd.DataFrame:
    """
    B1 (TF-IDF + one-hot + Ridge) prosedürünün aynısı; yalnız bölme rastgele satır bölmesi.
    Alpha seçimi: eğitim içinde 3 katlı RASTGELE iç bölme, ölçüt tesis dengeli MAE.
    """
    df = df_main.reset_index(drop=True)
    out = []
    for k in sorted(set(fold_ids.tolist())):
        tr = df[fold_ids != k].reset_index(drop=True)
        te = df[fold_ids == k]
        inner = KFold(n_splits=3, shuffle=True, random_state=inner_seed)
        scores = {}
        for a in CANDIDATE_ALPHAS:
            maes = []
            for itr, iva in inner.split(np.arange(len(tr))):
                p = build_b1_pipeline(alpha=a)
                p.fit(extract_features(tr.iloc[itr]), power_to_log1p(tr.iloc[itr]["power_kwp"]))
                yh = log1p_to_power(p.predict(extract_features(tr.iloc[iva])))
                maes.append(compute_facility_balanced_mae(tr.iloc[iva]["power_kwp"], yh, tr.iloc[iva]["entity_group_id"]))
            scores[a] = float(np.mean(maes))
        best = min(scores, key=scores.get)
        p = build_b1_pipeline(alpha=best)
        p.fit(extract_features(tr), power_to_log1p(tr["power_kwp"]))
        yh = log1p_to_power(p.predict(extract_features(te)))
        for (_, r), y in zip(te.iterrows(), yh):
            out.append({"record_id": r["record_id"], "outer_fold": int(k), "alpha": best,
                        "y_true": float(r["power_kwp"]), "y_pred": float(y), "entity_group_id": r["entity_group_id"]})
    return pd.DataFrame(out)


def leakage_diagnostic() -> Dict:
    df_main, _, _ = load_audited_data()
    df_main = df_main.reset_index(drop=True)
    fold_rand = make_random_splits(df_main)
    ratio_rand, per_fold_rand = shared_facility_ratio(df_main["entity_group_id"].to_numpy(), fold_rand)
    ratio_grp, per_fold_grp = shared_facility_ratio(df_main["entity_group_id"].to_numpy(),
                                                    df_main["outer_test_fold"].to_numpy())

    rand = run_b1_on_split(df_main, fold_rand)
    rand["etiket"] = LEAK_LABEL
    rand.to_csv(os.path.join(RESULTS_DIR, "leakage_predictions.csv"), index=False, encoding="utf-8")

    grp = pd.read_csv(os.path.join(RESULTS_DIR, "baselines_predictions.csv"))
    grp = grp[grp["model"] == "B1"].copy()
    grp["entity_group_id"] = grp["record_id"].map(dict(zip(df_main["record_id"], df_main["entity_group_id"])))

    def mets(d):
        return (compute_facility_balanced_mae(d["y_true"], d["y_pred"], d["entity_group_id"]),
                compute_record_mae(d["y_true"], d["y_pred"]))
    mg, rg = mets(grp)
    mr, rr = mets(rand)
    table = pd.DataFrame([
        {"bölme": "gruplu GroupKFold (GEÇERLİ)", "tesis_dengeli_MAE": round(mg, 2), "kayıt_MAE": round(rg, 2),
         "ortak_tesis_orani": round(ratio_grp, 3), "etiket": "geçerli skor"},
        {"bölme": "rastgele satır bölmesi", "tesis_dengeli_MAE": round(mr, 2), "kayıt_MAE": round(rr, 2),
         "ortak_tesis_orani": round(ratio_rand, 3), "etiket": LEAK_LABEL},
    ])
    table.to_csv(os.path.join(RESULTS_DIR, "leakage_diagnostic.csv"), index=False, encoding="utf-8")
    h2_rejected = not (mr < mg)   # kural: rastgele MAE gruplu MAE'den düşük değilse H2 reddedilir
    return {"table": table, "diff": mg - mr, "ratio_rand": ratio_rand, "per_fold_rand": per_fold_rand,
            "ratio_grp": ratio_grp, "h2_rejected": h2_rejected, "mg": mg, "mr": mr}


# ----------------------------------------------------------------------------------------------
# (3) HATA ANALİZİ
# ----------------------------------------------------------------------------------------------
def load_record_level_preds(df_main: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Model -> kayıt düzeyi: y_true, tohum ortalamalı tahmin, tohum ortalamalı mutlak hata."""
    b = pd.read_csv(os.path.join(RESULTS_DIR, "baselines_predictions.csv"))
    nn = pd.read_csv(os.path.join(RESULTS_DIR, "nn_predictions.csv"))
    out = {}
    for name, d in [("B1", b[b["model"] == "B1"]), ("D2", nn[nn["model"] == "D2"]), ("D3", nn[nn["model"] == "D3"])]:
        d = d.copy()
        d["abs_err"] = (d["y_true"] - d["y_pred"]).abs()
        g = d.groupby("record_id").agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), abs_err=("abs_err", "mean"))
        out[name] = g
    return out


def probable_cause(true: float, err: float, group_true: np.ndarray, unk_words: List[str]) -> str:
    notes = []
    if len(group_true) >= 2 and group_true.max() / max(group_true.min(), 1e-9) >= 2.0:
        notes.append("aynı tesisin diğer yıl/etap kayıtlarında güç çok farklı (ad bu farkı taşımıyor)")
    if unk_words:
        notes.append(f"ad/ilçe sözcüğü eğitimde yok [UNK]: {', '.join(unk_words[:3])}")
    if true >= 300 and err < 0:
        notes.append("büyük kapasite: log-hedef ve ters dönüşümle düşük tahmin, eğitimde az örnek")
    if true <= 30 and err > 0:
        notes.append("küçük tesis aşırı tahmin: ortalamaya çekilme")
    if not notes:
        notes.append("ad/ilçe bilgisi gücü ayırt etmiyor (ortalamaya çekilme)")
    return "; ".join(notes) + " [sezgisel, nedensellik kanıtı değil]"


def error_analysis() -> Dict:
    df_main, _, _ = load_audited_data()
    df_main = df_main.reset_index(drop=True)
    preds = load_record_level_preds(df_main)
    grp_of = dict(zip(df_main["record_id"], df_main["entity_group_id"]))
    year_of = dict(zip(df_main["record_id"], df_main["file_year"]))
    n_g = df_main.groupby("entity_group_id")["record_id"].count()
    G = df_main["entity_group_id"].nunique()
    group_true = df_main.groupby("entity_group_id")["power_kwp"].apply(lambda s: s.to_numpy())

    # Her kaydın dış eğitim sözlüğüne göre [UNK] sözcükleri
    unk_of: Dict[str, List[str]] = {}
    for f in sorted(df_main["outer_test_fold"].unique()):
        tr = prepare_tokenized_dataset(df_main[df_main["outer_test_fold"] != f], fit_vocab=True)
        vocab = tr["vocab"]
        from src.data_prep import format_sequence, tokenize_text
        for _, r in df_main[df_main["outer_test_fold"] == f].iterrows():
            toks = tokenize_text(format_sequence(r["district_clean"], r["file_year"], r["name_clean"]))
            unk_of[r["record_id"]] = [t for t in toks if t not in vocab.token_to_idx]

    tops = {}
    for m, d in preds.items():
        d = d.copy()
        d["group"] = d.index.map(grp_of)
        d["year"] = d.index.map(year_of)
        d["signed_err"] = d["y_pred"] - d["y_true"]
        d["katki_kWp"] = d.apply(lambda r: r["abs_err"] / (n_g[r["group"]] * G), axis=1)
        mae_check = float(d["katki_kWp"].sum())
        d["katki_%"] = 100.0 * d["katki_kWp"] / mae_check
        d = d.sort_values("katki_kWp", ascending=False)
        top = d.head(5).copy()
        top["olası_neden"] = [probable_cause(r["y_true"], r["signed_err"], group_true[r["group"]], unk_of.get(i, []))
                              for i, r in top.iterrows()]
        tops[m] = {"top": top.reset_index(), "mae_check": mae_check}

    big_ids = df_main.sort_values("power_kwp", ascending=False).head(10)["record_id"].tolist()
    big = pd.DataFrame({"record_id": big_ids,
                        "grup": [grp_of[i] for i in big_ids], "yıl": [year_of[i] for i in big_ids],
                        "gerçek": [preds["B1"].loc[i, "y_true"] for i in big_ids]})
    summ = []
    for m in ["B1", "D2", "D3"]:
        big[f"{m}_işaretli_hata"] = [preds[m].loc[i, "y_pred"] - preds[m].loc[i, "y_true"] for i in big_ids]
        e = np.array([preds[m].loc[i, "y_pred"] - preds[m].loc[i, "y_true"] for i in big_ids])
        rest = preds[m].drop(index=big_ids)
        summ.append({"model": m, "düşük_tahmin_sayısı(10_kayıttan)": int((e < 0).sum()),
                     "büyük10_ort_işaretli_hata": round(float(e.mean()), 1),
                     "büyük10_ort_tahmin/gerçek": round(float(np.mean([preds[m].loc[i, "y_pred"] / preds[m].loc[i, "y_true"] for i in big_ids])), 2),
                     "diğer141_ort_işaretli_hata": round(float((rest["y_pred"] - rest["y_true"]).mean()), 1)})
    return {"tops": tops, "big": big, "big_summary": pd.DataFrame(summ)}


# ----------------------------------------------------------------------------------------------
# (4) ATTENTION YORUMU: attention ağırlıkları vs token silme
# ----------------------------------------------------------------------------------------------
def delete_token(ids_row: torch.Tensor, mask_row: torch.Tensor, j: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    j. token dizi İÇİNDEN ÇIKARILIR (sonraki tokenlar sola kayar, sona PAD eklenir, maske yeniden kurulur).
    Bu, j. konumu PAD ile DEĞİŞTİRMEK değildir: dizi gerçekten bir token kısalır.
    """
    real = int((~mask_row).sum())
    seq = ids_row[:real].tolist()
    del seq[j]
    pad_n = len(ids_row) - len(seq)
    new_ids = torch.tensor(seq + [PAD_IDX] * pad_n, dtype=ids_row.dtype)
    new_mask = torch.tensor([False] * len(seq) + [True] * pad_n, dtype=torch.bool)
    return new_ids, new_mask


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    r = spearmanr(a, b).correlation
    return float(r)


def attention_vs_deletion() -> Dict:
    df_main, _, _ = load_audited_data()
    df_main = df_main.reset_index(drop=True)
    sel = pd.read_csv(os.path.join(RESULTS_DIR, "nn_selected.csv"))

    per_record, examples = [], []
    weights_dir = os.path.join(RESULTS_DIR, "weights")
    for f in sorted(df_main["outer_test_fold"].unique()):
        tr_df = df_main[df_main["outer_test_fold"] != f]
        te_df = df_main[df_main["outer_test_fold"] == f]

        tr = prepare_tokenized_dataset(tr_df, fit_vocab=True)
        te = prepare_tokenized_dataset(te_df, vocab=tr["vocab"], max_len=tr["max_len"], fit_vocab=False)
        vocab = tr["vocab"]

        sel_d2 = sel[(sel["model"] == "D2") & (sel["outer_fold"] == f)].iloc[0]
        sel_d3 = sel[(sel["model"] == "D3") & (sel["outer_fold"] == f)].iloc[0]

        # Adım 8b (3): Dış eğitilen modellerin kaydedilmiş ağırlıklarını yükle (tohum 42)
        m2 = ModelD2(vocab_size=len(vocab), d_model=32, dropout=float(sel_d2["dropout"]))
        w2_path = os.path.join(weights_dir, f"D2_fold{f}_seed42.pt")
        m2.load_state_dict(torch.load(w2_path, weights_only=True))
        m2.eval()

        m3 = ModelD3(vocab_size=len(vocab), max_len=64, d_model=32, dropout=float(sel_d3["dropout"]))
        w3_path = os.path.join(weights_dir, f"D3_fold{f}_seed42.pt")
        m3.load_state_dict(torch.load(w3_path, weights_only=True))
        m3.eval()
        order = sorted(range(len(te_df)), key=lambda i: te_df.iloc[i]["record_id"])
        for rank_in_fold, i in enumerate(order):
            ids, mask = te["token_ids"][i], te["key_padding_mask"][i]
            real = int((~mask).sum())
            pos = list(range(1, real))             # [CLS] (konum 0) hariç; silinirse D3 okuma noktası kaybolur
            if len(pos) < 3:
                continue
            with torch.no_grad():
                z2, alpha = m2(ids[None], padding_mask=mask[None], return_attention_weights=True)
                z3, (w1, w2) = m3(ids[None], padding_mask=mask[None], return_attention_weights=True)
                dels = [delete_token(ids, mask, j) for j in pos]
                bi = torch.stack([d[0] for d in dels]); bm = torch.stack([d[1] for d in dels])
                dz2 = (m2(bi, padding_mask=bm) - z2).abs().numpy()
                dz3 = (m3(bi, padding_mask=bm) - z3).abs().numpy()
            a2 = alpha[0, pos].numpy()
            a31 = w1[0, :, 0, :].mean(0)[pos].numpy()   # [CLS] satırı, başlık ortalaması, blok 1
            a32 = w2[0, :, 0, :].mean(0)[pos].numpy()   # [CLS] satırı, başlık ortalaması, blok 2
            rec = te_df.iloc[i]["record_id"]
            row = {"record_id": rec, "outer_fold": int(f), "n_token": len(pos),
                   "rho_D2": _rho(a2, dz2), "rho_D3_blok1": _rho(a31, dz3), "rho_D3_blok2": _rho(a32, dz3)}
            per_record.append(row)
            if rank_in_fold == 0:                    # kural: her katın (record_id sırasıyla) ilk test kaydı
                examples.append({"record": rec, "fold": int(f), "row": row,
                                 "tokens": vocab.decode(ids[pos].tolist()) if False else
                                 [vocab.idx_to_token.get(t, "[UNK]") for t in ids[pos].tolist()],
                                 "a2": a2, "a31": a31, "a32": a32, "dz2": dz2, "dz3": dz3,
                                 "name": te_df.iloc[i]["name_clean"]})
    df_rec = pd.DataFrame(per_record)
    df_rec.to_csv(os.path.join(RESULTS_DIR, "attention_deletion.csv"), index=False, encoding="utf-8")
    return {"per_record": df_rec, "examples": examples}


def _print_df(df, **kw):
    print(df.to_string(index=False, **kw))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)

    print("=" * 100)
    print(f"2) {LEAK_LABEL.upper()}  (B1, rastgele satır bölmesi, 5 kat, tohum 42)")
    print("=" * 100)
    lk = leakage_diagnostic()
    _print_df(lk["table"])
    print(f"Gruplu − rastgele tesis dengeli MAE farkı: {lk['diff']:+.2f} kWp (pozitif = rastgele bölme daha düşük hata)")
    print(f"Rastgele bölmede kat bazında ortak tesis oranı: {[round(x, 3) for x in lk['per_fold_rand']]}")
    print("H2 kararı (kural: rastgele MAE gruplu MAE'den düşük değilse red): "
          + ("REDDEDİLDİ" if lk["h2_rejected"] else "REDDEDİLMEDİ") +
          f"  (rastgele {lk['mr']:.2f} {'<' if lk['mr'] < lk['mg'] else '>='} gruplu {lk['mg']:.2f}; ortak tesis oranı {lk['ratio_rand']:.3f}, gruplu {lk['ratio_grp']:.3f})")
    print("UYARI: Rastgele bölme sonucu SIZINTI TANISIDIR, geçerli skor değildir; hiçbir model/ölçüt seçiminde kullanılmaz.")

    print("\n" + "=" * 100)
    print("3) HATA ANALİZİ: tesis dengeli MAE'ye en çok katkı veren 5 kayıt (tohum ortalamalı hata)")
    print("   katkı = |hata| / (grup kayıt sayısı x 76); katkıların toplamı = modelin tesis dengeli MAE'si")
    print("=" * 100)
    ea = error_analysis()
    for m, v in ea["tops"].items():
        print(f"\n--- {m} (katkı toplamı kontrolü = tesis dengeli MAE {v['mae_check']:.2f} kWp) ---")
        t = v["top"][["record_id", "group", "year", "y_true", "y_pred", "signed_err", "katki_kWp", "katki_%", "olası_neden"]].copy()
        for c in ["y_true", "y_pred", "signed_err"]:
            t[c] = t[c].round(1)
        t["katki_kWp"] = t["katki_kWp"].round(2)
        t["katki_%"] = t["katki_%"].round(1)
        _print_df(t)
    print("\n--- En büyük 10 kayıt (gerçek güce göre): işaretli hata = tahmin − gerçek (kWp) ---")
    big = ea["big"].copy()
    for c in big.columns:
        if c.endswith("işaretli_hata") or c == "gerçek":
            big[c] = big[c].round(1)
    _print_df(big)
    _print_df(ea["big_summary"])

    print("\n" + "=" * 100)
    print("4) ATTENTION vs TOKEN SİLME (D2 ağırlıkları ve D3 [CLS] attention'ı)")
    print("   Token silme: token diziden ÇIKARILIR (PAD ile değiştirilmez); önem = |Δ log1p kestirim|, [CLS] hariç.")
    print("   NOT: Yüksek attention nedensel önem OLARAK YORUMLANMAMALIDIR; silme analizi de yalnızca modelin kendi")
    print("   duyarlılığını ölçer (gerçek dünyadaki nedensellik değil). Modeller Adım 8b kapsamında results/weights/ altına")
    print("   kaydedilen dış eğitilmiş nihai modellerdir (tohum 42).")
    print("=" * 100)
    av = attention_vs_deletion()
    for ex in av["examples"]:
        print(f"\nÖrnek: {ex['record']} (dış kat {ex['fold']}) | ad: {ex['name']}")
        d = pd.DataFrame({"token": ex["tokens"], "D2_alpha": ex["a2"].round(3), "D3_blok1_CLS": ex["a31"].round(3),
                          "D3_blok2_CLS": ex["a32"].round(3), "D2_silme_dz": ex["dz2"].round(3), "D3_silme_dz": ex["dz3"].round(3)})
        _print_df(d)
        r = ex["row"]
        print(f"  Spearman: D2 α~silme = {r['rho_D2']:.2f} | D3 blok1 ~silme = {r['rho_D3_blok1']:.2f} | D3 blok2 ~silme = {r['rho_D3_blok2']:.2f}")
    rec = av["per_record"]
    print(f"\nTüm {len(rec)} test kaydı üzerinde kayıt başına Spearman (NaN olanlar hariç):")
    rows = []
    for c, lab in [("rho_D2", "D2 α ~ D2 silme"), ("rho_D3_blok1", "D3 blok1 [CLS] ~ D3 silme"), ("rho_D3_blok2", "D3 blok2 [CLS] ~ D3 silme")]:
        v = rec[c].dropna()
        rows.append({"karşılaştırma": lab, "geçerli_kayıt": len(v), "ortalama_rho": round(v.mean(), 3),
                     "medyan_rho": round(v.median(), 3), "rho>0_oranı": round(float((v > 0).mean()), 3)})
    _print_df(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
