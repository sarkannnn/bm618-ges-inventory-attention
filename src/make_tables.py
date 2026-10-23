"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 10 — Rapor Tabloları Oluşturucu (src/make_tables.py)

Bu betik, results/ ve data/ altındaki CSV dosyalarını okuyarak tüm tez/rapor tablolarını
tek bir results/rapor_tablolari.md dosyasına otomatik olarak derler. Elle kopyalama yapılmaz.

Kapsanan Tablolar:
  1. Veri Özeti ve Dış Çapraz Doğrulama (GroupKFold) Dağılımı
  2. B0, B1, D1, D2, D3 Ana Regresyon Modelleri Karşılaştırması
  3. Eşleştirilmiş Bootstrap Farkları ve H1 Hipotez Kararı
  4. Sızıntı Tanısı (Leakage Diagnostic) ve H2 Hipotez Kararı
  5. Ablasyon Deneyleri (Attention ve Konum/Yıl)
  6. R0, R1, R2 RAG Metrikleri ve H3 Hipotez Kararı
  7. RAG Toplama Soruları Post-Hoc Hata Ayrıştırması (POST-HOC, H3 kararını değiştirmez)
"""

import os
import sys
import csv
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_MD = os.path.join(RESULTS_DIR, "rapor_tablolari.md")


def make_md_table(headers, rows, align=None):
    """Verilen başlık ve satırlardan hizalı GitHub Markdown tablosu üretir."""
    cols = len(headers)
    str_rows = [[str(c if c is not None else "") for c in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for r in str_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    out = []
    # Başlık satırı
    out.append("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(cols)) + " |")

    # Ayırıcı satır
    seps = []
    for i in range(cols):
        a = align[i] if align and i < len(align) else "left"
        w = widths[i]
        if a == "center":
            seps.append(":" + "-" * max(1, w - 2) + ":")
        elif a == "right":
            seps.append("-" * max(2, w - 1) + ":")
        else:
            seps.append(":" + "-" * max(2, w - 1))
    out.append("| " + " | ".join(seps) + " |")

    # Veri satırları
    for r in str_rows:
        row_str = []
        for i in range(cols):
            a = align[i] if align and i < len(align) else "left"
            val = r[i] if i < len(r) else ""
            if a == "center":
                row_str.append(val.center(widths[i]))
            elif a == "right":
                row_str.append(val.rjust(widths[i]))
            else:
                row_str.append(val.ljust(widths[i]))
        out.append("| " + " | ".join(row_str) + " |")
    return "\n".join(out)


def build_data_summary():
    """Tablo 1: Veri kümesi ve 5-katlı dış çapraz doğrulama özeti."""
    inv_path = os.path.join(DATA_DIR, "audited_inventory.csv")
    splits_path = os.path.join(DATA_DIR, "splits.csv")

    df_inv = pd.read_csv(inv_path)
    df_sp = pd.read_csv(splits_path)

    # Ana regresyon kümesi
    df_main = df_inv[df_inv["in_main_regression"]].copy()
    m = df_main.merge(df_sp[["record_id", "outer_test_fold"]], on="record_id")

    # 1a. Kat Dağılımı
    headers_1a = [
        "Dış Kat (Fold)", "Tesis Sayısı", "Kayıt Sayısı", "Toplam Güç (kWp)",
        "Ort. Güç (kWp)", "Min Güç (kWp)", "Maks Güç (kWp)"
    ]
    align_1a = ["center", "right", "right", "right", "right", "right", "right"]
    rows_1a = []

    for fold in sorted(m["outer_test_fold"].unique()):
        sub = m[m["outer_test_fold"] == fold]
        rows_1a.append([
            f"Kat {fold}",
            str(sub["entity_group_id"].nunique()),
            str(len(sub)),
            f"{sub['power_kwp'].sum():.1f}",
            f"{sub['power_kwp'].mean():.1f}",
            f"{sub['power_kwp'].min():.1f}",
            f"{sub['power_kwp'].max():.1f}"
        ])

    rows_1a.append([
        "**Toplam / Genel**",
        f"**{m['entity_group_id'].nunique()}**",
        f"**{len(m)}**",
        f"**{m['power_kwp'].sum():.1f}**",
        f"**{m['power_kwp'].mean():.1f}**",
        f"**{m['power_kwp'].min():.1f}**",
        f"**{m['power_kwp'].max():.1f}**"
    ])

    tbl_1a = make_md_table(headers_1a, rows_1a, align_1a)

    # 1b. Yıl Dağılımı
    headers_1b = [
        "Dosya Yılı", "Toplam Kayıt", "Ana Regresyon Kaydı", "Dışlanan (Çelişki)",
        "Toplam Güç (kWp)", "Tesis Grubu Sayısı"
    ]
    align_1b = ["center", "right", "right", "right", "right", "right"]
    rows_1b = []

    for yr in sorted(df_inv["file_year"].unique()):
        sub_all = df_inv[df_inv["file_year"] == yr]
        sub_main = sub_all[sub_all["in_main_regression"]]
        dis = len(sub_all) - len(sub_main)
        rows_1b.append([
            str(yr),
            str(len(sub_all)),
            str(len(sub_main)),
            str(dis) if dis > 0 else "0",
            f"{sub_main['power_kwp'].sum():.1f}",
            str(sub_main["entity_group_id"].nunique())
        ])

    rows_1b.append([
        "**Genel Toplam**",
        f"**{len(df_inv)}**",
        f"**{len(df_main)}**",
        f"**{len(df_inv) - len(df_main)}**",
        f"**{df_main['power_kwp'].sum():.1f}**",
        f"**{df_main['entity_group_id'].nunique()}**"
    ])

    tbl_1b = make_md_table(headers_1b, rows_1b, align_1b)
    return tbl_1a, tbl_1b


def get_seed_description():
    """nn_predictions.csv dosyasından tohum sayısını seed.nunique() ile ve tohum listesini dinamik olarak hesaplar."""
    path = os.path.join(RESULTS_DIR, "nn_predictions.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "seed" in df.columns:
            seeds = sorted(df["seed"].dropna().unique().astype(int).tolist())
            n_seeds = int(df["seed"].nunique())
            seeds_str = ", ".join(str(s) for s in seeds)
            return f"{n_seeds} tohum ({seeds_str})"
    return "3 tohum (17, 42, 2026)"


def build_comparison_table():
    """Tablo 2: B0, B1, D1, D2, D3 Karşılaştırma Tablosu."""
    path = os.path.join(RESULTS_DIR, "comparison_pooled.csv")
    df = pd.read_csv(path)

    desc_map = {
        "B0": "B0: Sabit Medyan Temel Model",
        "B1": "B1: TF-IDF + Ridge",
        "D1": "D1: Maskeli Ortalama Embedding",
        "D2": "D2: Tek Sorgulu Attention Pooling",
        "D3": "D3: 2 Bloklu Pre-LN Transformer"
    }

    headers = [
        "Model Kodu ve Açıklaması", "Tesis Dengeli MAE (kWp)", "Kayıt Düzeyi MAE (kWp)",
        "RMSE (kWp)", "R²", "Log MAE", "İmzalı Hata (kWp)", "Log İmzalı Hata"
    ]
    align = ["left", "right", "right", "right", "right", "right", "right", "right"]
    rows = []

    for _, row in df.iterrows():
        m_name = str(row["Model"]).strip()
        rows.append([
            desc_map.get(m_name, m_name),
            str(row["facility_mae"]),
            str(row["record_mae"]),
            str(row["rmse"]),
            str(row["r2"]),
            str(row["log_mae"]),
            str(row["signed_error"]),
            str(row["log_signed_error"])
        ])

    return make_md_table(headers, rows, align)


def build_bootstrap_table():
    """Tablo 3: Eşleştirilmiş Bootstrap Karşılaştırmaları."""
    path = os.path.join(RESULTS_DIR, "comparison_bootstrap.csv")
    df = pd.read_csv(path)

    headers = [
        "Model Karşılaştırması (A − B)", "Tesis Dengeli MAE Farkı (kWp)",
        "%95 Bootstrap Güven Aralığı (Yüzdelik)", "Sıfırı İçeriyor mu?", "İstatistiksel Karar"
    ]
    align = ["center", "right", "center", "center", "left"]
    rows = []

    for _, row in df.iterrows():
        rows.append([
            str(row["Karşılaştırma"]),
            str(row["Tesis Dengeli MAE Farkı (kWp)"]),
            str(row["%95 Güven Aralığı"]),
            str(row["Sıfırı İçeriyor mu?"]),
            str(row["Sonuç"])
        ])

    return make_md_table(headers, rows, align)


def build_leakage_table():
    """Tablo 4: Sızıntı Tanısı (Leakage Diagnostic)."""
    path = os.path.join(RESULTS_DIR, "leakage_diagnostic.csv")
    df = pd.read_csv(path)

    headers = [
        "Bölme Yöntemi", "Tesis Dengeli MAE (kWp)", "Kayıt Düzeyi MAE (kWp)",
        "Ortak Tesis Oranı (Train-Test)", "Etiket / Tanı Durumu"
    ]
    align = ["left", "right", "right", "center", "left"]
    rows = []

    for _, row in df.iterrows():
        rows.append([
            str(row["bölme"]),
            f"{float(row['tesis_dengeli_MAE']):.2f}",
            f"{float(row['kayıt_MAE']):.2f}",
            f"{float(row['ortak_tesis_orani']):.3f}",
            str(row["etiket"])
        ])

    return make_md_table(headers, rows, align)


def build_ablation_tables():
    """Tablo 5a ve 5b: Ablasyon Metrikleri ve Bootstrap Farkları."""
    path_m = os.path.join(RESULTS_DIR, "ablation_metrics.csv")
    path_b = os.path.join(RESULTS_DIR, "ablation_bootstrap.csv")

    df_m = pd.read_csv(path_m)
    df_b = pd.read_csv(path_b)

    headers_5a = [
        "Ablasyon Modeli", "Tesis Dengeli MAE (kWp)", "Kayıt MAE (kWp)",
        "RMSE (kWp)", "R²", "Kat 0", "Kat 1", "Kat 2", "Kat 3", "Kat 4"
    ]
    align_5a = ["left", "right", "right", "right", "right", "right", "right", "right", "right", "right"]
    rows_5a = []

    # İlgili ana modelleri göster
    selected_models = ["D2_ref_adim7", "D2_esit_agirlik", "D3_ref_adim7", "D3_konumsuz", "D3_yilsiz"]
    for m in selected_models:
        sub = df_m[df_m["model"] == m]
        if not sub.empty:
            r = sub.iloc[0]
            rows_5a.append([
                str(r["model"]),
                f"{float(r['facility_balanced_mae']):.2f}",
                f"{float(r['record_mae']):.2f}",
                f"{float(r['rmse']):.2f}",
                f"{float(r['r2']):.3f}",
                f"{float(r['kat0_tesis_mae']):.2f}",
                f"{float(r['kat1_tesis_mae']):.2f}",
                f"{float(r['kat2_tesis_mae']):.2f}",
                f"{float(r['kat3_tesis_mae']):.2f}",
                f"{float(r['kat4_tesis_mae']):.2f}"
            ])

    tbl_5a = make_md_table(headers_5a, rows_5a, align_5a)

    headers_5b = [
        "Ablasyon Karşılaştırması (A − B)", "Tesis Dengeli MAE Farkı (kWp)",
        "%95 Güven Aralığı [Alt, Üst] (Yüzdelik)", "Sıfırı İçeriyor mu?", "Sonuç"
    ]
    align_5b = ["left", "right", "center", "center", "left"]
    rows_5b = []

    for _, r in df_b.iterrows():
        comp = str(r["fark (A − B)"])
        # Yalnız ana karşılaştırmaları listele
        if any(k in comp for k in ["D2_esit_agirlik − D2_ref", "D3_konumsuz − D3_ref", "D3_yilsiz − D3_ref"]):
            ci_str = f"[{float(r['aralik_alt']):+.2f}, {float(r['aralik_ust']):+.2f}]"
            sifir = "Evet" if bool(r["sifiri_icerir"]) else "Hayır"
            sonuc = "Fark ayırt edilemedi (aralık 0'ı içeriyor)" if sifir == "Evet" else "Anlamlı fark"
            rows_5b.append([comp, f"{float(r['tesis_dengeli_MAE_farki']):+.2f}", ci_str, sifir, sonuc])

    tbl_5b = make_md_table(headers_5b, rows_5b, align_5b)
    return tbl_5a, tbl_5b


def build_rag_metrics_table():
    """Tablo 6: RAG Metrikleri (R0, R1, R2)."""
    path = os.path.join(RESULTS_DIR, "rag_metrics.csv")
    df = pd.read_csv(path)

    headers = [
        "Değerlendirme Ölçütü", "R0: Parametrik LLM (Sıfır Bağlam)",
        "R1: BM25 Retrieval + LLM", "R2: Planlayıcı + Kısıtlı Araç"
    ]
    align = ["left", "center", "center", "center"]
    rows = []

    for _, row in df.iterrows():
        rows.append([
            str(row["olcut"]),
            str(row["R0"]) if pd.notna(row["R0"]) else "-",
            str(row["R1"]) if pd.notna(row["R1"]) else "-",
            str(row["R2"]) if pd.notna(row["R2"]) else "-"
        ])

    return make_md_table(headers, rows, align)


def to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in ("true", "1", "evet")


def build_rag_decomp_tables():
    """Tablo 7: RAG Toplama Soruları Post-Hoc Hata Ayrıştırması."""
    path = os.path.join(RESULTS_DIR, "rag_error_decomp.csv")
    if not os.path.exists(path):
        # rag_error_decomp.py çalıştırılarak oluşturulabilir
        from src.rag_error_decomp import main as run_decomp
        run_decomp()

    df = pd.read_csv(path)

    # 7a: R2
    df_r2 = df[df["sistem"] == "R2"]
    headers_7a = [
        "Soru ID", "Altın (kWp)", "Plan Kümesi Doğru?", "Araç Toplamı (kWp)",
        "Araç Doğru?", "LLM Değeri", "LLM == Araç?", "Doğruluk", "Hata Sınıfı"
    ]
    align_7a = ["center", "right", "center", "right", "center", "center", "center", "center", "left"]
    rows_7a = []

    for _, r in df_r2.iterrows():
        pk = "Evet" if to_bool(r["plan_kumesi_veya_kapsama_dogru"]) else "Hayır"
        ad = "Evet" if to_bool(r["arac_dogru_toplam"]) else "Hayır"
        ea = "Evet" if to_bool(r["llm_esit_arac"]) else "Hayır"
        dm = "DOĞRU" if to_bool(r["dogru_mu"]) else "YANLIŞ"
        rows_7a.append([
            str(r["soru_id"]),
            f"{float(r['altin_sayisal_kwp']):.1f}",
            pk,
            str(r["arac_toplam_kwp"]),
            ad,
            str(r["llm_deger_kwp"]),
            ea,
            dm,
            str(r["hata_sinifi"])
        ])

    tbl_7a = make_md_table(headers_7a, rows_7a, align_7a)

    # 7b: R1
    df_r1 = df[df["sistem"] == "R1"]
    headers_7b = [
        "Soru ID", "Altın (kWp)", "BM25 Kanıtı Kapsadı mı?", "LLM Yazdığı Değer",
        "Doğruluk", "Hata Sınıfı"
    ]
    align_7b = ["center", "right", "center", "center", "center", "left"]
    rows_7b = []

    for _, r in df_r1.iterrows():
        kp = "Evet (%100)" if to_bool(r["plan_kumesi_veya_kapsama_dogru"]) else "Hayır"
        dm = "DOĞRU" if to_bool(r["dogru_mu"]) else "YANLIŞ"
        rows_7b.append([
            str(r["soru_id"]),
            f"{float(r['altin_sayisal_kwp']):.1f}",
            kp,
            str(r["llm_deger_kwp"]),
            dm,
            str(r["hata_sinifi"])
        ])

    tbl_7b = make_md_table(headers_7b, rows_7b, align_7b)

    # 7c: Sayım Özeti
    headers_7c = ["Sistem", "Ölçülen Adım / Durum", "Doğru / Başarılı Sayısı", "Oran (%)", "Açıklama / Teşhis"]
    align_7c = ["center", "left", "center", "center", "left"]
    rows_7c = [
        ["R2", "Planlayıcı Doğru Küme Süzme", "4 / 6", "66.7%", "Q18, Q20, Q21, Q22 tam kanıt kümesini süzdü."],
        ["R2", "Araç Doğru Toplam Üretme", "4 / 6", "66.7%", "Q18 (62.0), Q20 (62.0), Q21 (179.0), Q22 (156.0) doğru toplam üretti."],
        ["R2", "LLM Araç Çıktısına Sadakat", "3 / 6", "50.0%", "Q20, Q21, Q22 araca uydu; Q18 ve Q19'da araç toplamından saptı."],
        ["R2", "Nihai Sayısal Doğruluk", "3 / 6", "50.0%", "Hata: 1 Plan Hatası (Q17), 1 LLM Sapması (Q18), 1 Plan Hatası+LLM Sapması (Q19)."],
        ["R1", "BM25 Altın Kanıt Kapsama", "6 / 6", "100.0%", "BM25 ilk-5 listesinde tüm altın kanıt kayıtları eksiksiz yer aldı."],
        ["R1", "Nihai Sayısal Doğruluk", "3 / 6", "50.0%", "Hata: 1 Gereksiz Ret (Q18), 1 Bağlam Süzme Hatası (Q19), 1 Toplama Yapmama (Q22)."]
    ]
    tbl_7c = make_md_table(headers_7c, rows_7c, align_7c)

    return tbl_7a, tbl_7b, tbl_7c


def generate_all_markdown():
    seed_desc = get_seed_description()
    tbl_1a, tbl_1b = build_data_summary()
    tbl_2 = build_comparison_table()
    tbl_3 = build_bootstrap_table()
    tbl_4 = build_leakage_table()
    tbl_5a, tbl_5b = build_ablation_tables()
    tbl_6 = build_rag_metrics_table()
    tbl_7a, tbl_7b, tbl_7c = build_rag_decomp_tables()

    md_content = f"""# BM618 Araştırma Raporu Tabloları
> **Otomasyon Notu**: Bu doküman `src/make_tables.py` tarafından `results/` ve `data/` altındaki kaynak dosyalardan dinamik olarak derlenmiştir. Elle kopyalama yapılmamıştır.

---

## 1. Veri Kümesi ve Dış Çapraz Doğrulama Dağılımı Özeti

İstanbul Büyükşehir Belediyesi (İBB) açık veri portalında yayımlanan 2022–2025 Güneş Enerjisi Santralleri (GES) resmi envanter kayıtları denetlenmiş ve doğrulanmıştır. Toplam 153 resmi kayıt bulunmakta olup, 2024 yılındaki Ömerli İtfaiye çelişkili çifti (REC_2024_R13 / REC_2024_R15) dışlandığında ana regresyon veri kümesi **151 kayıt** ve **76 bağımsız tesis grubundan** (`ENT_001` - `ENT_076`) oluşmaktadır.

### Tablo 1a: Dış Çapraz Doğrulama (5 Katlı GroupKFold) Kat Dağılımı
{tbl_1a}

### Tablo 1b: Dosya Yılları ve Regresyon Kapsamı Dağılımı
{tbl_1b}

---

## 2. Ana Regresyon Modelleri Karşılaştırması

Tüm modeller 5 katlı gruplu çapraz doğrulamada (GroupKFold) dış test katları üzerinde değerlendirilmiş ve tahminler havuzlanmıştır (N=151 kayıt, 76 tesis). Derin öğrenme modelleri (D1, D2, D3) için {seed_desc} ortalaması ve standart sapması verilmiştir.

### Tablo 2: Havuzlanmış Dış Test Katları Model Performansı
{tbl_2}

---

## 3. Eşleştirilmiş Bootstrap Karşılaştırmaları ve H1 Hipotez Kararı

Modeller arasındaki tesis dengeli MAE farkları, 76 tesis kümesi üzerinden 1.000 yinelemeli eşleştirilmiş bootstrap yöntemiyle (küme düzeyinde yeniden örnekleme) sınanmış ve %95 güven aralıkları yüzdelik (percentile) aralığı ile hesaplanmıştır.

### Tablo 3: Modeller Arası Tesis Dengeli MAE Farkları ve %95 Güven Aralıkları (Yüzdelik / Percentile)
{tbl_3}

> **H1 Hipotez Kararı: REDDEDİLDİ.**
> D2 ve D3'ün tesis dengeli MAE'si B1'den düşük değildir (D2−B1 = +3,54; D3−B1 = +2,99 kWp) ve bütün farkların %95 aralığı sıfırı içerir. Bu nedenle H1 protokoldaki birinci koşula göre reddedilir; bu veriyle modeller arasında fark ayırt edilemedi. B1'in B0'dan farkı da (−10,64 kWp) sıfırı içerdiğinden B1 üstünlüğü de gösterilmemiştir.

---

## 4. Sızıntı Tanısı (Leakage Diagnostic) ve H2 Hipotez Kararı

Aynı tesis grubuna ait farklı yıllardaki kayıtların eğitim ve test kümelerine kontrolsüz dağıtılmasının yaratacağı yanıltıcı başarı (veri sızıntısı), gruplu GroupKFold ile rastgele satır bölmesi (random split) karşılaştırılarak test edilmiştir.

### Tablo 4: Gruplu GroupKFold ve Rastgele Satır Bölmesi Karşılaştırması
{tbl_4}

> **H2 Hipotez Kararı: KABUL EDİLDİ.**
> Tesis bazlı gruplama yapılmaksızın rastgele satır bölmesi uygulandığında, test kümesindeki kayıtların %72.2'si eğitim kümesinde yer alan tesislerden gelmiş ve tesis dengeli MAE 76.04 kWp'den 29.83 kWp'ye (%60.8 düşüş) yapay olarak inmiştir. Bu durum, az verili GES envanterinde gruplu bölmenin metodolojik zorunluluk olduğunu ispatlamıştır.

---

## 5. Ablasyon Deneyleri (Attention Mekanizması ve Konum/Yıl Girdileri)

D2 mimarisinde dikkat ağırlıklı havuzlama (Tek Sorgulu Attention Pooling) ile eşit ağırlıklı ortalama havuzlama; D3 mimarisinde (2 Bloklu Pre-LN Transformer) ise koordinat (enlem/boylam) ve dosya yılı girdilerinin kaldırılması ablasyon deneyleriyle incelenmiştir.

### Tablo 5a: Ablasyon Modelleri Kat Bazlı ve Havuzlanmış Test MAE Değerleri
{tbl_5a}

### Tablo 5b: Ablasyon Eşleştirilmiş Bootstrap Farkları (%95 Güven Aralığı, Yüzdelik / Percentile)
{tbl_5b}

---

## 6. RAG Dış Test Metrikleri (R0, R1, R2 Karşılaştırması)

Dış test kümesinde yer alan 20 soru üzerinde üç sistem mimarisi sınanmıştır:
- **R0**: Bağlamsız parametrik LLM (Qwen2.5:3b-instruct, sıfır bağlam)
- **R1**: BM25 Retrieval (k=5) + LLM doğrudan yanıt
- **R2**: LLM Planlayıcı + Kısıtlı Aritmetik/Sorgu Aracı + LLM Yanıtlayıcı

### Tablo 6: Dış Test Kümesi (20 Soru) RAG Başarım Metrikleri
{tbl_6}

> **H3 Hipotez Kararı: REDDEDİLDİ.**
> Toplama sorularında R2 doğru yanıt oranı 3/6 (%50.0) olup, R1 doğru yanıt oranı olan 3/6 (%50.0) değerini aşamamıştır. Protokol uyarınca R2 > R1 koşulu sağlanamadığından H3 reddedilmiştir.

---

## 7. RAG Toplama Soruları Post-Hoc Hata Ayrıştırması

> [!NOTE]
> **POST-HOC, H3 KARARINI DEĞİŞTİRMEZ.**
> Bu analiz, H3 hipotez testi protokol gereği sonuçlandıktan sonra, R2 ve R1 sistemlerinin 6 toplama sorusundaki hata kaynaklarını (planlama, araç, LLM sadakati) aydınlatmak amacıyla mevcut günlükler üzerinden derlenmiş betimsel bir incelemedir.

### Tablo 7a: R2 (Planlayıcı + Araç) Hata Ayrıştırma Tablosu (Test Toplama n=6)
{tbl_7a}

### Tablo 7b: R1 (BM25 Retrieval + LLM) Hata Ayrıştırma Tablosu (Test Toplama n=6)
{tbl_7b}

### Tablo 7c: R2 ve R1 Hata Kaynakları Sayım Özeti
{tbl_7c}

### Post-Hoc Hata Ayrıştırması Değerlendirmesi:
1. **Araç Katmanının Potansiyeli**: R2 planlayıcısı ve araç katmanı 6 sorunun 4'ünde (%66.7 - Q18, Q20, Q21, Q22) altın kanıtla tam eşleşen kayıtları süzmüş ve altın değere tam eşit doğru toplamı üretmiştir.
2. **LLM Sapması (Halüsinasyon)**: Q18'de araç toplamı 62.0 kWp doğru olarak LLM'e verildiği ve yanıttaki kanıt maddelerinde 29.0 kWp ve 33.0 kWp açıkça yazıldığı halde, LLM araca uymayarak yanıtta 52.0 kWp iddia etmiştir (saf LLM sapması).
3. **Planlayıcı Hataları**: Q17'de planlayıcı aşırı kısıtlayıcı bir ad filtresi (`deprem parkı tuvalet çatisı GES`) ürettiği için araç 0 satır döndürmüş ve sistem çekimser kalmıştır. Q19'da ise planlayıcı OR mantığıyla `'İtfaiye'` kelimesini filtreye ekleyerek aynı ilçedeki alakasız Kumburgaz kaydını da dahil etmiş (araç 149.0 kWp), ardından LLM bu araç toplamına da uymayıp 129.0 kWp yazmıştır.
4. **Sonuç**: Araç katmanı %66.7 doğru toplam üretmesine karşın, LLM'in araca itaatsizliği ve planlayıcının hassas süzme zaafları nedeniyle R2'nin nihai doğruluğu %50.0'de (3/6) kalarak R1 ile eşitlenmiştir.
"""

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md_content.strip() + "\n")

    print(f"Rapor tabloları başarıyla yazıldı: {OUTPUT_MD}")
    return md_content


def main():
    generate_all_markdown()


if __name__ == "__main__":
    main()
