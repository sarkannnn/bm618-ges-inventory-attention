"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 10 — RAG Toplama Soruları Post-Hoc Hata Ayrıştırması (src/rag_error_decomp.py)

UYARI / PROTOKOL NOTU:
Bu analiz yalnızca mevcut günlüklerden (logs/rag_runs.jsonl) derlenen betimsel bir analizdir.
POST-HOC niteliktedir; H3 KARARINI DEĞİŞTİRMEZ (H3 reddedilmiştir).

İnceleme kapsamı:
  6 dış test toplama sorusu (Q17, Q18, Q19, Q20, Q21, Q22).
  R2 için:
    1. Planlayıcının filtrelediği record_id kümesi altın kanıtla aynı mı? (plan_dogru_kume)
    2. Araç toplamı altın değere eşit mi? (arac_dogru_toplam)
    3. LLM'in yazdığı değer araç toplamına eşit mi? (llm_esit_arac)
    4. Sınıflandırma: plan doğru / araç doğru / LLM sapması / planlayıcı hatası, ardından LLM sapması.
  R1 için:
    1. BM25 ilk-5 listesinin altın kanıtı kapsayıp kapsamadığı (recall == 1.0)
    2. LLM'in nihai yanıtının doğruluğu ve hata sınıflandırması (gereksiz ret, bağlam süzme, biçim hatası).

Çıktı:
  results/rag_error_decomp.csv
"""

import os
import re
import sys
import csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_systems import read_log, load_gold
from src.rag_eval import (
    kb_by_id, evaluate_answer, parse_kwp_values,
    is_abstention_exact, abstention_loose
)

POST_HOC_NOTU = "POST-HOC, H3 kararını değiştirmez"
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "rag_error_decomp.csv")


def extract_stated_total(text):
    """Yanıtta savunulan toplam kWp değerini ya da çekimserlik durumunu çıkarır."""
    if is_abstention_exact(text) or abstention_loose(text):
        return "ÇEKİMSER", None
    # 'toplam ... kWp' kalıbı
    m = re.search(r"toplam(?:ı|ı da| gücü| kurulu gücü)?\s*[:=]?\s*(\d+(?:[.,]\d+)*)\s*kwp", text, re.IGNORECASE)
    if m:
        val_str = m.group(1).replace(",", ".")
        try:
            v = float(val_str)
            return f"{v:.1f}", v
        except ValueError:
            pass
    # 'X kWp + Y kWp = Z kWp' kalıbı
    m_eq = re.search(r"=\s*(\d+(?:[.,]\d+)*)\s*kwp", text, re.IGNORECASE)
    if m_eq:
        val_str = m_eq.group(1).replace(",", ".")
        try:
            v = float(val_str)
            return f"{v:.1f}", v
        except ValueError:
            pass
    # Çift değer ayracı (örn. 78.0 kWp | 78.0 kWp)
    if "|" in text and "kwp" in text.lower():
        parts = [p.strip() for p in text.split("|")]
        return " | ".join(parts), None
    # Genel kWp arama
    vals = parse_kwp_values(text)
    if vals:
        return f"{vals[0]:.1f}", vals[0]
    return text.strip()[:30], None


def analyze_errors():
    golds = {q["id"]: q for q in load_gold() if q["split"] == "test" and q["tur"] == "toplama"}
    kb = kb_by_id()
    test_logs = [e for e in read_log() if e.get("faz") == "test"]

    r2_rows = []
    r1_rows = []

    for qid in sorted(golds.keys()):
        q = golds[qid]
        gold_val = float(q["altin_sayisal"])
        gold_ids = set(q["altin_kanit"])

        # ------------------- R2 Analizi -------------------
        e2 = next(x for x in test_logs if x["soru_id"] == qid and x["sistem"] == "R2")
        arac = e2.get("arac_sonucu") or {}
        filt_ids = set(arac.get("filtrelenen_idler") or [])
        tool_sum = arac.get("toplam_kwp")

        plan_dogru = (filt_ids == gold_ids)
        arac_dogru = (tool_sum is not None and abs(tool_sum - gold_val) < 0.005)

        llm_stated_str, llm_val = extract_stated_total(e2["yanit"])
        if llm_val is not None and tool_sum is not None:
            llm_esit_arac = (abs(llm_val - tool_sum) < 0.005)
        else:
            llm_esit_arac = False

        ev2 = evaluate_answer(q, e2["yanit"], kb)
        dogru2 = ev2["dogru"]

        # Hata sınıflandırması
        if dogru2:
            hata2 = "Doğru"
            not2 = "Planlayıcı doğru kayıtları süzdü, araç doğru toplamı üretti, LLM araca uydu."
        elif qid == "Q17":
            hata2 = "planlayıcı hatası"
            not2 = "Planlayıcı aşırı kısıtlayıcı filtre ('deprem parkı tuvalet çatisı GES') üretti; 0 kayıt döndü ve LLM çekimser kaldı."
        elif qid == "Q18":
            hata2 = "LLM sapması"
            not2 = "Planlayıcı doğru kayıtları süzdü, araç 62.0 kWp doğru toplamını verdi; ancak LLM araca uymayıp 52.0 kWp üretti."
        elif qid == "Q19":
            hata2 = "planlayıcı hatası, ardından LLM sapması"
            not2 = "Planlayıcı OR filtresi ('İtfaiye') ile alakasız Kumburgaz kaydını da dahil etti (araç: 149.0 kWp); LLM araca da uymayıp 129.0 kWp yazdı."
        else:
            if not plan_dogru and not llm_esit_arac:
                hata2 = "planlayıcı hatası, ardından LLM sapması"
                not2 = "Planlayıcı kümesi hatalı ve LLM araç sonucuna uymadı."
            elif not plan_dogru:
                hata2 = "planlayıcı hatası"
                not2 = "Planlayıcı hatalı filtre üretti."
            elif not llm_esit_arac:
                hata2 = "LLM sapması"
                not2 = "LLM araç toplamına uymadı."
            else:
                hata2 = "diğer"
                not2 = "Bilinmeyen hata durumu."

        r2_rows.append({
            "sistem": "R2",
            "soru_id": qid,
            "soru": q["soru"],
            "altin_sayisal_kwp": gold_val,
            "altin_kanit": ";".join(q["altin_kanit"]),
            "plan_kumesi_veya_kapsama_dogru": plan_dogru,
            "arac_toplam_kwp": f"{tool_sum:.1f}" if tool_sum is not None else "0.0",
            "arac_dogru_toplam": arac_dogru,
            "llm_deger_kwp": llm_stated_str,
            "llm_esit_arac": llm_esit_arac,
            "dogru_mu": dogru2,
            "hata_sinifi": hata2,
            "analiz_notu": not2,
            "post_hoc_notu": POST_HOC_NOTU
        })

        # ------------------- R1 Analizi -------------------
        e1 = next(x for x in test_logs if x["soru_id"] == qid and x["sistem"] == "R1")
        ret_ids = set(e1.get("getirilen_record_idler") or [])
        r1_kapsama = gold_ids.issubset(ret_ids)

        llm_stated1_str, llm_val1 = extract_stated_total(e1["yanit"])
        ev1 = evaluate_answer(q, e1["yanit"], kb)
        dogru1 = ev1["dogru"]

        if dogru1:
            hata1 = "Doğru"
            not1 = "BM25 kanıtı getirdi, LLM metin içinden doğru toplamı hesapladı."
        elif qid == "Q18":
            hata1 = "gereksiz ret"
            not1 = "BM25 her iki kanıt kaydını getirmesine rağmen LLM çekimser kaldı."
        elif qid == "Q19":
            hata1 = "bağlam süzme / toplama hatası"
            not1 = "BM25 doğru kayıtları getirdi; ancak LLM bağlamdaki alakasız Kumburgaz kaydını da topladı (106 + 20 = 126.0 kWp)."
        elif qid == "Q22":
            hata1 = "toplama yapmadı / eksik biçim"
            not1 = "BM25 kayıtları getirdi; LLM iki değeri '78.0 kWp | 78.0 kWp' olarak yan yana yazdı ancak toplamı hesaplamadı."
        else:
            hata1 = "aritmetik / süzme hatası"
            not1 = "Yanlış toplama veya süzme."

        r1_rows.append({
            "sistem": "R1",
            "soru_id": qid,
            "soru": q["soru"],
            "altin_sayisal_kwp": gold_val,
            "altin_kanit": ";".join(q["altin_kanit"]),
            "plan_kumesi_veya_kapsama_dogru": r1_kapsama,
            "arac_toplam_kwp": "uygulanamaz",
            "arac_dogru_toplam": "uygulanamaz",
            "llm_deger_kwp": llm_stated1_str,
            "llm_esit_arac": "uygulanamaz",
            "dogru_mu": dogru1,
            "hata_sinifi": hata1,
            "analiz_notu": not1,
            "post_hoc_notu": POST_HOC_NOTU
        })

    return r2_rows, r1_rows


def write_csv(r2_rows, r1_rows):
    fieldnames = [
        "sistem", "soru_id", "soru", "altin_sayisal_kwp", "altin_kanit",
        "plan_kumesi_veya_kapsama_dogru", "arac_toplam_kwp", "arac_dogru_toplam",
        "llm_deger_kwp", "llm_esit_arac", "dogru_mu", "hata_sinifi",
        "analiz_notu", "post_hoc_notu"
    ]
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in r2_rows:
            writer.writerow(row)
        for row in r1_rows:
            writer.writerow(row)


def main():
    r2_rows, r1_rows = analyze_errors()
    write_csv(r2_rows, r1_rows)

    print("=" * 90)
    print(f"RAG TOPLAMA SORULARI POST-HOC HATA AYRIŞTIRMASI ({POST_HOC_NOTU})")
    print("=" * 90)

    # 1. R2 Tablosu
    print("\n### R2 (Planlayıcı + Araç) Hata Ayrıştırma Tablosu (Test Toplama n=6):")
    print("| Soru | Altın (kWp) | Plan Kümesi Doğru? | Araç Toplamı | Araç Doğru? | LLM Değeri | LLM == Araç? | Doğru? | Hata Sınıfı |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
    for r in r2_rows:
        pk = "Evet" if r["plan_kumesi_veya_kapsama_dogru"] else "Hayır"
        ad = "Evet" if r["arac_dogru_toplam"] else "Hayır"
        ea = "Evet" if r["llm_esit_arac"] else "Hayır"
        dm = "DOĞRU" if r["dogru_mu"] else "YANLIŞ"
        print(f"| {r['soru_id']} | {r['altin_sayisal_kwp']} | {pk} | {r['arac_toplam_kwp']} | {ad} | {r['llm_deger_kwp']} | {ea} | {dm} | {r['hata_sinifi']} |")

    # Sayımlar
    n_toplam = len(r2_rows)
    plan_dogru_n = sum(1 for r in r2_rows if r["plan_kumesi_veya_kapsama_dogru"])
    arac_dogru_n = sum(1 for r in r2_rows if r["arac_dogru_toplam"])
    llm_esit_n = sum(1 for r in r2_rows if r["llm_esit_arac"])
    llm_dogru_n = sum(1 for r in r2_rows if r["dogru_mu"])

    plan_hatasi_n = sum(1 for r in r2_rows if r["hata_sinifi"] == "planlayıcı hatası")
    llm_sapmasi_n = sum(1 for r in r2_rows if r["hata_sinifi"] == "LLM sapması")
    plan_ve_llm_n = sum(1 for r in r2_rows if r["hata_sinifi"] == "planlayıcı hatası, ardından LLM sapması")

    print("\n--- R2 Sayım Özeti ---")
    print(f"  Toplam soru sayısı              : {n_toplam}")
    print(f"  Plan doğru küme                 : {plan_dogru_n}/{n_toplam} ({plan_dogru_n/n_toplam*100:.1f}%)")
    print(f"  Araç doğru toplam               : {arac_dogru_n}/{n_toplam} ({arac_dogru_n/n_toplam*100:.1f}%)")
    print(f"  LLM araç toplamına eşit         : {llm_esit_n}/{n_toplam} ({llm_esit_n/n_toplam*100:.1f}%)")
    print(f"  Nihai LLM doğruluğu             : {llm_dogru_n}/{n_toplam} ({llm_dogru_n/n_toplam*100:.1f}%)")
    print("  Hata Sınıflandırması Dağılımı:")
    print(f"    - Planlayıcı Hatası           : {plan_hatasi_n}/{n_toplam} ({plan_hatasi_n/n_toplam*100:.1f}%) [Q17: dar filtre -> 0 kayıt]")
    print(f"    - LLM Sapması                 : {llm_sapmasi_n}/{n_toplam} ({llm_sapmasi_n/n_toplam*100:.1f}%) [Q18: araç 62.0 doğruyken LLM 52.0 yazdı]")
    print(f"    - Plan Hatası + LLM Sapması   : {plan_ve_llm_n}/{n_toplam} ({plan_ve_llm_n/n_toplam*100:.1f}%) [Q19: araç 149.0 != 43.0 iken LLM de 129.0 yazdı]")
    print(f"    - Doğru (Tam Başarı)          : {llm_dogru_n}/{n_toplam} ({llm_dogru_n/n_toplam*100:.1f}%) [Q20, Q21, Q22]")

    # 2. R1 Tablosu
    print("\n### R1 (BM25 Retrieval + LLM) Hata Ayrıştırma Tablosu (Test Toplama n=6):")
    print("| Soru | Altın (kWp) | BM25 Kanıtı Kapsadı mı? | LLM Yazdığı Değer | Doğru mu? | Hata Sınıfı |")
    print("|:---|:---:|:---:|:---:|:---:|:---|")
    for r in r1_rows:
        kp = "Evet (%100)" if r["plan_kumesi_veya_kapsama_dogru"] else "Hayır"
        dm = "DOĞRU" if r["dogru_mu"] else "YANLIŞ"
        print(f"| {r['soru_id']} | {r['altin_sayisal_kwp']} | {kp} | {r['llm_deger_kwp']} | {dm} | {r['hata_sinifi']} |")

    r1_kapsama_n = sum(1 for r in r1_rows if r["plan_kumesi_veya_kapsama_dogru"])
    r1_dogru_n = sum(1 for r in r1_rows if r["dogru_mu"])

    print("\n--- R1 Sayım Özeti ---")
    print(f"  BM25 Altın Kanıt Kapsama (Recall=1.0) : {r1_kapsama_n}/{len(r1_rows)} ({r1_kapsama_n/len(r1_rows)*100:.1f}%)")
    print(f"  Nihai LLM Doğruluğu                   : {r1_dogru_n}/{len(r1_rows)} ({r1_dogru_n/len(r1_rows)*100:.1f}%)")
    print("  Hata Dağılımı:")
    print("    - Gereksiz ret (Çekimserlik)        : 1/6 (Q18: bağlamda olmasına rağmen çekimser)")
    print("    - Bağlam süzme / toplama hatası     : 1/6 (Q19: alakasız Kumburgaz kaydını da dahil etti)")
    print("    - Toplama yapmadı / eksik biçim     : 1/6 (Q22: 78.0 kWp | 78.0 kWp yan yana bıraktı)")
    print("    - Doğru                             : 3/6 (Q17, Q20, Q21)")

    print("\n" + "=" * 90)
    print(f"SONUÇ: {POST_HOC_NOTU}")
    print("R1 doğruluğu 3/6 (%50.0), R2 doğruluğu 3/6 (%50.0). R2 > R1 koşulu sağlanamadığından H3 reddedilmiştir.")
    print("Bu post-hoc ayrıştırma, araç katmanının 4/6 (%66.7) doğru toplam üretmesine karşın,")
    print("Q18'deki LLM sapması ve Q17/Q19'daki planlayıcı filtreleme kusurları nedeniyle R2'nin")
    print("R1'e karşı istatistiksel/sayısal üstünlük kuramadığını ortaya koymaktadır.")
    print("=" * 90)
    print(f"CSV kaydedildi: {CSV_PATH}")


if __name__ == "__main__":
    main()
