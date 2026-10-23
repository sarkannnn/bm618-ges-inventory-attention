"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 9b — RAG değerlendirme (src/rag_eval.py)

Ölçütler (otomatik, kurallar yanıtlardan ÖNCE sabitlenmiştir):
  Recall@k          : BM25 ilk-k listesinde altın kanıt kayıtlarının oranı (yanıtlanabilir sorular; R1/R2)
  doğru yanıt       : tekil/toplama -> yanıttaki ilk VEYA son kWp değeri altın değere eşit (|fark|<0.005);
                      çelişki -> iki çelişen değerin ikisi de yanıtta (Q07 için iki kurum adı parçası);
                      yanıtlanamaz -> yanıt tam olarak "ÇEKİMSER".
  sayısal doğruluk  : yalnız toplama sorularında doğru yanıt oranı (+ ortalama mutlak sapma, sayı içeren yanıtlarda)
  atıf doğruluğu    : yanıtın atıf yaptığı (dosya yılı, Excel satırı) çiftlerinin altın kanıtla kesişim kesri
                      (kesinlik); yanıtlanabilir ve çekimser OLMAYAN yanıtlar üzerinde; atıf yoksa 0
  kanıt kapsamı     : altın kanıt çiftlerinin atıf yapılan çiftlerle kapsanma kesri (yanıtlanabilir sorular; çekimser=0)
  doğru çekimserlik : yanıtlanamaz sorularda yanıt TAM "ÇEKİMSER" olanların oranı (gevşek: içinde çekimser geçen)
  gereksiz ret      : yanıtlanabilir sorularda çekimser (içinde ÇEKİMSER geçen) yanıt oranı
  Paydası boş ölçüt "uygulanamaz" yazılır.
H3: toplama sorularında R2 doğru yanıt oranı > R1 doğru yanıt oranı (protokol: aksi halde reddedilir).

Komutlar:
  python src/rag_eval.py secim    # dev günlüğünden seçimi yapar, docs/rag_secim.md yazar
  python src/rag_eval.py rapor    # test günlüğünden metrikleri, soru bazlı tabloyu ve H3 kararını verir
"""

import os
import re
import sys
import json
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_index import tr_lower, load_kb_docs, build_index
from src.rag_systems import (load_gold, read_log, LOG_PATH, SECIM_MD, VARYANTLAR, SISTEM_MESAJI, PLAN_PROMPTU,
                             K_ADAYLARI, PROMPT_ADAYLARI, SECENEKLER, MODEL_ADI)

TOL = 0.005
# Q07 çelişkisi bir kurum adı değişimidir (güç artışı çelişki değildir); diğer çelişkilerde iki güç değeri aranır.
CELISKI_AD_ANAHTARI = {"Q07": ["kadın ve aile", "sosyal hizmetler"]}
UNAPPLICABLE = "uygulanamaz"


# ---------------------------------------------------------------------------
# Yanıt ayrıştırma
# ---------------------------------------------------------------------------
def _to_float(s):
    """'1.124,2' / '1,124.2' / '126,0' / '1124.2' / '1.124' -> float."""
    s = s.strip().strip(".,")
    if "." in s and "," in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        parts = s.split(sep)
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            s = "".join(parts)           # binlik ayıracı
        else:
            s = s.replace(",", ".")
    return float(s)


def parse_kwp_values(text):
    t = tr_lower(text)
    vals = []
    for m in re.finditer(r"(\d+(?:[.,]\d+)*)\s*kwp?\b", t):
        try:
            vals.append(_to_float(m.group(1)))
        except ValueError:
            pass
    return vals


def is_abstention_exact(text):
    if text is None:
        return False
    return text.strip().rstrip(".").strip() == "ÇEKİMSER"


def abstention_loose(text):
    return text is not None and "çekimser" in tr_lower(text)


_ROW_RE = re.compile(r"sat[ıi]r(?:ı|i|lar|ları|leri)?\b\s*[:=]?\s*(\d{1,3}(?:\s*(?:,|ve|ile|-)\s*\d{1,3}(?![\d.,]*\s*kw))*)(?![\d.,]*\s*kw)")
_ROW_DOT_RE = re.compile(r"(\d{1,3})\.\s*sat[ıi]r")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def parse_citations(text):
    """Yanıttan (dosya yılı, excel satırı) çiftleri çıkarır.
    Kural: her cümle/satır parçasındaki satır numaraları, aynı parçadaki en yakın yıla bağlanır;
    parçada yıl yoksa önceki parçalardaki son yıl kullanılır. REC_YYYY_Rn kodları doğrudan okunur."""
    t = tr_lower(text)
    cites = set()
    for m in re.finditer(r"rec_(\d{4})_r(\d+)", t):
        cites.add((int(m.group(1)), int(m.group(2))))
    last_year = None
    for seg in re.split(r"[\n;]|(?<=[a-zçğıöşü\)])\.\s", t):
        years = [(m.start(), int(m.group(1))) for m in _YEAR_RE.finditer(seg)]
        rows = []
        for m in _ROW_RE.finditer(seg):
            for n in re.findall(r"\d{1,3}", m.group(1)):
                rows.append((m.start(), int(n)))
        for m in _ROW_DOT_RE.finditer(seg):
            rows.append((m.start(), int(m.group(1))))
        for pos, row in rows:
            before = [y for y in years if y[0] < pos]
            if before:
                yr = max(before, key=lambda y: y[0])[1]       # satırdan önceki en yakın yıl
            elif years:
                yr = min(years, key=lambda y: abs(y[0] - pos))[1]  # önce yıl yoksa en yakın sonraki yıl
            elif last_year is not None:
                yr = last_year
            else:
                continue
            cites.add((yr, row))
        if years:
            last_year = years[-1][1]
    return cites


# ---------------------------------------------------------------------------
# Tek yanıt değerlendirme
# ---------------------------------------------------------------------------
def kb_by_id():
    return {d["record_id"]: d for d in load_kb_docs()}


def gold_pairs(q, kb):
    return {(int(kb[r]["file_year"]), int(kb[r]["excel_row"])) for r in q["altin_kanit"]}


def recall_at_k(gold_ids, retrieved_ids):
    if not gold_ids:
        return None
    return len(set(gold_ids) & set(retrieved_ids)) / len(set(gold_ids))


def evaluate_answer(q, answer, kb):
    tur = q["tur"]
    exact = is_abstention_exact(answer)
    loose = abstention_loose(answer)
    vals = parse_kwp_values(answer)
    res = {"tur": tur, "cekimser_tam": exact, "cekimser_gevsek": loose, "dogru": False,
           "atif_kesinlik": None, "kanit_kapsami": None, "talep_edilen": None, "mutlak_sapma": None}
    if tur == "yanıtlanamaz":
        res["dogru"] = exact
        return res
    gold_ids = q["altin_kanit"]
    # atıf ölçütleri (yanıtlanabilir sorular)
    gp = gold_pairs(q, kb)
    cp = parse_citations(answer) if not loose else set()
    res["kanit_kapsami"] = (len(gp & cp) / len(gp)) if gp else None
    if not loose:
        res["atif_kesinlik"] = (len(gp & cp) / len(cp)) if cp else 0.0
    if loose:
        return res
    if tur in ("tekil", "toplama"):
        gold_val = float(q["altin_sayisal"])
        if vals:
            res["talep_edilen"] = vals[-1]
            res["mutlak_sapma"] = min(abs(vals[0] - gold_val), abs(vals[-1] - gold_val))
            res["dogru"] = abs(vals[0] - gold_val) < TOL or abs(vals[-1] - gold_val) < TOL
    elif tur == "çelişki":
        if q["id"] in CELISKI_AD_ANAHTARI:
            tl = tr_lower(answer)
            res["dogru"] = all(a in tl for a in CELISKI_AD_ANAHTARI[q["id"]])
        else:
            gv = [float(kb[r]["power_kwp"]) for r in gold_ids]
            res["dogru"] = all(any(abs(v - g) < TOL for v in vals) for g in gv)
    return res


# ---------------------------------------------------------------------------
# Toplu metrikler
# ---------------------------------------------------------------------------
def _frac(num, den):
    if den == 0:
        return UNAPPLICABLE
    return f"{num}/{den} = {num / den:.3f}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _fmt(x):
    return UNAPPLICABLE if x is None else f"{x:.3f}"


def metrics_for(entries, gold_by_id, kb):
    """entries: aynı sistem (ve aynı yapılandırma) için kayıtlar. Sözlük döndürür."""
    rows = []
    for e in entries:
        q = gold_by_id[e["soru_id"]]
        ev = evaluate_answer(q, e["yanit"], kb)
        ev["soru_id"] = e["soru_id"]
        ev["retrieved"] = e.get("getirilen_record_idler", [])
        ev["recall"] = recall_at_k(q["altin_kanit"], ev["retrieved"]) if e["sistem"] != "R0" else None
        rows.append(ev)
    answerable = [r for r in rows if r["tur"] != "yanıtlanamaz"]
    unans = [r for r in rows if r["tur"] == "yanıtlanamaz"]
    topl = [r for r in rows if r["tur"] == "toplama"]
    answered = [r for r in answerable if not r["cekimser_gevsek"]]
    sistem = entries[0]["sistem"] if entries else None
    out = {
        "n": len(rows),
        "dogru_n": sum(r["dogru"] for r in rows),
        "dogru_n_tur": {t: (sum(r["dogru"] for r in rows if r["tur"] == t), sum(1 for r in rows if r["tur"] == t))
                        for t in ("tekil", "toplama", "çelişki", "yanıtlanamaz")},
        "recall": (None if sistem == "R0" else _mean([r["recall"] for r in answerable])),
        "sayisal_n": (sum(r["dogru"] for r in topl), len(topl)),
        "sayisal_mae": _mean([r["mutlak_sapma"] for r in topl]),
        "atif_kesinlik": _mean([r["atif_kesinlik"] for r in answered]),
        "kanit_kapsami": _mean([r["kanit_kapsami"] for r in answerable]),
        "cekimser_tam": (sum(r["cekimser_tam"] for r in unans), len(unans)),
        "cekimser_gevsek": (sum(r["cekimser_gevsek"] for r in unans), len(unans)),
        "gereksiz_ret": (sum(r["cekimser_gevsek"] for r in answerable), len(answerable)),
        "rows": rows,
    }
    return out


def metrics_text(m):
    d = m["dogru_n_tur"]
    return {
        "doğru yanıt": _frac(m["dogru_n"], m["n"]),
        "  tekil": _frac(*d["tekil"]), "  toplama": _frac(*d["toplama"]),
        "  çelişki": _frac(*d["çelişki"]), "  yanıtlanamaz": _frac(*d["yanıtlanamaz"]),
        "Recall@k": _fmt(m["recall"]) if m["recall"] is not None else UNAPPLICABLE,
        "sayısal doğruluk (toplama)": _frac(*m["sayisal_n"]),
        "sayısal ort. mutlak sapma (kWp)": _fmt(m["sayisal_mae"]),
        "atıf doğruluğu": _fmt(m["atif_kesinlik"]),
        "kanıt kapsamı": _fmt(m["kanit_kapsami"]),
        "doğru çekimserlik (tam biçim)": _frac(*m["cekimser_tam"]),
        "doğru çekimserlik (gevşek)": _frac(*m["cekimser_gevsek"]),
        "gereksiz ret": _frac(*m["gereksiz_ret"]),
    }


# ---------------------------------------------------------------------------
# Dev seçimi
# ---------------------------------------------------------------------------
SECIM_KRITERI = (
    "Seçim ölçütü (dev günlüğü okunmadan ÖNCE sabitlendi): (prompt, k) çiftleri arasından R1 ve R2'nin dev "
    "doğru yanıt sayıları toplamı (en çok 20) en yüksek olan seçilir. Eşitlikte sırasıyla: (1) R1+R2 atıf "
    "doğruluğu ortalaması yüksek olan, (2) küçük k, (3) küçük prompt numarası. Yalnız 10 dev sorusu kullanılır."
)


def dev_tablo(entries=None):
    entries = entries if entries is not None else read_log()
    gold_by_id = {q["id"]: q for q in load_gold()}
    kb = kb_by_id()
    dev = [e for e in entries if e.get("faz") == "dev_secim"]
    for e in dev:
        assert gold_by_id[e["soru_id"]]["split"] == "dev", "dev günlüğünde test sorusu var!"
    tablo = []
    for p in PROMPT_ADAYLARI:
        for k in K_ADAYLARI:
            r = {"prompt": p, "k": k}
            for s in ("R1", "R2"):
                es = [e for e in dev if e["sistem"] == s and e["prompt"] == p and e["k"] == k]
                r[s] = metrics_for(es, gold_by_id, kb) if len(es) == 10 else None
            tablo.append(r)
    return tablo


def sec(tablo):
    adaylar = [r for r in tablo if r["R1"] is not None and r["R2"] is not None]
    assert len(adaylar) == 9, f"9 yapılandırma bekleniyordu, {len(adaylar)} tam yapılandırma var."

    def key(r):
        atif = _mean([r["R1"]["atif_kesinlik"], r["R2"]["atif_kesinlik"]]) or 0.0
        return (-(r["R1"]["dogru_n"] + r["R2"]["dogru_n"]), -atif, r["k"], r["prompt"])

    return sorted(adaylar, key=key)[0]


def secim_yaz(md_path=None):
    tablo = dev_tablo()
    secilen = sec(tablo)
    gold = [q for q in load_gold() if q["split"] == "dev"]
    kb = kb_by_id()
    idx = build_index()
    rec = {}
    for k in K_ADAYLARI:
        vals = [recall_at_k(q["altin_kanit"], [r for r, _ in idx.search(q["soru"], k)])
                for q in gold if q["tur"] != "yanıtlanamaz"]
        rec[k] = sum(vals) / len(vals)
    L = []
    L.append("# RAG Prompt ve k Seçimi (Adım 9b) — YALNIZ 10 dev sorusu")
    L.append("")
    L.append("Bu belge, 20 test sorusu çalıştırılmadan ÖNCE yazılmış ve SHA-256'sı `docs/protocol_freeze.txt` içine "
             "eklenmiştir. Test soruları (Q11–Q30) bu aşamada hiçbir sistemle çalıştırılmamıştır.")
    L.append("")
    L.append("## Seçim")
    L.append("")
    L.append(f"**Seçilen prompt: {secilen['prompt']}, k = {secilen['k']}**")
    L.append("")
    L.append("```json")
    L.append(json.dumps({"prompt": secilen["prompt"], "k": secilen["k"]}, ensure_ascii=False))
    L.append("```")
    L.append("")
    L.append(SECIM_KRITERI)
    L.append("")
    L.append("## Dev sonuçları (10 dev soru; R1 ve R2; doğru yanıt sayısı / 10)")
    L.append("")
    L.append("| prompt | k | R1 doğru | R2 doğru | R1+R2 | R1 atıf | R2 atıf | R1 gereksiz ret | R2 gereksiz ret | R1 çekimserlik | R2 çekimserlik |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in tablo:
        a, b = r["R1"], r["R2"]
        mark = " **(seçilen)**" if (r["prompt"], r["k"]) == (secilen["prompt"], secilen["k"]) else ""
        L.append(f"| {r['prompt']}{mark} | {r['k']} | {a['dogru_n']}/10 | {b['dogru_n']}/10 | {a['dogru_n'] + b['dogru_n']}/20 | "
                 f"{_fmt(a['atif_kesinlik'])} | {_fmt(b['atif_kesinlik'])} | "
                 f"{a['gereksiz_ret'][0]}/{a['gereksiz_ret'][1]} | {b['gereksiz_ret'][0]}/{b['gereksiz_ret'][1]} | "
                 f"{a['cekimser_tam'][0]}/{a['cekimser_tam'][1]} | {b['cekimser_tam'][0]}/{b['cekimser_tam'][1]} |")
    L.append("")
    L.append("## Dev soru bazlı doğruluk (seçilen yapılandırma)")
    L.append("")
    L.append("| soru | tür | R1 | R2 |")
    L.append("|---|---|---|---|")
    ra = {x["soru_id"]: x for x in secilen["R1"]["rows"]}
    rb = {x["soru_id"]: x for x in secilen["R2"]["rows"]}
    for q in gold:
        L.append(f"| {q['id']} | {q['tur']} | {'✓' if ra[q['id']]['dogru'] else '✗'} | {'✓' if rb[q['id']]['dogru'] else '✗'} |")
    L.append("")
    L.append("## Yalnız getirme (LLM'siz) dev Recall@k — yanıtlanabilir 8 dev soru")
    L.append("")
    L.append("| k | Recall@k |")
    L.append("|---|---|")
    for k in K_ADAYLARI:
        L.append(f"| {k} | {rec[k]:.3f} |")
    L.append("")
    L.append("## Sabit bileşenler")
    L.append("")
    L.append(f"- Model: `{MODEL_ADI}` (Q4_K_M), Ollama, CPU. Ayarlar: `{json.dumps(SECENEKLER)}`")
    L.append("- Sistem mesajı (tüm sistemlerde aynı, Ollama'nın varsayılan Qwen mesajı değiştirilir):")
    L.append("")
    L.append(f"> {SISTEM_MESAJI}")
    L.append("")
    L.append("- Prompt varyantları (yalnız kullanıcı mesajındaki talimat farklıdır):")
    L.append("")
    for p in PROMPT_ADAYLARI:
        v = VARYANTLAR[p]
        L.append(f"  - **{p}** — {v['aciklama']}")
        L.append(f"    - talimat: {v['talimat'] if v['talimat'] else '(yok)'}")
        L.append(f"    - R2 ek talimat: {v['arac_ek'] if v['arac_ek'] else '(yok)'}")
    L.append("")
    L.append("- Aday k değerleri: 3, 5, 10. Getirme: kendi yazdığımız BM25 (k1=1.5, b=0.75), sorgu = soru metni.")
    L.append("")
    out = md_path or SECIM_MD
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    return secilen, tablo


# ---------------------------------------------------------------------------
# Test raporu
# ---------------------------------------------------------------------------
def rapor():
    entries = [e for e in read_log() if e.get("faz") == "test"]
    gold_by_id = {q["id"]: q for q in load_gold()}
    kb = kb_by_id()
    assert len(entries) == 60, f"60 test kaydı bekleniyordu, {len(entries)} var."
    by_sys = {s: [e for e in entries if e["sistem"] == s] for s in ("R0", "R1", "R2")}
    M = {s: metrics_for(es, gold_by_id, kb) for s, es in by_sys.items()}
    print("=" * 100)
    print("TEST (20 soru) METRİKLERİ — R0 / R1 / R2")
    print("=" * 100)
    T = {s: metrics_text(M[s]) for s in M}
    print(f"{'ölçüt':<38}{'R0':<26}{'R1':<26}{'R2':<26}")
    for key in T["R0"]:
        print(f"{key:<38}{T['R0'][key]:<26}{T['R1'][key]:<26}{T['R2'][key]:<26}")
    # soru bazlı
    print("\nSORU BAZLI TABLO (✓ doğru, ✗ yanlış; Rec = getirme Recall@k)")
    rows_out = []
    for q in [x for x in load_gold() if x["split"] == "test"]:
        rec = {s: next(r for r in M[s]["rows"] if r["soru_id"] == q["id"]) for s in M}
        print(f"{q['id']} [{q['tur']:<12}] R0:{'✓' if rec['R0']['dogru'] else '✗'}  R1:{'✓' if rec['R1']['dogru'] else '✗'}"
              f" (Rec {_fmt(rec['R1']['recall'])})  R2:{'✓' if rec['R2']['dogru'] else '✗'}")
        for s in M:
            e = next(x for x in by_sys[s] if x["soru_id"] == q["id"])
            rows_out.append({"soru_id": q["id"], "tur": q["tur"], "sistem": s, "dogru": int(rec[s]["dogru"]),
                             "recall": rec[s]["recall"], "atif_kesinlik": rec[s]["atif_kesinlik"],
                             "kanit_kapsami": rec[s]["kanit_kapsami"], "cekimser_tam": int(rec[s]["cekimser_tam"]),
                             "sure_sn": e["sure_sn"], "altin": q["altin_yanit"], "yanit": e["yanit"]})
    # H3
    r1 = M["R1"]["dogru_n_tur"]["toplama"]
    r2 = M["R2"]["dogru_n_tur"]["toplama"]
    h3 = r2[0] > r1[0]
    print("\nH3 (toplama): R1 doğru = %d/%d, R2 doğru = %d/%d -> %s" % (
        r1[0], r1[1], r2[0], r2[1],
        "R2 > R1: H3 reddedilmez (desteklendi)" if h3 else "R2 > R1 DEĞİL: H3 REDDEDİLİR"))
    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rag_soru_bazli.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    with open(os.path.join(out_dir, "rag_metrics.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["olcut", "R0", "R1", "R2"])
        for key in T["R0"]:
            w.writerow([key.strip(), T["R0"][key], T["R1"][key], T["R2"][key]])
        w.writerow(["H3_toplama_R2_gt_R1", "", "", "evet" if h3 else "hayır"])
    return M, h3


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "secim":
        s, t = secim_yaz()
        print(f"Seçilen: prompt={s['prompt']} k={s['k']} -> {SECIM_MD}")
    elif cmd == "rapor":
        rapor()
    else:
        print("Kullanım: python src/rag_eval.py [secim|rapor]")
