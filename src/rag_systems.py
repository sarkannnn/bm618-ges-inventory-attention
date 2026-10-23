"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 9b — RAG sistemleri (src/rag_systems.py)

Sistemler (hepsi AYNI LLM, AYNI ayar, AYNI sistem mesajı):
  R0: kanıtsız (yalnızca soru)
  R1: BM25 ile ilk-k belge bağlama konur
  R2: R1 + araç (LLM yalnız JSON plan üretir; kod doğrular ve filtrele/say/topla çalıştırır;
      araç sonucu, filtrelenen satır sayısı (kapsam) ile birlikte bağlama konur)

LLM: Ollama, qwen2.5:3b-instruct (Q4_K_M), CPU. Ayarlar sabit:
  temperature=0, seed=42, num_ctx=8192, num_predict=400.
Ollama'nın varsayılan Qwen sistem mesajı ("You are Qwen ...") her çağrıda kendi sistem
mesajımızla DEĞİŞTİRİLİR (messages[0] rolü system olarak açıkça gönderilir).

Komutlar:
  python src/rag_systems.py dev      # yalnız 10 dev sorusu, 3 prompt x k{3,5,10} x {R1,R2}
  python src/rag_systems.py freeze   # docs/rag_secim.md SHA-256'sını protocol_freeze.txt'ye ekler
  python src/rag_systems.py test     # 20 test sorusu x {R0,R1,R2}, TEK KEZ (donduruldan sonra)

Test soruları dev aşamasında ASLA çağrılmaz: dev koşucusu soru listesini yapısal olarak
split == "dev" ile süzer ve test koşucusu dondurma kaydı yoksa çalışmayı reddeder.
"""

import os
import re
import sys
import json
import time
import hashlib
import datetime

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_index import build_index, load_kb_docs
from src.rag_tools import (PlanError, parse_plan_text, run_tool, load_kb_df, format_tool_result)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_ADI = "qwen2.5:3b-instruct"
SECENEKLER = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 400}

GOLD_PATH = os.path.join(PROJECT_ROOT, "data", "gold_questions.jsonl")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "rag_runs.jsonl")
SECIM_MD = os.path.join(PROJECT_ROOT, "docs", "rag_secim.md")
FREEZE_TXT = os.path.join(PROJECT_ROOT, "docs", "protocol_freeze.txt")

K_ADAYLARI = (3, 5, 10)
PROMPT_ADAYLARI = ("P1", "P2", "P3")

SISTEM_MESAJI = ("Yalnızca Türkçe yanıt ver. Yalnızca verilen kanıta dayan. Dosya yılı ve Excel satırı belirt. "
                 "Çelişki varsa iki değeri de yaz. Kanıt yoksa tam olarak ÇEKİMSER yaz.")

# En fazla 3 prompt varyantı. Sistem mesajı HEP aynıdır; varyantlar yalnız kullanıcı mesajındaki talimatı değiştirir.
VARYANTLAR = {
    "P1": {
        "aciklama": "Yalın: kanıt + soru, ek talimat yok",
        "talimat": "",
        "arac_ek": "",
    },
    "P2": {
        "aciklama": "Açık yanıt biçimi: eşleşen kayıttan değeri kWp olarak yaz, parantezli atıf (ÇEKİMSER kuralı yalnız sistem mesajında)",
        "talimat": ("Soruyu yukarıdaki kanıta göre yanıtla. Kanıtta sorudaki dosya yılı ve ilçe veya tesis adıyla eşleşen kayıt varsa "
                    "değeri kWp olarak yaz ve her değer için dosya yılını ve Excel satırını parantez içinde belirt, "
                    "örneğin (2022 dosyası, Excel satırı 12)."),
        "arac_ek": (" Araç sonucu varsa toplamı ve sayıyı aynen araç sonucundan al, kendin hesaplama; "
                    "araç sonucundaki filtrelenen satır sayısını yanıtında belirt."),
    },
    "P3": {
        "aciklama": "Adım adım: önce eşleşen satırları seç, sonra topla ve atıf yap (ÇEKİMSER kuralı yalnız sistem mesajında)",
        "talimat": ("Şu adımları izle: (1) Kanıttaki satırlardan dosya yılı ve ilçesi veya tesis adı sorudakiyle uyuşanları seç. "
                    "(2) Toplam isteniyorsa seçtiğin tüm satırların gücünü topla. "
                    "(3) Aynı tesis için farklı değerler veren satırlar varsa her iki değeri de yaz. "
                    "(4) Yanıtta dosya yılını ve Excel satırını belirt."),
        "arac_ek": (" Araç sonucu verildiyse toplam ve sayım için yalnızca araç sonucunu kullan (kendin toplama yapma) "
                    "ve filtrelenen satır sayısını yanıtında belirt."),
    },
}

# R2 araç planlayıcısı: tüm dev/test sorularından bağımsız, uydurma örneklerle (Pendik/Esenler/Maltepe).
PLAN_PROMPTU = (
    "Aşağıdaki soruyu yanıtlamak için envanter tablosunda yapılacak işlemi yalnızca JSON olarak yaz.\n"
    "Biçim: {\"islem\": \"topla\" veya \"say\" veya \"filtrele\", \"ilce\": metin veya metin listesi veya null, "
    "\"yil\": 4 haneli sayı veya null, \"ad_icerir\": metin veya metin listesi veya null}\n"
    "Kurallar:\n"
    "- islem: toplam güç isteniyorsa \"topla\"; kaç kayıt veya tesis olduğu soruluyorsa \"say\"; "
    "diğer durumlarda (tek tesis, karşılaştırma, listeleme) \"filtrele\".\n"
    "- ilce: yalnızca ilçe adı (ör. \"Pendik\"); soruda ilçe yoksa null.\n"
    "- yil: sorudaki dosya yılı; soruda yıl yoksa veya birden çok yıl varsa null.\n"
    "- ad_icerir: tesis adında geçen ayırt edici sözcük veya sözcükler; yoksa null.\n"
    "- Başka anahtar veya işlem yazma. Yalnızca tek bir JSON nesnesi yaz.\n"
    "Örnekler:\n"
    "Soru: 2025 dosyasında Pendik ilçesinde kayıtlı GES tesislerinin toplam gücü kaç kWp'dir?\n"
    "{\"islem\": \"topla\", \"ilce\": \"Pendik\", \"yil\": 2025, \"ad_icerir\": null}\n"
    "Soru: 2023 dosyasında Esenler ilçesinde kaç GES kaydı vardır?\n"
    "{\"islem\": \"say\", \"ilce\": \"Esenler\", \"yil\": 2023, \"ad_icerir\": null}\n"
    "Soru: 2024 dosyasında Maltepe Belediye Hizmet Binası GES tesisinin gücü nedir?\n"
    "{\"islem\": \"filtrele\", \"ilce\": \"Maltepe\", \"yil\": 2024, \"ad_icerir\": [\"Hizmet Binası\"]}\n"
    "Soru: Pendik Sahil Parkı GES tesisinin 2022 ve 2025 dosyalarındaki kayıtları nelerdir?\n"
    "{\"islem\": \"filtrele\", \"ilce\": \"Pendik\", \"yil\": null, \"ad_icerir\": [\"Sahil Parkı\"]}\n"
    "Soru: {soru}\n"
)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_gold(path=None):
    qs = []
    with open(path or GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qs.append(json.loads(line))
    return qs


def read_log(path=None):
    path = path or LOG_PATH
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def append_log(entry, path=None):
    path = path or LOG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()


def chat(messages, json_format=False, timeout=900):
    """Ollama /api/chat çağrısı. Sistem mesajı messages[0] (role=system) olarak gönderilir."""
    body = {"model": MODEL_ADI, "messages": messages, "stream": False,
            "keep_alive": "60m", "options": dict(SECENEKLER)}
    if json_format:
        body["format"] = "json"
    t0 = time.perf_counter()
    r = requests.post(OLLAMA_URL, json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return {"content": j["message"]["content"],
            "sure_sn": round(time.perf_counter() - t0, 2),
            "prompt_tokens": j.get("prompt_eval_count"),
            "eval_tokens": j.get("eval_count")}


def mesajlar(kullanici_metni):
    return [{"role": "system", "content": SISTEM_MESAJI}, {"role": "user", "content": kullanici_metni}]


def belge_satiri(doc):
    return (f"- Dosya yılı: {doc['file_year']} | Excel satırı: {doc['excel_row']} | İlçe: {doc['district_clean']} | "
            f"Tesis: {doc['name_clean']} | Güç: {doc['power_kwp']} kWp")


def kanit_metni(docs):
    if not docs:
        return "(Eşleşen kanıt bulunamadı.)"
    return "\n".join(belge_satiri(d) for d in docs)


def kullanici_mesaji(prompt_id, soru, kanit, arac_metni, sistem):
    v = VARYANTLAR[prompt_id]
    parts = ["KANIT:", kanit, ""]
    if arac_metni is not None:
        parts += ["ARAÇ SONUCU (kod hesapladı):", arac_metni, ""]
    parts += [f"SORU: {soru}"]
    talimat = v["talimat"] + (v["arac_ek"] if sistem == "R2" else "")
    if talimat.strip():
        parts += ["", talimat.strip()]
    return "\n".join(parts)


class Ortam:
    """İndeks + belge sözlüğü + araç DataFrame'i (yalnız kb_docs.jsonl'dan)."""

    def __init__(self):
        self.docs = load_kb_docs()
        self.docs_by_id = {d["record_id"]: d for d in self.docs}
        self.index = build_index()
        self.df = load_kb_df()


def plan_uret(soru, chat_fn):
    """LLM'den JSON plan alır (kod çalıştırılmaz); doğrulanamazsa plan_hata döner."""
    cevap = chat_fn(mesajlar(PLAN_PROMPTU.replace("{soru}", soru)), json_format=True)
    ham = cevap["content"]
    info = {"ham": ham, "plan": None, "hata": None}
    try:
        obj = parse_plan_text(ham)
        from src.rag_tools import validate_plan
        info["plan"] = validate_plan(obj)
    except PlanError as e:
        info["hata"] = str(e)
    return info, cevap


def sistem_calistir(sistem, q, prompt_id, k, ortam, chat_fn=chat, plan_onbellek=None):
    """Tek soru için R0/R1/R2 yanıtı üretir ve günlük kaydı sözlüğü döndürür (yazmaz)."""
    t0 = time.perf_counter()
    soru = q["soru"]
    llm_cagri = []
    retrieved, arac, sorgu_plani = [], None, None

    if sistem == "R0":
        kanit = "(Kanıt verilmedi.)"
        user = kullanici_mesaji(prompt_id, soru, kanit, None, "R0")
        sorgu_plani = None
    else:
        hits = ortam.index.search(soru, k)
        retrieved = [rid for rid, _ in hits]
        kanit = kanit_metni([ortam.docs_by_id[r] for r in retrieved])
        arac_metni = None
        if sistem == "R1":
            sorgu_plani = {"yontem": "BM25", "sorgu": soru, "k": k}
        else:  # R2
            if plan_onbellek is not None and soru in plan_onbellek:
                info, pc = plan_onbellek[soru]
                plan_onbellekten = True
            else:
                info, pc = plan_uret(soru, chat_fn)
                plan_onbellekten = False
                if plan_onbellek is not None:
                    plan_onbellek[soru] = (info, pc)
            llm_cagri.append({"rol": "plan", "sure_sn": pc["sure_sn"], "prompt_tokens": pc["prompt_tokens"],
                              "eval_tokens": pc["eval_tokens"], "onbellekten": plan_onbellekten})
            sorgu_plani = {"yontem": "BM25+arac", "sorgu": soru, "k": k, "llm_plan_ham": info["ham"],
                           "llm_plan_dogrulanmis": info["plan"], "plan_hata": info["hata"]}
            if info["plan"] is not None:
                arac = run_tool(info["plan"], ortam.df)
                arac_metni = format_tool_result(arac)
            else:
                arac_metni = "Araç çalıştırılamadı (geçersiz plan)."
        user = kullanici_mesaji(prompt_id, soru, kanit, arac_metni, sistem)

    c = chat_fn(mesajlar(user))
    llm_cagri.append({"rol": "yanit", "sure_sn": c["sure_sn"], "prompt_tokens": c["prompt_tokens"],
                      "eval_tokens": c["eval_tokens"], "onbellekten": False})
    sure = sum(x["sure_sn"] for x in llm_cagri)  # planlayıcı süresi onbellekten gelse de R2 süresine eklenir
    return {
        "soru_id": q["id"], "split": q["split"], "sistem": sistem, "prompt": prompt_id,
        "k": (None if sistem == "R0" else k),
        "sorgu_plani": sorgu_plani, "getirilen_record_idler": retrieved,
        "arac_sonucu": ({kk: vv for kk, vv in arac.items() if kk != "kayitlar"} if arac else None),
        "yanit": c["content"], "sure_sn": round(sure, 2),
        "duvar_sure_sn": round(time.perf_counter() - t0, 2),
        "llm_cagri": llm_cagri, "model": MODEL_ADI, "ayarlar": dict(SECENEKLER),
    }


def _done_keys(entries, faz):
    return {(e["soru_id"], e["sistem"], e.get("prompt"), e.get("k")) for e in entries
            if e.get("faz") == faz and "soru_id" in e}


def dev_izgarasi(chat_fn=chat, log_path=None, ks=K_ADAYLARI, prompts=PROMPT_ADAYLARI, ortam=None, verbose=True):
    """YALNIZ dev soruları: prompt x k x {R1,R2}. Dev soru listesi yapısal olarak split=='dev' ile süzülür."""
    gold = [q for q in load_gold() if q["split"] == "dev"]
    assert len(gold) == 10 and all(q["split"] == "dev" for q in gold)
    ortam = ortam or Ortam()
    log_path = log_path or LOG_PATH
    done = _done_keys(read_log(log_path), "dev_secim")
    plan_cache = {}
    n = 0
    for p in prompts:
        for k in ks:
            for q in gold:
                for sistem in ("R1", "R2"):
                    if (q["id"], sistem, p, k) in done:
                        continue
                    e = sistem_calistir(sistem, q, p, k, ortam, chat_fn, plan_cache)
                    e["faz"] = "dev_secim"
                    e["zaman_utc"] = _utc_now()
                    append_log(e, log_path)
                    n += 1
                    if verbose:
                        print(f"[dev] {p} k={k} {q['id']} {sistem} {e['sure_sn']}sn | {e['yanit'][:90]!r}", flush=True)
    return n


def secim_oku(md_path=None):
    """docs/rag_secim.md içindeki ```json bloğundan seçilen {prompt, k} değerini okur."""
    with open(md_path or SECIM_MD, "r", encoding="utf-8") as f:
        t = f.read()
    m = re.search(r"```json\s*(\{.*?\})\s*```", t, flags=re.DOTALL)
    assert m, "rag_secim.md içinde ```json seçim bloğu bulunamadı."
    s = json.loads(m.group(1))
    assert s["prompt"] in VARYANTLAR and int(s["k"]) in K_ADAYLARI
    return s["prompt"], int(s["k"])


def dondur(freeze_path=None, log_path=None, md_path=None):
    """rag_secim.md SHA-256'sını protocol_freeze.txt'ye (gerçek UTC ile, SONA EKLEYEREK) yazar ve günlüğe olay ekler."""
    md_path = md_path or SECIM_MD
    freeze_path = freeze_path or FREEZE_TXT
    log_path = log_path or LOG_PATH
    assert os.path.exists(md_path), "docs/rag_secim.md yok; önce `python src/rag_eval.py secim` çalıştırın."
    if any(e.get("olay") == "secim_donduruldu" for e in read_log(log_path)):
        raise RuntimeError("Seçim zaten donduruldu (günlükte secim_donduruldu olayı var).")
    h = sha256_file(md_path)
    now = _utc_now()
    prompt_id, k = secim_oku(md_path)
    blok = (
        "\r\n================================================================================\r\n"
        "DONDURMA GÜNCELLEMESİ (VERSION 1.4 FREEZE UPDATE - ADIM 9b RAG PROMPT VE k SEÇİMİ)\r\n"
        "================================================================================\r\n"
        f"Dondurma Zamanı (UTC): {now}\r\n"
        "Sürüm: 1.4 (Kilitli / Frozen)\r\n"
        "Bağlayıcı İlke: prompt varyantı ve k YALNIZ 10 dev sorusuyla seçildi; 20 test sorusu bu dondurmadan\r\n"
        "ÖNCE hiçbir RAG sistemiyle (R0, R1, R2) çalıştırılmadı. Test soruları dondurmadan sonra TEK KEZ çalıştırılacaktır.\r\n"
        f"Seçilen: prompt={prompt_id}, k={k}\r\n"
        "\r\n"
        "KİLİTLEYİCİ SAĞLAMA ÖZETİ (v1.4):\r\n"
        "--------------------------------------------------------------------------------\r\n"
        "Dosya: docs/rag_secim.md [KİLİTLEYİCİ HASH]\r\n"
        f"SHA-256: {h}\r\n"
        "================================================================================\r\n"
    )
    with open(freeze_path, "ab") as f:
        f.write(blok.encode("utf-8"))
    append_log({"olay": "secim_donduruldu", "sha256": h, "zaman_utc": now, "prompt": prompt_id, "k": k}, log_path)
    return h, now


def test_kosusu(chat_fn=chat, log_path=None, md_path=None, freeze_path=None, ortam=None, verbose=True):
    """20 test sorusu x {R0,R1,R2}; yalnızca seçim dondurulduktan sonra; her (soru,sistem) TEK KEZ."""
    log_path = log_path or LOG_PATH
    md_path = md_path or SECIM_MD
    freeze_path = freeze_path or FREEZE_TXT
    entries = read_log(log_path)
    olaylar = [e for e in entries if e.get("olay") == "secim_donduruldu"]
    if len(olaylar) != 1:
        raise RuntimeError("Test koşusu reddedildi: günlükte tam 1 'secim_donduruldu' olayı olmalı.")
    h = sha256_file(md_path)
    if olaylar[0]["sha256"] != h:
        raise RuntimeError("Test koşusu reddedildi: rag_secim.md dondurulduktan sonra değişmiş.")
    with open(freeze_path, "r", encoding="utf-8") as f:
        if h not in f.read():
            raise RuntimeError("Test koşusu reddedildi: SHA-256 protocol_freeze.txt içinde yok.")
    prompt_id, k = secim_oku(md_path)
    gold = [q for q in load_gold() if q["split"] == "test"]
    assert len(gold) == 20
    ortam = ortam or Ortam()
    done = {(e["soru_id"], e["sistem"]) for e in entries if e.get("faz") == "test"}
    if len(done) >= 60:
        raise RuntimeError("Test koşusu reddedildi: 60 test kaydı zaten var (TEK KEZ kuralı).")
    n = 0
    for q in gold:
        for sistem in ("R0", "R1", "R2"):
            if (q["id"], sistem) in done:
                continue
            e = sistem_calistir(sistem, q, prompt_id, k, ortam, chat_fn, None)
            e["faz"] = "test"
            e["zaman_utc"] = _utc_now()
            append_log(e, log_path)
            n += 1
            if verbose:
                print(f"[test] {q['id']} {sistem} {e['sure_sn']}sn | {e['yanit'][:90]!r}", flush=True)
    return n


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "dev":
        print(f"{dev_izgarasi()} yeni dev kaydı yazıldı.")
    elif cmd == "freeze":
        h, t = dondur()
        print(f"Donduruldu: sha256={h} utc={t}")
    elif cmd == "test":
        print(f"{test_kosusu()} yeni test kaydı yazıldı.")
    else:
        print("Kullanım: python src/rag_systems.py [dev|freeze|test]")
