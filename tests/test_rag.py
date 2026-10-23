"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 9b — RAG testleri (tests/test_rag.py)

(a) Araç yalnız izinli işlem (filtrele, say, topla) çalıştırıyor; keyfî kod/işlem reddediliyor.
(b) Test soruları seçim aşamasında çağrılmadı: yapısal (dev koşucusu yalnız dev) + günlük sırası
    (dondurma olayından önce hiçbir test kaydı yok; hash protocol_freeze.txt'de; test koşusu TEK KEZ).
(c) R2 toplamı pandas ile aynı (bağımsız pandas hesabı; günlükteki araç sonuçları dahil).
(d) Çekimser yanıt biçimi: tam olarak "ÇEKİMSER"; sistem mesajı ve ayarlar tüm çağrılarda sabit.
(e) Yanıt ayrıştırıcıları (kWp değeri, atıf).

LLM ÇAĞRILMAZ: sahte (fake) sohbet işlevi ve sahte HTTP kullanılır.
"""

import os
import ast
import sys
import json
import hashlib
import tempfile
import datetime

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import rag_tools, rag_systems, rag_eval
from src.rag_tools import PlanError, run_tool, load_kb_df, ALLOWED_OPS, TOOL_FUNCS, validate_plan, parse_plan_text
from src.rag_systems import (load_gold, read_log, SISTEM_MESAJI, SECENEKLER, LOG_PATH, SECIM_MD, FREEZE_TXT,
                             OLLAMA_URL, MODEL_ADI)
from src.rag_index import tr_lower

KB_PATH = os.path.join(PROJECT_ROOT, "data", "kb_docs.jsonl")

# Araç doğruluğu için elle yazılmış ideal planlar (LLM yok): 9 toplama sorusu (dev + test).
IDEAL_PLANS = {
    "Q04": {"islem": "topla", "ilce": "Bakırköy", "yil": 2022, "ad_icerir": None},
    "Q05": {"islem": "topla", "ilce": "Silivri", "yil": 2023, "ad_icerir": None},
    "Q06": {"islem": "topla", "ilce": "Fatih", "yil": 2022, "ad_icerir": None},
    "Q17": {"islem": "topla", "ilce": ["Şişli", "Zeytinburnu"], "yil": 2023, "ad_icerir": ["Deprem Parkı"]},
    "Q18": {"islem": "topla", "ilce": "Beykoz", "yil": 2024, "ad_icerir": None},
    "Q19": {"islem": "topla", "ilce": "Büyükçekmece", "yil": 2024, "ad_icerir": ["Mimar Sinan", "Büyükçekmece İtfaiye"]},
    "Q20": {"islem": "topla", "ilce": "Kartal", "yil": 2024, "ad_icerir": ["Yakacık", "Soğanlık"]},
    "Q21": {"islem": "topla", "ilce": "Başakşehir", "yil": 2022, "ad_icerir": ["Engelliler Merkezi"]},
    "Q22": {"islem": "topla", "ilce": "Eyüpsultan", "yil": 2024, "ad_icerir": ["Tevfik Aydeniz", "Yeşilpınar"]},
}


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _pandas_filter(df_raw, plan):
    """Araç koduna HİÇ bağlı olmayan bağımsız pandas filtresi (ham kb_docs.jsonl üzerinde)."""
    m = pd.Series(True, index=df_raw.index)
    if plan.get("yil") is not None:
        m &= df_raw["file_year"] == int(plan["yil"])
    il = plan.get("ilce")
    if il:
        il = [il] if isinstance(il, str) else il
        wanted = {tr_lower(x) for x in il}
        m &= df_raw["district_clean"].map(tr_lower).isin(wanted)
    ad = plan.get("ad_icerir")
    if ad:
        ad = [ad] if isinstance(ad, str) else ad
        sub = pd.Series(False, index=df_raw.index)
        for a in ad:
            sub |= df_raw["name_clean"].map(tr_lower).str.contains(tr_lower(a), regex=False)
        m &= sub
    return df_raw[m]


def test_a_arac_yalniz_izinli_islem():
    assert tuple(ALLOWED_OPS) == ("filtrele", "say", "topla")
    assert set(TOOL_FUNCS.keys()) == set(ALLOWED_OPS), "Araç tablosu yalnız izinli işlemleri içermeli."
    df = load_kb_df()
    before = _sha(KB_PATH)

    # İzinli üç işlem çalışır
    for op in ALLOWED_OPS:
        r = run_tool({"islem": op, "ilce": "Bakırköy", "yil": 2022, "ad_icerir": None}, df)
        assert r["islem"] == op and r["satir_sayisi"] == 2

    # İzinsiz işlemler / anahtarlar / türler reddedilir
    kotu_planlar = [
        {"islem": "sil", "ilce": "Fatih"},
        {"islem": "exec", "ilce": "Fatih"},
        {"islem": "eval", "yil": 2022},
        {"islem": "__import__('os').system('dir')"},
        {"islem": "ortalama", "yil": 2022},
        {"islem": "TOPLA", "yil": 2022},
        {"islem": "topla", "kod": "print(1)"},
        {"islem": "topla", "ilce": "Fatih", "siralama": "guc"},
        {"islem": "topla", "yil": "abc"},
        {"islem": "topla", "yil": 1850},
        {"islem": "topla", "ilce": {"a": 1}},
        {"islem": "topla", "ilce": [1, 2]},
        {"islem": "topla", "ilce": ["a", "b", "c", "d", "e", "f"]},
        {},
        [],
        "topla",
        None,
    ]
    for p in kotu_planlar:
        try:
            run_tool(p, df)
        except PlanError:
            continue
        raise AssertionError(f"İzinsiz plan KABUL EDİLDİ: {p!r}")

    # LLM metni içindeki kod parçası ÇALIŞTIRILMAZ, yalnızca düz alt dize olarak aranır
    marker = os.path.join(PROJECT_ROOT, "zz_rag_marker_dir")
    assert not os.path.exists(marker)
    kod = "__import__('os').mkdir('zz_rag_marker_dir')"
    assert len(kod) < rag_tools.MAX_TEXT_LEN
    try:
        r = run_tool({"islem": "topla", "ilce": None, "yil": None, "ad_icerir": kod}, df)
        assert r["satir_sayisi"] == 0 and not os.path.exists(marker), "LLM metni kod olarak çalıştırıldı!"
    finally:
        if os.path.exists(marker):
            os.rmdir(marker)
    try:
        parse_plan_text("import os; os.system('dir')")
        raise AssertionError("JSON olmayan metin kabul edildi")
    except PlanError:
        pass
    assert validate_plan(parse_plan_text('Plan: {"islem": "say", "yil": "2023"}'))["yil"] == 2023

    # Statik denetim: rag_tools.py içinde kod çalıştırma yok
    with open(os.path.join(PROJECT_ROOT, "src", "rag_tools.py"), "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    yasak_cagri = {"eval", "exec", "compile", "__import__", "open", "system", "popen", "run", "Popen"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            assert name not in yasak_cagri, f"rag_tools.py içinde yasak çağrı: {name}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            for mod in mods:
                assert mod.split(".")[0] in {"os", "re", "sys", "json", "pandas", "src"}, f"Beklenmeyen içe aktarma: {mod}"

    assert _sha(KB_PATH) == before, "Araç kb_docs.jsonl dosyasını değiştirdi!"
    print("Test (a) (Araç yalnız izinli işlem: filtrele/say/topla; kod çalıştırma yok) BAŞARILI!")


class _FakeChat:
    """LLM yerine kullanılır; gelen mesajları kaydeder."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, json_format=False, timeout=0):
        self.calls.append({"messages": messages, "json": json_format})
        if json_format:
            return {"content": '{"islem": "topla", "ilce": null, "yil": 2022, "ad_icerir": null}',
                    "sure_sn": 0.0, "prompt_tokens": 1, "eval_tokens": 1}
        return {"content": "ÇEKİMSER", "sure_sn": 0.0, "prompt_tokens": 1, "eval_tokens": 1}


def test_b_test_sorulari_secimde_cagrilmadi():
    gold = load_gold()
    dev_ids = {q["id"] for q in gold if q["split"] == "dev"}
    test_ids = {q["id"] for q in gold if q["split"] == "test"}
    test_text = [q["soru"] for q in gold if q["split"] == "test"]
    assert len(dev_ids) == 10 and len(test_ids) == 20

    # (1) Yapısal: dev koşucusu sahte LLM ile tam ızgarayı çalıştırır; hiçbir çağrıda test sorusu yok
    ortam = rag_systems.Ortam()
    fake = _FakeChat()
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "rag_runs.jsonl")
        n = rag_systems.dev_izgarasi(chat_fn=fake, log_path=lp, ortam=ortam, verbose=False)
        assert n == 3 * 3 * 10 * 2, f"Beklenen 180 dev kaydı, {n} yazıldı."
        ent = read_log(lp)
        assert {e["soru_id"] for e in ent} == dev_ids
        assert all(e["faz"] == "dev_secim" and e["split"] == "dev" for e in ent)
        for c in fake.calls:
            blob = " ".join(m["content"] for m in c["messages"])
            for tq in test_text:
                assert tq not in blob, "Dev aşamasında bir LLM çağrısında test sorusu metni var!"
        # (2) Dondurma olmadan test koşusu reddedilir
        try:
            rag_systems.test_kosusu(chat_fn=fake, log_path=lp, ortam=ortam, verbose=False)
            raise AssertionError("Dondurma olmadan test koşusu çalıştı!")
        except RuntimeError:
            pass
        # (3) Hash uyuşmazlığında reddedilir
        md = os.path.join(td, "rag_secim.md")
        with open(md, "w", encoding="utf-8") as f:
            f.write('x\n```json\n{"prompt": "P1", "k": 3}\n```\n')
        rag_systems.append_log({"olay": "secim_donduruldu", "sha256": "0" * 64, "zaman_utc": "x"}, lp)
        try:
            rag_systems.test_kosusu(chat_fn=fake, log_path=lp, md_path=md, ortam=ortam, verbose=False)
            raise AssertionError("Hash uyuşmazlığında test koşusu çalıştı!")
        except RuntimeError:
            pass
        assert not any(e.get("faz") == "test" for e in read_log(lp))

    # (4) Gerçek günlük sırası
    entries = read_log(LOG_PATH)
    if not entries:
        print("Test (b) (yapısal kısım) BAŞARILI; gerçek günlük henüz yok.")
        return
    ev_idx = [i for i, e in enumerate(entries) if e.get("olay") == "secim_donduruldu"]
    test_pos = [i for i, e in enumerate(entries) if e.get("soru_id") in test_ids]
    dev_pos = [i for i, e in enumerate(entries) if e.get("faz") == "dev_secim"]
    assert all(entries[i]["soru_id"] in dev_ids for i in dev_pos), "dev_secim kaydında test sorusu var!"
    if not ev_idx:
        assert not test_pos, "Dondurma olayından önce test sorusu çalıştırılmış!"
        print("Test (b) BAŞARILI (henüz dondurulmadı; günlükte test sorusu yok).")
        return
    assert len(ev_idx) == 1, "Günlükte tam bir dondurma olayı olmalı."
    fi = ev_idx[0]
    assert all(i < fi for i in dev_pos), "Dev kayıtları dondurmadan önce bitmeli."
    assert all(i > fi for i in test_pos), "Test kaydı dondurma olayından ÖNCE yazılmış!"
    assert all(entries[i]["faz"] == "test" for i in test_pos)
    olay = entries[fi]
    assert olay["sha256"] == _sha(SECIM_MD), "rag_secim.md dondurulduktan sonra değişmiş."
    with open(FREEZE_TXT, "r", encoding="utf-8") as f:
        ft = f.read()
    assert olay["sha256"] in ft, "Hash protocol_freeze.txt içinde yok."
    assert f"Dondurma Zamanı (UTC): {olay['zaman_utc']}" in ft, "Dondurma UTC zamanı protocol_freeze.txt ile uyuşmuyor."
    if dev_pos:
        assert entries[dev_pos[-1]]["zaman_utc"] <= olay["zaman_utc"]
    if test_pos:
        assert olay["zaman_utc"] <= entries[test_pos[0]]["zaman_utc"]
        keys = [(entries[i]["soru_id"], entries[i]["sistem"]) for i in test_pos]
        assert len(keys) == len(set(keys)), "Bir (soru, sistem) birden fazla kez çalıştırılmış (TEK KEZ ihlali)."
        assert set(keys) <= {(t, s) for t in test_ids for s in ("R0", "R1", "R2")}
        # Test koşusu dondurulan seçimle yapılmış olmalı
        pid, k = rag_systems.secim_oku()
        for i in test_pos:
            e = entries[i]
            assert e["prompt"] == pid and (e["k"] == (None if e["sistem"] == "R0" else k))
    print("Test (b) (Test soruları seçimde çağrılmadı; günlük sırası, hash ve TEK KEZ kuralı) BAŞARILI!")


def test_c_r2_toplami_pandas_ile_ayni():
    gold = {q["id"]: q for q in load_gold()}
    df_raw = pd.read_json(KB_PATH, lines=True)
    df = load_kb_df()
    toplama = [qid for qid, q in gold.items() if q["tur"] == "toplama"]
    assert sorted(toplama) == sorted(IDEAL_PLANS.keys()) and len(toplama) == 9

    for qid, plan in IDEAL_PLANS.items():
        r = run_tool(plan, df)
        pdf = _pandas_filter(df_raw, plan)
        assert r["satir_sayisi"] == len(pdf), f"{qid}: satır sayısı farklı"
        assert sorted(r["filtrelenen_idler"]) == sorted(pdf["record_id"].tolist()), f"{qid}: filtrelenen kayıtlar farklı"
        assert abs(r["toplam_kwp"] - round(float(pdf["power_kwp"].sum()), 2)) < 1e-9, f"{qid}: toplam farklı"
        assert abs(r["toplam_kwp"] - float(gold[qid]["altin_sayisal"])) < 1e-9, f"{qid}: altın toplamla farklı"
        assert sorted(r["filtrelenen_idler"]) == sorted(gold[qid]["altin_kanit"]), f"{qid}: altın kanıtla farklı"

    # say işlemi satır sayısı verir, topla toplamı verir, filtrele toplam vermez
    plan = IDEAL_PLANS["Q18"]
    assert run_tool({**plan, "islem": "say"}, df)["toplam_kwp"] is None
    assert run_tool({**plan, "islem": "filtrele"}, df)["toplam_kwp"] is None

    # Günlükteki gerçek R2 araç sonuçları (varsa) bağımsız pandas ile aynı olmalı
    n_chk = 0
    for e in read_log(LOG_PATH):
        if e.get("sistem") != "R2" or not e.get("arac_sonucu"):
            continue
        a = e["arac_sonucu"]
        plan = e["sorgu_plani"]["llm_plan_dogrulanmis"]
        pdf = _pandas_filter(df_raw, plan)
        assert a["satir_sayisi"] == len(pdf) == len(a["filtrelenen_idler"])
        assert sorted(a["filtrelenen_idler"]) == sorted(pdf["record_id"].tolist())
        if a["islem"] == "topla":
            assert abs(a["toplam_kwp"] - round(float(pdf["power_kwp"].sum()), 2)) < 1e-9
        n_chk += 1
    print(f"Test (c) (R2 toplamı pandas ile aynı: 9 toplama sorusu + {n_chk} günlük araç sonucu) BAŞARILI!")


def test_d_cekimser_yanit_bicimi():
    from src.rag_eval import is_abstention_exact, abstention_loose, evaluate_answer, kb_by_id
    assert is_abstention_exact("ÇEKİMSER")
    assert is_abstention_exact("  ÇEKİMSER\n")
    assert is_abstention_exact("ÇEKİMSER.")
    for kotu in ["çekimser", "Çekimser", "ÇEKİMSER, kanıt yok", "Cevap: ÇEKİMSER", "ÇEKİMSER ÇEKİMSER", "", "CEKIMSER"]:
        assert not is_abstention_exact(kotu), f"Biçim ihlali kabul edildi: {kotu!r}"
    assert abstention_loose("Cevap: ÇEKİMSER") and not abstention_loose("62.0 kWp")

    # Yanıtlanamaz soruda doğru = tam ÇEKİMSER
    gold = {q["id"]: q for q in load_gold()}
    kb = kb_by_id()
    assert evaluate_answer(gold["Q09"], "ÇEKİMSER", kb)["dogru"] is True
    assert evaluate_answer(gold["Q09"], "ÇEKİMSER, kanıt yok", kb)["dogru"] is False
    assert evaluate_answer(gold["Q09"], "126.0 kWp", kb)["dogru"] is False
    # Yanıtlanabilir soruda çekimser = gereksiz ret
    ev = evaluate_answer(gold["Q04"], "ÇEKİMSER", kb)
    assert ev["dogru"] is False and ev["cekimser_gevsek"] is True

    # Sistem mesajı kuralları ve aynılık: R0/R1/R2 (plan dahil) hepsinde messages[0] aynı
    for parca in ["Yalnızca Türkçe yanıt ver.", "Yalnızca verilen kanıta dayan.", "Dosya yılı ve Excel satırı belirt.",
                  "Çelişki varsa iki değeri de yaz.", "Kanıt yoksa tam olarak ÇEKİMSER yaz."]:
        assert parca in SISTEM_MESAJI
    ortam = rag_systems.Ortam()
    fake = _FakeChat()
    q = {"id": "X", "split": "dev", "soru": "2022 dosyasında Bakırköy ilçesinde kayıtlı GES tesislerinin toplam gücü kaç kWp'dir?"}
    for s in ("R0", "R1", "R2"):
        for p in rag_systems.PROMPT_ADAYLARI:
            e = rag_systems.sistem_calistir(s, q, p, 3, ortam, fake)
            assert e["yanit"] == "ÇEKİMSER"
    assert len(fake.calls) == 3 * 3 + 3  # 9 yanıt çağrısı + R2 için 3 plan çağrısı (önbelleksiz)
    for c in fake.calls:
        assert c["messages"][0] == {"role": "system", "content": SISTEM_MESAJI}
        assert c["messages"][1]["role"] == "user"
    # R0 bağlamında kanıt yok; R1/R2 bağlamında k belge var
    u_r0 = rag_systems.kullanici_mesaji("P1", q["soru"], "(Kanıt verilmedi.)", None, "R0")
    assert "(Kanıt verilmedi.)" in u_r0

    # Gerçek HTTP gövdesi: model, ayarlar, sistem mesajı (sahte requests.post ile, ağ yok)
    yakalanan = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ÇEKİMSER"}, "prompt_eval_count": 1, "eval_count": 1}

    orijinal = rag_systems.requests.post

    def sahte_post(url, json=None, timeout=None):
        yakalanan["url"], yakalanan["body"] = url, json
        return _Resp()

    rag_systems.requests.post = sahte_post
    try:
        rag_systems.chat(rag_systems.mesajlar("merhaba"))
    finally:
        rag_systems.requests.post = orijinal
    assert yakalanan["url"] == "http://localhost:11434/api/chat" == OLLAMA_URL
    b = yakalanan["body"]
    assert b["model"] == "qwen2.5:3b-instruct" == MODEL_ADI and b["stream"] is False
    assert b["options"] == {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 400} == SECENEKLER
    assert b["messages"][0] == {"role": "system", "content": SISTEM_MESAJI}

    # Gerçek günlükte çekimser biçimi: bilgi amaçlı sayım (başarısızlık sebebi değildir, ölçüt olarak raporlanır)
    ihlal = [(e["soru_id"], e["sistem"]) for e in read_log(LOG_PATH)
             if "soru_id" in e and abstention_loose(e["yanit"]) and not is_abstention_exact(e["yanit"])]
    print(f"Test (d) (Çekimser biçimi, sistem mesajı ve sabit ayarlar) BAŞARILI! "
          f"[günlükte tam-biçim dışı çekimser yanıt: {len(ihlal)}]")


def test_e_ayristiricilar():
    from src.rag_eval import parse_kwp_values, parse_citations
    assert parse_kwp_values("Toplam 1.124,2 kWp; 126,0 kWp; 62 kW; 8.5 kWp") == [1124.2, 126.0, 62.0, 8.5]
    assert parse_kwp_values("Excel satırı 12 ve 15") == []
    assert parse_citations("2022 dosyasında Excel satırı 12: 99 kWp (2024 dosyası, Excel satırı 13 ve 15).") == {
        (2022, 12), (2024, 13), (2024, 15)}
    assert parse_citations("Dosya yılı: 2022 | Excel satırı: 16\nDosya yılı: 2022 | Excel satırı: 17") == {(2022, 16), (2022, 17)}
    assert parse_citations("2024 dosyasında 13. satırda 29 kWp, 15. satırda 25 kWp vardır.") == {(2024, 13), (2024, 15)}
    assert parse_citations("REC_2022_R12 kaydı") == {(2022, 12)}
    assert parse_citations("ÇEKİMSER") == set()
    print("Test (e) (kWp ve atıf ayrıştırıcıları) BAŞARILI!")


def run_all():
    print("=" * 80)
    print("BM618: ADIM 9B — RAG TESTLERİ (LLM ÇAĞRILMADAN) BAŞLADI")
    print("=" * 80)
    test_a_arac_yalniz_izinli_islem()
    test_b_test_sorulari_secimde_cagrilmadi()
    test_c_r2_toplami_pandas_ile_ayni()
    test_d_cekimser_yanit_bicimi()
    test_e_ayristiricilar()
    print("=" * 80)
    print("[TÜM RAG TESTLERİ (A, B, C, D, E) EKSİKSİZ BAŞARIYLA GEÇTİ]")
    print("=" * 80)


if __name__ == "__main__":
    run_all()
