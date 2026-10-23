"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 9a — Altın Soru Seti ve Bilgi Tabanı Doğrulama Testleri (tests/test_gold.py)

Doğrulama Ölçütleri:
(a) 30 soru, 10 dev + 20 test; alan ve kimlik bütünlüğü; "dosyasında kayıtlı" ifadesi kontrolü
    ("kurulan / devreye alınan / kurulumu tamamlanan" ifadeleri kesinlikle bulunmaz).
(b) Tür dağılımı: 10 tekil (3 dev, 7 test), 9 toplama (3 dev, 6 test),
    5 çelişki (2 dev, 3 test), 6 yanıtlanamaz (2 dev, 4 test). Test seti: 7/6/3/4.
(c) Her altın kanıt record_id kb_docs'ta var; Ömerli 2024 kayıtları çelişki etiketli ve
    hiçbir dev sorusunda yer almaz (yalnızca test Q26'da yer alır);
    Florya dev çelişki sorularında yer almaz (test Q23'te yer alır).
(d) Toplama ve tekil yanıtları pandas ile yeniden hesaplanınca tam olarak aynı.
(e) Yanıtlanamaz Q09, Q10, Q29 için ilgili ilçe-yıl kombinasyonunda kb_docs'ta kayıt olmadığı kodla doğrulanır.
(f) Q06 sorusunda "ilçe sütununda Fatih yazan" ibaresi yer alır ve Silivri Kapı Buz Pisti bu kapsamda doğrulanır.
"""

import os
import sys
import json
import pandas as pd

# UTF-8 terminal çıktısı
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Proje kök dizini mutlak yol stringi olmadan dinamik bulunur
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def load_kb_docs(root: str = PROJECT_ROOT):
    kb_path = os.path.join(root, "data", "kb_docs.jsonl")
    assert os.path.exists(kb_path), f"kb_docs.jsonl bulunamadı: {kb_path}"
    docs = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line.strip()))
    return docs


def load_gold_questions(root: str = PROJECT_ROOT):
    gold_path = os.path.join(root, "data", "gold_questions.jsonl")
    assert os.path.exists(gold_path), f"gold_questions.jsonl bulunamadı: {gold_path}"
    questions = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))
    return questions


def test_a_total_and_split_counts(gold_questions=None):
    """
    (a) 30 soru, 10 dev + 20 test; her soru gerekli alanları eksiksiz içerir.
    Soru metinlerinde "kurulan / kurulumu tamamlanan / devreye alınan" ifadeleri bulunmaz;
    "X dosyasında kayıtlı" ifadesi yer alır.
    """
    if gold_questions is None:
        gold_questions = load_gold_questions()

    assert len(gold_questions) == 30, f"Toplam 30 soru bekleniyordu, {len(gold_questions)} bulundu."

    dev_qs = [q for q in gold_questions if q.get("split") == "dev"]
    test_qs = [q for q in gold_questions if q.get("split") == "test"]

    assert len(dev_qs) == 10, f"Geliştirme setinde 10 soru olmalı, {len(dev_qs)} bulundu."
    assert len(test_qs) == 20, f"Test setinde 20 soru olmalı, {len(test_qs)} bulundu."

    ids = [q["id"] for q in gold_questions]
    assert len(set(ids)) == 30, "Soru ID'leri benzersiz olmalı."

    required_fields = ["id", "split", "tur", "soru", "altin_yanit", "altin_kanit"]
    forbidden_terms = ["kurulan", "kurulumu tamamlanan", "devreye alınan"]

    for q in gold_questions:
        for field in required_fields:
            assert field in q, f"{q.get('id')} sorusunda '{field}' alanı eksik."
            assert q[field] is not None, f"{q.get('id')} sorusunda '{field}' None olamaz."

        s_lower = q["soru"].lower()
        for term in forbidden_terms:
            assert term not in s_lower, (
                f"Soru {q['id']} yasaklı ifade içeriyor: '{term}' (Soru: {q['soru']})"
            )
        assert "dosya" in s_lower, f"Soru {q['id']} 'dosyasında' ifadesini içermelidir (Soru: {q['soru']})"

    print("Test (a) (30 soru, 10 dev + 20 test, Alan Bütünlüğü & 'X dosyasında kayıtlı' Şartı) BAŞARILI!")


def test_b_tur_dagilimi(gold_questions=None):
    """
    (b) Tür dağılımı: 10 tekil (3 dev, 7 test), 9 toplama (3 dev, 6 test),
             5 çelişki (2 dev, 3 test), 6 yanıtlanamaz (2 dev, 4 test).
    Test seti dağılımı: 7/6/3/4 (veride saf çelişki kısıtı nedeniyle protokole gerekçeli yazılmıştır).
    """
    if gold_questions is None:
        gold_questions = load_gold_questions()

    from collections import Counter
    overall_counts = Counter(q["tur"] for q in gold_questions)
    dev_counts = Counter(q["tur"] for q in gold_questions if q["split"] == "dev")
    test_counts = Counter(q["tur"] for q in gold_questions if q["split"] == "test")

    # Toplam dağılım kontrolü
    assert overall_counts["tekil"] == 10, f"Toplam 10 tekil soru olmalı, {overall_counts['tekil']} bulundu."
    assert overall_counts["toplama"] == 9, f"Toplam 9 toplama sorusu olmalı, {overall_counts['toplama']} bulundu."
    assert overall_counts["çelişki"] == 5, f"Toplam 5 çelişki sorusu olmalı, {overall_counts['çelişki']} bulundu."
    assert overall_counts["yanıtlanamaz"] == 6, f"Toplam 6 yanıtlanamaz soru olmalı, {overall_counts['yanıtlanamaz']} bulundu."

    # Dev seti dağılım kontrolü (3 + 3 + 2 + 2 = 10)
    assert dev_counts["tekil"] == 3
    assert dev_counts["toplama"] == 3
    assert dev_counts["çelişki"] == 2
    assert dev_counts["yanıtlanamaz"] == 2

    # Test seti dağılım kontrolü (7 + 6 + 3 + 4 = 20)
    assert test_counts["tekil"] == 7
    assert test_counts["toplama"] == 6
    assert test_counts["çelişki"] == 3
    assert test_counts["yanıtlanamaz"] == 4

    print("Test (b) (Tür Dağılımı: 10 tekil, 9 toplama, 5 çelişki, 6 yanıtlanamaz; Test 7/6/3/4) BAŞARILI!")


def test_c_her_altin_kanit_kb_docsta_var(gold_questions=None, kb_docs=None):
    """
    (c) Her altın kanıt record_id kb_docs'ta var;
    Ömerli 2024 çifti DAHİL ve çelişki etiketli;
    Ömerli 2024 çifti dev setinde HİÇBİR SORUDA YER ALMAZ, yalnızca test setinde (Q26) yer alır;
    Florya dev çelişki setinde YER ALMAZ, test setinde (Q23) yer alır.
    Yanıtlanamaz sorularda altın yanıt "ÇEKİMSER" ve kanıt boş liste.
    """
    if gold_questions is None:
        gold_questions = load_gold_questions()
    if kb_docs is None:
        kb_docs = load_kb_docs()

    assert len(kb_docs) == 153, f"kb_docs.jsonl içinde 153 belge olmalı, {len(kb_docs)} bulundu."
    kb_dict = {doc["record_id"]: doc for doc in kb_docs}

    # Ömerli 2024 çelişki kontrolü
    assert "REC_2024_R13" in kb_dict, "REC_2024_R13 kb_docs'ta bulunamadı."
    assert "REC_2024_R15" in kb_dict, "REC_2024_R15 kb_docs'ta bulunamadı."
    assert kb_dict["REC_2024_R13"]["has_conflict"] is True
    assert kb_dict["REC_2024_R15"]["has_conflict"] is True
    assert "CELISKI_VAR" in kb_dict["REC_2024_R13"]["celiski_etiketi"]
    assert "CELISKI_VAR" in kb_dict["REC_2024_R15"]["celiski_etiketi"]

    # Dev setinde Ömerli kaydının hiçbir soruda olmadığını doğrulama
    dev_questions = [q for q in gold_questions if q["split"] == "dev"]
    dev_all_evidence = [rec for q in dev_questions for rec in q["altin_kanit"]]
    assert "REC_2024_R13" not in dev_all_evidence, "Ömerli REC_2024_R13 dev setinde yer alamaz!"
    assert "REC_2024_R15" not in dev_all_evidence, "Ömerli REC_2024_R15 dev setinde yer alamaz!"

    # Dev çelişki sorularında Florya kaydı olmadığını doğrulama (Q07 Ataköy, Q08 Kumburgaz)
    dev_celiski_questions = [q for q in dev_questions if q["tur"] == "çelişki"]
    dev_celiski_evidence = [rec for q in dev_celiski_questions for rec in q["altin_kanit"]]
    assert "REC_2022_R02" not in dev_celiski_evidence, "Florya REC_2022_R02 dev çelişki setinde yer alamaz!"
    assert "REC_2024_R07" not in dev_celiski_evidence, "Florya REC_2024_R07 dev çelişki setinde yer alamaz!"

    # Test Q26'da Ömerli kaydının olduğunu doğrulama
    q26 = [q for q in gold_questions if q["id"] == "Q26"][0]
    assert "REC_2024_R13" in q26["altin_kanit"]
    assert "REC_2024_R15" in q26["altin_kanit"]

    # Test Q23'te Florya kaydının olduğunu doğrulama
    q23 = [q for q in gold_questions if q["id"] == "Q23"][0]
    assert "REC_2022_R02" in q23["altin_kanit"]
    assert "REC_2024_R07" in q23["altin_kanit"]

    # Her soru için kanıt ve yanıtlanamazlık kontrolü
    for q in gold_questions:
        kanitlar = q["altin_kanit"]
        assert isinstance(kanitlar, list), f"{q['id']} altin_kanit bir liste olmalı."

        if q["tur"] == "yanıtlanamaz":
            assert len(kanitlar) == 0, f"Yanıtlanamaz soru {q['id']} kanıt içermemeli."
            assert q["altin_yanit"] == "ÇEKİMSER", f"Yanıtlanamaz soru {q['id']} altın yanıtı 'ÇEKİMSER' olmalı."
        else:
            assert len(kanitlar) > 0, f"{q['tur']} türündeki {q['id']} sorusunun en az bir kanıtı olmalı."
            for rec_id in kanitlar:
                assert rec_id in kb_dict, f"{q['id']} sorusunun kanıtı {rec_id} kb_docs'ta bulunamadı."

    print("Test (c) (Kanıtlar KB'de Mevcut, Ömerli ve Florya Dev'den İzole Edildi) BAŞARILI!")


def test_d_toplama_yanitlari_pandas_ile_yeniden_hesaplaninca_ayni(gold_questions=None, kb_docs=None):
    """
    (d) Toplama yanıtları pandas ile yeniden hesaplanınca tam olarak aynı.
    Her toplama sorusunun kanıt listesindeki kayıtların gücü pandas ile toplanıp
    altın yanıttaki sayısal değerle karşılaştırılır.
    """
    if gold_questions is None:
        gold_questions = load_gold_questions()
    if kb_docs is None:
        kb_docs = load_kb_docs()

    df_kb = pd.DataFrame(kb_docs).set_index("record_id")

    toplama_questions = [q for q in gold_questions if q["tur"] == "toplama"]
    assert len(toplama_questions) == 9

    for q in toplama_questions:
        kanit_ids = q["altin_kanit"]
        recalculated_sum = round(float(df_kb.loc[kanit_ids, "power_kwp"].sum()), 2)

        yanit_str = q["altin_yanit"].replace(" kWp", "").strip()
        yanit_val = round(float(yanit_str), 2)

        assert abs(recalculated_sum - yanit_val) < 1e-4, (
            f"Soru {q['id']} için yeniden hesaplanan toplam ({recalculated_sum}) "
            f"altın yanıttaki değerle ({yanit_val}) uyuşmuyor!"
        )

        if q.get("altin_sayisal") is not None:
            assert abs(recalculated_sum - float(q["altin_sayisal"])) < 1e-4

    # Tekil soruların da pandas ile tam doğrulaması
    tekil_questions = [q for q in gold_questions if q["tur"] == "tekil"]
    for q in tekil_questions:
        rec_id = q["altin_kanit"][0]
        actual_val = float(df_kb.loc[rec_id, "power_kwp"])
        yanit_val = float(q["altin_yanit"].replace(" kWp", "").strip())
        assert abs(actual_val - yanit_val) < 1e-4

    print("Test (d) (Toplama ve Tekil Yanıtları Pandas ile Birebir Yeniden Doğrulandı) BAŞARILI!")


def test_e_unanswerable_district_year_absence(gold_questions=None, kb_docs=None):
    """
    (e) Yanıtlanamaz Q09, Q10, Q29 için ilgili ilçe-yıl kombinasyonunda
    kb_docs'ta hiçbir kayıt bulunmadığı kodla doğrulanır.
    - Q09: 2021 yılı & Kadıköy -> kb_docs içinde 0 kayıt
    - Q10: 2023 yılı & Beşiktaş -> kb_docs içinde 0 kayıt
    - Q29: 2024 yılı & Adalar -> kb_docs içinde 0 kayıt
    """
    if kb_docs is None:
        kb_docs = load_kb_docs()

    # Q09: 2021 & Kadıköy
    matches_q09 = [d for d in kb_docs if d.get("file_year") == 2021 and d.get("district_clean", "").lower() == "kadıköy"]
    assert len(matches_q09) == 0, f"Q09 için beklenmeyen kayıt bulundu: {matches_q09}"

    # Q10: 2023 & Beşiktaş
    matches_q10 = [d for d in kb_docs if d.get("file_year") == 2023 and d.get("district_clean", "").lower() == "beşiktaş"]
    assert len(matches_q10) == 0, f"Q10 için beklenmeyen kayıt bulundu: {matches_q10}"

    # Q29: 2024 & Adalar
    matches_q29 = [d for d in kb_docs if d.get("file_year") == 2024 and d.get("district_clean", "").lower() == "adalar"]
    assert len(matches_q29) == 0, f"Q29 için beklenmeyen kayıt bulundu: {matches_q29}"

    print("Test (e) (Yanıtlanamaz Q09, Q10, Q29 İlçe-Yıl Yokluğu Kodla Doğrulandı) BAŞARILI!")


def test_f_q06_silivri_kapi_fatih_ilce_sutunu(gold_questions=None, kb_docs=None):
    """
    (f) Q06 sorusunda Silivri Kapı Buz Pisti kaydının ham Excel dosyasında
    ilçe sütununda "Fatih" yazdığı ve bu sebeple Fatih toplamına dahil edildiği doğrulanır.
    """
    if gold_questions is None:
        gold_questions = load_gold_questions()
    if kb_docs is None:
        kb_docs = load_kb_docs()

    q06 = [q for q in gold_questions if q["id"] == "Q06"][0]
    assert "ilçe sütununda fatih yazan" in q06["soru"].lower()
    assert "REC_2022_R16" in q06["altin_kanit"]
    assert "REC_2022_R17" in q06["altin_kanit"]

    kb_dict = {doc["record_id"]: doc for doc in kb_docs}
    r17 = kb_dict["REC_2022_R17"]
    assert r17["district_clean"] == "Fatih"
    assert "Silivri Kapı" in r17["name_clean"]
    assert r17["excel_row"] == 17

    print("Test (f) (Q06 Silivri Kapı Buz Pisti Fatih İlçe Sütunu Şartı Doğrulandı) BAŞARILI!")


def run_all():
    print("=" * 80)
    print("BM618: ADIM 9A — ALTIN SET VE BİLGİ TABANI DOĞRULAMA TESTLERİ BAŞLADI")
    print("=" * 80)
    gqs = load_gold_questions()
    kb = load_kb_docs()
    test_a_total_and_split_counts(gqs)
    test_b_tur_dagilimi(gqs)
    test_c_her_altin_kanit_kb_docsta_var(gqs, kb)
    test_d_toplama_yanitlari_pandas_ile_yeniden_hesaplaninca_ayni(gqs, kb)
    test_e_unanswerable_district_year_absence(gqs, kb)
    test_f_q06_silivri_kapi_fatih_ilce_sutunu(gqs, kb)
    print("=" * 80)
    print("[TÜM TESTLER (A, B, C, D, E, F) EKSİKSİZ BAŞARIYLA GEÇTİ]")
    print("=" * 80)


if __name__ == '__main__':
    run_all()
