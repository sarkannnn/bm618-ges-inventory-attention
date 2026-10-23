"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 9a — Altın Set (Gold QA Set) ve Bilgi Tabanı (Knowledge Base Documents)

Bu modül:
1) data/kb_docs.jsonl: Her kayıt için bir doküman (ham/temiz alanlar, dosya, sayfa, güç, çelişki etiketi).
   Ömerli 2024 çifti dahil ve çelişki etiketli.
2) data/gold_questions.jsonl: 30 soru (10 dev, 20 test).
   Dağılım: 10 tekil (3 dev, 7 test), 9 toplama (3 dev, 6 test),
            5 çelişki (2 dev, 3 test), 6 yanıtlanamaz (2 dev, 4 test).
   Test seti dağılımı: 7 tekil, 6 toplama, 3 çelişki, 4 yanıtlanamaz (7/6/3/4).
   Q07 (dev, Ataköy) yalnızca kurum adı değişimini içerir, güç artışı çelişki olarak sunulmaz.
   Q24 Başakşehir Engelliler 1. Etap ayrı alt birim olduğundan tekil türe dönüştürülmüştür.
   Q06 sorusunda Silivri Kapı Buz Pisti kaydının ham Excel dosyasında ilçe sütununda "Fatih" yazdığı belirtilmiştir.
   Dev setinde Florya ve Ömerli yer almaz.
   Altın yanıtlar pandas ile dinamik hesaplanır; yanıtlanamaz sorular "ÇEKİMSER".
"""

import os
import json
import hashlib
import pandas as pd


def get_project_root() -> str:
    """Return the repository root using relative directory resolution."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_kb_docs(df: pd.DataFrame, output_path: str):
    """
    Build data/kb_docs.jsonl from audited inventory dataframe.
    Every record is serialized into a standalone knowledge document.
    """
    records = []
    for _, row in df.iterrows():
        rec_id = str(row['record_id'])
        is_omerli_conflict = (rec_id in ['REC_2024_R13', 'REC_2024_R15'])
        
        if is_omerli_conflict:
            conflict_label = "CELISKI_VAR: 2024 Çekmeköy Ömerli İtfaiye güç çelişkisi (R13=29.0 kWp vs R15=25.0 kWp)"
        else:
            conflict_label = "CELISKI_YOK"
            
        content_text = (
            f"Kayıt Kodu: {rec_id}\n"
            f"Tesis Adı: {row['name_clean']}\n"
            f"Ham Tesis Adı: {row['name_raw']}\n"
            f"İlçe: {row['district_clean']}\n"
            f"Ham İlçe: {row['district_raw']}\n"
            f"Kurulu Güç: {row['power_kwp']} kWp\n"
            f"Durum: {row['status_clean']}\n"
            f"Ham Durum: {row['status_raw']}\n"
            f"Envanter Dosya Yılı: {int(row['file_year'])}\n"
            f"Kaynak Dosya: {row['source_file']}\n"
            f"Sayfa: {row['sheet']}\n"
            f"Excel Satırı: {int(row['excel_row'])}\n"
            f"Varlık Grubu: {row['entity_group_id']}\n"
            f"Kalite Bayrağı: {row['quality_flag']}\n"
            f"Çelişki Durumu: {conflict_label}"
        )
        
        doc = {
            "doc_id": rec_id,
            "record_id": rec_id,
            "source_file": str(row['source_file']),
            "sheet": str(row['sheet']),
            "excel_row": int(row['excel_row']),
            "file_year": int(row['file_year']),
            "district_raw": str(row['district_raw']) if pd.notna(row['district_raw']) else "",
            "district_clean": str(row['district_clean']),
            "name_raw": str(row['name_raw']) if pd.notna(row['name_raw']) else "",
            "name_clean": str(row['name_clean']),
            "power_kwp": float(row['power_kwp']),
            "status_raw": str(row['status_raw']) if pd.notna(row['status_raw']) else "",
            "status_clean": str(row['status_clean']),
            "entity_group_id": str(row['entity_group_id']),
            "quality_flag": str(row['quality_flag']),
            "has_conflict": bool(is_omerli_conflict),
            "celiski_etiketi": conflict_label,
            "content": content_text
        }
        records.append(doc)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
    print(f"[OK] data/kb_docs.jsonl created: {len(records)} documents.")
    return records


def define_question_specs():
    """
    Returns specifications for 30 benchmark questions.
    10 dev (3 tekil, 3 toplama, 2 çelişki, 2 yanıtlanamaz).
    20 test (7 tekil, 6 toplama, 3 çelişki, 4 yanıtlanamaz).
    """
    specs = [
        # ==========================================
        # GELİŞTİRME SETİ (DEV) - 10 SORU
        # ==========================================
        # --- Tekil (3) ---
        {
            "id": "Q01",
            "split": "dev",
            "tur": "tekil",
            "soru": "2022 dosyasında kayıtlı Çatalca Kaleiçi İtfaiye İstasyonu GES tesisinin kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2022_R07"]
        },
        {
            "id": "Q02",
            "split": "dev",
            "tur": "tekil",
            "soru": "2023 dosyasında Avcılar ilçesinde kayıtlı Avcılar Sahil Tuvaleti Çatısı Güneş Enerji Santrali (GES) gücü kaç kWp olarak bildirilmiştir?",
            "altin_kanit": ["REC_2023_R04"]
        },
        {
            "id": "Q03",
            "split": "dev",
            "tur": "tekil",
            "soru": "2024 dosyasında Bağcılar Azerbaycan Dostluk Parkı Güneş Enerji Santrali tesisinin kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2024_R04"]
        },

        # --- Toplama (3) ---
        {
            "id": "Q04",
            "split": "dev",
            "tur": "toplama",
            "soru": "2022 dosyasında Bakırköy ilçesinde kayıtlı GES tesislerinin toplam kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2022_R02", "REC_2022_R03"]
        },
        {
            "id": "Q05",
            "split": "dev",
            "tur": "toplama",
            "soru": "2023 dosyasında Silivri ilçesinde kayıtlı GES tesislerinin toplam kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2023_R06", "REC_2023_R08"]
        },
        {
            "id": "Q06",
            "split": "dev",
            "tur": "toplama",
            "soru": "2022 dosyasında ilçe sütununda Fatih yazan kayıtların (Fatih Spor Kompleksi ve Silivri Kapı Buz Pisti) toplam kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2022_R16", "REC_2022_R17"]
        },

        # --- Çelişki (2) ---
        {
            "id": "Q07",
            "split": "dev",
            "tur": "çelişki",
            "soru": "Ataköy yerleşkesindeki GES tesisi için 2022 dosyasındaki kurum adı ile 2025 dosyasındaki kurum adı arasında hangi ad değişimi / çelişkisi bulunmaktadır?",
            "altin_kanit": ["REC_2022_R03", "REC_2025_R13"]
        },
        {
            "id": "Q08",
            "split": "dev",
            "tur": "çelişki",
            "soru": "Kumburgaz İtfaiye GES tesisinin 2023 ve 2024 dosyalarındaki kayıtları karşılaştırıldığında ilçe ve kurulu güç açısından hangi çelişki görülmektedir?",
            "altin_kanit": ["REC_2023_R08", "REC_2024_R10"]
        },

        # --- Yanıtlanamaz (2) ---
        {
            "id": "Q09",
            "split": "dev",
            "tur": "yanıtlanamaz",
            "soru": "2021 dosyasında Kadıköy ilçesinde kayıtlı GES tesislerinin toplam gücü kaç kWp'dir?",
            "altin_kanit": []
        },
        {
            "id": "Q10",
            "split": "dev",
            "tur": "yanıtlanamaz",
            "soru": "2023 dosyasında Beşiktaş ilçesinde kayıtlı GES tesisinin adı ve kurulu gücü nedir?",
            "altin_kanit": []
        },

        # ==========================================
        # TEST SETİ (TEST) - 20 SORU (7/6/3/4)
        # ==========================================
        # --- Tekil (7) ---
        {
            "id": "Q11",
            "split": "test",
            "tur": "tekil",
            "soru": "2022 dosyasında kayıtlı Sultangazi 75. Yıl Spor Salonu GES tesisinin kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2022_R09"]
        },
        {
            "id": "Q12",
            "split": "test",
            "tur": "tekil",
            "soru": "2022 dosyasında kayıtlı Şile Engelliler Yaz Kampı GES tesisinin kurulu kapasitesi kaç kWp olarak bildirilmiştir?",
            "altin_kanit": ["REC_2022_R14"]
        },
        {
            "id": "Q13",
            "split": "test",
            "tur": "tekil",
            "soru": "2023 dosyasında Şişli Maçka Deprem Parkı Tuvaleti Çatısı Güneş Enerji Santrali (GES) gücü kaç kWp olarak kayıtlıdır?",
            "altin_kanit": ["REC_2023_R11"]
        },
        {
            "id": "Q14",
            "split": "test",
            "tur": "tekil",
            "soru": "2024 dosyasında Bahçelievler Dr. Enver Ören Kültür Merkezi Çatısı Güneş Enerji Santrali tesisinin kurulu gücü nedir?",
            "altin_kanit": ["REC_2024_R05"]
        },
        {
            "id": "Q15",
            "split": "test",
            "tur": "tekil",
            "soru": "2024 dosyasında Kartal Bülent Ecevit Kültür Merkezi Çatısı Güneş Enerji Santrali kurulu kapasitesi kaç kWp'dir?",
            "altin_kanit": ["REC_2024_R23"]
        },
        {
            "id": "Q16",
            "split": "test",
            "tur": "tekil",
            "soru": "2024 dosyasında Tuzla Bahar Merkezi Çatısı Güneş Enerji Santrali tesisinin kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2024_R46"]
        },
        {
            "id": "Q24",
            "split": "test",
            "tur": "tekil",
            "soru": "2022 dosyasında kayıtlı Başakşehir Engelliler Merkezi 1. Etap GES tesisinin kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2022_R12"]
        },

        # --- Toplama (6) ---
        {
            "id": "Q17",
            "split": "test",
            "tur": "toplama",
            "soru": "2023 dosyasında Şişli ve Zeytinburnu ilçelerindeki deprem parkı tuvalet çatısı GES tesislerinin toplam kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2023_R11", "REC_2023_R12"]
        },
        {
            "id": "Q18",
            "split": "test",
            "tur": "toplama",
            "soru": "2024 dosyasında Beykoz ilçesinde kayıtlı GES tesislerinin toplam kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2024_R08", "REC_2024_R09"]
        },
        {
            "id": "Q19",
            "split": "test",
            "tur": "toplama",
            "soru": "2024 dosyasında Büyükçekmece ilçesinde kayıtlı Mimar Sinan ve Büyükçekmece İtfaiye GES tesislerinin toplam kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2024_R11", "REC_2024_R12"]
        },
        {
            "id": "Q20",
            "split": "test",
            "tur": "toplama",
            "soru": "2024 dosyasında Kartal ilçesinde kayıtlı Yakacık İtfaiye ve Soğanlık Gasilhane bina çatısı GES tesislerinin toplam gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2024_R24", "REC_2024_R25"]
        },
        {
            "id": "Q21",
            "split": "test",
            "tur": "toplama",
            "soru": "2022 dosyasında Başakşehir Engelliler Merkezi 1. Etap ve 2. Etap GES tesislerinin toplam kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2022_R12", "REC_2022_R13"]
        },
        {
            "id": "Q22",
            "split": "test",
            "tur": "toplama",
            "soru": "2024 dosyasında Eyüpsultan ilçesinde kayıtlı Tevfik Aydeniz Spor Kompleksi ve Yeşilpınar Kapalı Yüzme Havuzu GES tesislerinin toplam kurulu gücü kaç kWp'dir?",
            "altin_kanit": ["REC_2024_R17", "REC_2024_R18"]
        },

        # --- Çelişki (3) ---
        {
            "id": "Q23",
            "split": "test",
            "tur": "çelişki",
            "soru": "Florya Atatürk Kent Ormanı yerleşkesi için 2022 ve 2024 dosyalarındaki kayıtlarda ad ve kurulu güç açısından hangi farklılık bulunmaktadır?",
            "altin_kanit": ["REC_2022_R02", "REC_2024_R07"]
        },
        {
            "id": "Q25",
            "split": "test",
            "tur": "çelişki",
            "soru": "Kartal Yakacık İtfaiye Bina Çatısı GES tesisinin 2024 ve 2025 dosyalarındaki resmi kayıtlarında kurulu güç açısından hangi çelişkili değerler görülmektedir?",
            "altin_kanit": ["REC_2024_R24", "REC_2025_R42"]
        },
        {
            "id": "Q26",
            "split": "test",
            "tur": "çelişki",
            "soru": "2024 dosyasında Çekmeköy Ömerli İtfaiye GES tesisinin resmi Excel belgesindeki 13. ve 15. satırlarındaki güç değerleri nelerdir?",
            "altin_kanit": ["REC_2024_R13", "REC_2024_R15"]
        },

        # --- Yanıtlanamaz (4) ---
        {
            "id": "Q27",
            "split": "test",
            "tur": "yanıtlanamaz",
            "soru": "2025 dosyasında kayıtlı rüzgar enerjisi santrallerinin (RES) toplam kurulu gücü kaç MW'dir?",
            "altin_kanit": []
        },
        {
            "id": "Q28",
            "split": "test",
            "tur": "yanıtlanamaz",
            "soru": "Ankara Büyükşehir Belediyesi'ne ait 2024 dosyasında kayıtlı GES tesislerinin toplam kurulu kapasitesi kaç kWp'dir?",
            "altin_kanit": []
        },
        {
            "id": "Q29",
            "split": "test",
            "tur": "yanıtlanamaz",
            "soru": "2024 dosyasında Adalar ilçesinde kayıtlı GES tesislerinin adları ve kurulu güçleri nelerdir?",
            "altin_kanit": []
        },
        {
            "id": "Q30",
            "split": "test",
            "tur": "yanıtlanamaz",
            "soru": "2020 dosyasında İstanbul genelinde kayıtlı ilk güneş enerjisi santralinin gücü kaç kWp'dir?",
            "altin_kanit": []
        }
    ]
    return specs


def compute_gold_answers(specs, df: pd.DataFrame) -> list:
    """
    Computes gold answers dynamically using pandas. Elle yazılmaz.
    """
    df_indexed = df.set_index('record_id')
    questions = []

    for spec in specs:
        q_id = spec["id"]
        split = spec["split"]
        tur = spec["tur"]
        soru = spec["soru"]
        kanit = spec["altin_kanit"]

        if tur == "tekil":
            rec_id = kanit[0]
            val = float(df_indexed.loc[rec_id, 'power_kwp'])
            altin_yanit = f"{val:.1f} kWp"
            altin_sayisal = val

        elif tur == "toplama":
            val = float(df_indexed.loc[kanit, 'power_kwp'].sum())
            val_rounded = round(val, 2)
            altin_yanit = f"{val_rounded:.1f} kWp"
            altin_sayisal = val_rounded

        elif tur == "çelişki":
            if q_id == "Q07":
                altin_yanit = "ÇELİŞKİ: Ataköy yerleşkesindeki tesis 2022 dosyasında 'İBB Ataköy Kadın ve Aile Hizmetleri Müdürlüğü GES' (REC_2022_R03) olarak kayıtlıyken, 2025 dosyasında 'Ataköy Sosyal Hizmetler Daire Başkanlığı Bina Çatısı Güneş Enerji Santrali' (REC_2025_R13) adıyla kaydedilmiştir."
            elif q_id == "Q08":
                p23 = float(df_indexed.loc["REC_2023_R08", "power_kwp"])
                d23 = df_indexed.loc["REC_2023_R08", "district_clean"]
                p24 = float(df_indexed.loc["REC_2024_R10", "power_kwp"])
                d24 = df_indexed.loc["REC_2024_R10", "district_clean"]
                altin_yanit = f"ÇELİŞKİ: 2023 dosyasında {d23} ilçesinde {p23:.1f} kWp (REC_2023_R08) kayıtlıyken, 2024 dosyasında {d24} ilçesinde {p24:.1f} kWp (REC_2024_R10) olarak kaydedilmiştir."
            elif q_id == "Q23":
                p22 = float(df_indexed.loc["REC_2022_R02", "power_kwp"])
                p24 = float(df_indexed.loc["REC_2024_R07", "power_kwp"])
                altin_yanit = f"ÇELİŞKİ: Florya Atatürk Kent Ormanı tesisi 2022 dosyasında 20.0 kWp (REC_2022_R02) kayıtlıyken 2024 dosyasında 7.0 kWp (REC_2024_R07) olarak bildirilmiştir."
            elif q_id == "Q25":
                p24 = float(df_indexed.loc["REC_2024_R24", "power_kwp"])
                p25 = float(df_indexed.loc["REC_2025_R42", "power_kwp"])
                altin_yanit = f"ÇELİŞKİ: Yakacık İtfaiye tesisi 2024 dosyasında {p24:.1f} kWp (REC_2024_R24) kayıtlıyken 2025 dosyasında {p25:.1f} kWp (REC_2025_R42) olarak daha düşük bir güçle kaydedilmiştir."
            elif q_id == "Q26":
                p13 = float(df_indexed.loc["REC_2024_R13", "power_kwp"])
                p15 = float(df_indexed.loc["REC_2024_R15", "power_kwp"])
                altin_yanit = f"ÇELİŞKİ: 2024 yılı Excel belgesinde 13. satırda (REC_2024_R13) {p13:.1f} kWp, 15. satırda (REC_2024_R15) {p15:.1f} kWp olarak kaydedilmiş olup aynı yıl içinde çelişki mevcuttur."
            else:
                altin_yanit = "ÇELİŞKİ"
            altin_sayisal = None

        elif tur == "yanıtlanamaz":
            altin_yanit = "ÇEKİMSER"
            altin_sayisal = None
        else:
            raise ValueError(f"Bilinmeyen soru türü: {tur}")

        q_item = {
            "id": q_id,
            "split": split,
            "tur": tur,
            "soru": soru,
            "altin_yanit": altin_yanit,
            "altin_kanit": kanit,
            "altin_sayisal": altin_sayisal
        }
        questions.append(q_item)

    return questions


def build_gold_questions(df: pd.DataFrame, output_path: str):
    """Generates data/gold_questions.jsonl."""
    specs = define_question_specs()
    questions = compute_gold_answers(specs, df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')

    print(f"[OK] data/gold_questions.jsonl created: {len(questions)} questions.")
    return questions


def compute_file_sha256(path: str) -> str:
    """Computes exact SHA-256 hash of a file."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    root = get_project_root()
    csv_path = os.path.join(root, 'data', 'audited_inventory.csv')
    kb_path = os.path.join(root, 'data', 'kb_docs.jsonl')
    gold_path = os.path.join(root, 'data', 'gold_questions.jsonl')

    df = pd.read_csv(csv_path)
    build_kb_docs(df, kb_path)
    build_gold_questions(df, gold_path)
    h_gold = compute_file_sha256(gold_path)
    print(f"[OK] Full data/gold_questions.jsonl SHA-256: {h_gold}")


if __name__ == '__main__':
    main()
