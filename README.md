# BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM

Bu tədqiqat layihəsi İstanbul Böyükşəhər Bələdiyyəsinin (İBB) rəsmi açıq məlumat portalında dərc olunan 2022–2025 Güneş Enerjisi Santralleri (GES) envanteri üzərində az məlumatlı rejimdə (N=151 qeyd, 76 müstəqil tesis qrupu) iki əsas elmi sualı araşdırır:
1. **Regresyon və Təmsil**: Məhdud sayda sənəd qeydlərindən santralın kurulu gücünü kestirərkən sadə tabuler xətti model (TF-IDF + Ridge) ilə dərin öyrənmə və diqqət (attention) memarlıqları (Maskeli Ortalama Embedding, Tək Sorğulu Attention Pooling, 2 Bloklu Pre-LN Transformer) arasında üstünlük fərqi varmı?
2. **Kanıta Dayalı Sual-Cavab (RAG)**: Bələdiyyə envanteri üzərində parametrli LLM (R0), standart BM25 axtarışlı RAG (R1) və kısıtlı SQL/Pandas alət planlayıcılı RAG (R2) sistemlərinin sayısal toplama və çəkimserlik dəqiqliyi necə dəyişir?

---

## 1. Layihənin Məqsədi və Girdi-Hədəf Strukturu

- **Giriş Dəyişənləri (3 Əsas Xüsusiyyət)**:
  - `name_clean`: Tesis adı (mətn semantikası və söz eyniləşdirməsi).
  - `district_clean`: Yerləşdiyi ilçə (coğrafi kateqoriya).
  - `file_year`: Rəsmi sənəd ili (2022, 2023, 2024, 2025). *(Qeyd: sənəd ili istismara verilmə tarixi deyil, qeydiyyat ilidir).*
- **Hədəf Dəyişən (Target)**:
  - `power_kwp`: Santralın rəsmi kurulu gücü (kWp cinsindən kəsilməz dəyər; təlimdə $z = \log(1 + \text{power\_kwp})$ hədəfi istifadə olunur).

---

## 2. Qovluq Strukturu

```
bm618-ges-inventory-attention/
├── data/                       # Təmizlənmiş məlumatlar, bölmələr və bilik bazası
│   ├── audited_inventory.csv   # Denetlənmiş 153 qeyd (151 əsas regresyon qeydi)
│   ├── splits.csv              # Rəsmi dondurulmuş 5-qatlı GroupKFold bölməsi
│   ├── inner_splits.csv        # 3-qatlı daxili hiperparametr axtarış bölmələri
│   ├── kb_docs.jsonl           # RAG bilik bazası (153 sənəd, 76 tesis qrupu)
│   └── gold_questions.jsonl    # Qızıl sual dəsti (10 dev, 20 test sualı)
├── docs/                       # Rəsmi dondurulmuş protokol və sənədlər
│   ├── protokol.md             # Tədqiqat protokolu və hipotez şərtnaməsi
│   ├── protocol_freeze.txt     # Rəsmi SHA-256 heşləri və UTC dondurma qeydləri
│   └── rag_secim.md            # RAG dev seçim manifestosu (prompt və k seçimi)
├── figures/                    # Vizual qrafiklər və diqqət xəritələri
├── logs/                       # İcra gündəlikləri
│   ├── data_audit.log          # Məlumat auditi qeydləri
│   └── rag_runs.jsonl          # Bütün RAG çağırışlarının dondurulmuş tam qeydi
├── raw/                        # İBB portalından yüklənən xam Excel faylları (2022-2025)
├── results/                    # Rəsmi nəticə cədvəlləri və model çəkiləri
│   ├── comparison_pooled.csv   # B0, B1, D1, D2, D3 birləşdirilmiş xarici test metrikələri
│   ├── comparison_bootstrap.csv# 1000 təkrarlı eşləşdirilmiş bootstrap fərqləri
│   ├── leakage_diagnostic.csv  # Qruplu vs Təsadüfi satır bölməsi sızıntı diaqnozu
│   ├── ablation_metrics.csv    # Attention və Mövqe/İl ablasyon metrikələri
│   ├── ablation_bootstrap.csv  # Ablasyon bootstrap fərqləri
│   ├── rag_metrics.csv         # R0, R1, R2 xarici test başarım metrikələri
│   ├── rag_soru_bazli.csv      # 20 test sualının sistemlər üzrə cavab və atıf detalları
│   ├── rag_error_decomp.csv    # Toplama sualları post-hoc xəta ayrışdırması
│   ├── rapor_tablolari.md      # Bütün hesabat cədvəllərinin avtomatik çıxarışı
│   └── weights/                # 5 qat və 3 toxum üçün öyrədilmiş PyTorch çəkiləri (*.pt)
├── src/                        # Bütün icra və qiymətləndirmə modulları
└── tests/                      # Avtomatlaşdırılmış bütövlük və vahid testləri
```

---

## 3. Sistem Tələbləri və Quraşdırma

- **Python Versiyası**: `Python 3.14.2` (64-bit)
- **Kitabxanaların Quraşdırılması**:
  ```bash
  pip install -r requirements.txt
  ```
  *(Quraşdırılan əsas paketlər: `torch==2.14.1`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.4`, `scipy==1.18.1`, `openpyxl==3.1.5`, `requests==2.34.2`)*

- **Lokal LLM (RAG üçün)**:
  - Proqram xarici bulud API-si işlətmir.
  - Lokal `Ollama` instansiyasında `qwen2.5:3b-instruct` modeli tələb olunur (`http://localhost:11434/api/chat`).
  - Heç bir API açarı və ya xüsusi ətraf mühit dəyişəni (env var) tələb olunmur.

---

## 4. Təkrar İstehsal Sırası (Execution Pipeline)

Bütün eksperimentləri sıfırdan təkrarlamaq üçün komandalar ardıcıllığı:

```bash
# 1. Xam Excel fayllarının auditi və təmizlənməsi
python src/audit.py

# 2. Varlıq və tesis eyniləşdirməsi (Entity Resolution)
python src/match_entities.py

# 3. 5-qatlı GroupKFold və daxili bölmələrin qurulması
python src/splits.py

# 4. Baza modellərinin işə salınması (B0 Medyan, B1 TF-IDF + Ridge)
python src/baselines.py

# 5. Dərin öyrənmə modellərinin təlimi və xarici test proqnozu (D1, D2, D3)
python src/run_main_experiment.py

# 6. Modellərarası müqayisə və bootstrap fərqlərinin hesablanması
python src/compare.py

# 7. Sızıntı diaqnozu, xəta təhlili və token silmə attention analizi
python src/diagnostics.py

# 8. Ablasyon eksperimentlərinin icrası (Attention və Mövqe/İl)
python src/ablation.py

# 9. RAG bilik bazası və qızıl sual dəstinin formalaşdırılması
python src/build_gold_kb.py

# 10. RAG dev seçimi və dondurulması
python src/rag_systems.py dev
python src/rag_eval.py secim
python src/rag_systems.py freeze

# 11. RAG xarici test suallarının icrası və qiymətləndirilməsi
python src/rag_systems.py test
python src/rag_eval.py rapor

# 12. RAG toplama sualları post-hoc xəta ayrışdırması
python src/rag_error_decomp.py

# 13. Yekun hesabat cədvəllərinin avtomatik toplanması
python src/make_tables.py
```

Bütövlük və test yoxlanışları üçün:
```bash
python tests/test_models.py
python tests/test_rag.py
python tests/test_gold.py
```

---

## 5. Toxum Protokolu və Bölmə Təhlükəsizliyi

- **Toxumlar (Seeds)**:
  - Daxili hiperparametr axtarışında sabit tohum: `42`.
  - Dərin öyrənmə modelləri xarici test katlarında 3 sabit toxumla müstəqil təlim olunmuşdur: **`17`**, **`42`**, **`2026`**.
- **Bölmə Metodu**:
  - 5 qatlı tesis əsaslı `GroupKFold`. Eyni tesis qrupuna (`entity_group_id`) aid müxtəlif illərin qeydləri qətiyyən həm təlim, həm sınaq qatına düşmür.
- **Kriptoqrafik Bütövlük**:
  - `data/splits.csv` SHA-256: `99bd8090449f359cc75cab57ee9e41dfe115df07e9898277c4d204444fdbecad`

---

## 6. Əsas Nəticələr və Hipotez Qərarları

Rəsmi nəticələr [results/rapor_tablolari.md](results/rapor_tablolari.md) sənədindən eynilə götürülmüşdür:

### Ana Regresyon Modelləri Performansı (5 Qat Birləşdirilmiş)
| Model Kodu və Açıklaması          | Tesis Dengeli MAE (kWp) | Kayıt Düzeyi MAE (kWp) | RMSE (kWp) | R² | Log MAE | İmzalı Hata (kWp) |
| :-------------------------------- | ----------------------: | ---------------------: | ---------: | --: | ------: | ----------------: |
| B0: Sabit Medyan Temel Model      |                   86.68 |                  89.07 |     141.91 | -0.11 |    1.07 |            -42.70 |
| B1: TF-IDF + Ridge                |               **76.04** |                  80.14 |     136.23 | -0.02 |    0.79 |            -38.49 |
| D1: Maskeli Ortalama Embedding    |            82.35 ± 3.47 |           86.38 ± 2.32 |     150.59 | -0.25 |    0.92 |            -38.00 |
| D2: Tek Sorgulu Attention Pooling |            79.58 ± 8.08 |           81.74 ± 6.62 |     129.66 |  0.07 |    0.93 |            -30.57 |
| D3: 2 Bloklu Pre-LN Transformer   |            79.03 ± 6.83 |           81.88 ± 6.00 |     134.93 | -0.01 |    0.91 |            -37.80 |

### Hipotez Nəticələri
1. **H1 Hipotezi: REDDEDİLDİ.**
   - D2 və D3-ün tesis dengeli MAE-si B1-dən aşağı deyil (D2 − B1 = +3.54 kWp; D3 − B1 = +2.99 kWp).
   - Modellərarası bütün 1.000 təkrarlı eşləşdirilmiş bootstrap %95 etibarlılıq aralıqları sıfırı (0) ehtiva edir (məsələn, D2 − B1: `[-8.54, +14.88]`, D3 − B1: `[-9.46, +13.58]`).
   - Bu məlumat dəsti ilə modellər arasında fərq ayırd edilə bilmədi. B1-in B0-dan fərqi də (`-10.64 kWp`, `[-22.85, +4.64]`) sıfırı ehtiva etdiyindən B1-in də statistik üstünlüyü sübut olunmamışdır.
2. **H2 Hipotezi (Veri Sızıntısı): KABUL EDİLDİ.**
   - Tesis qruplaşması nəzərə alınmadan təsadüfi satır bölməsi aparıldıqda sınaq dəstindəki qeydlərin %72.2-si təlimdə görünən tesis qruplarından gəlmiş və tesis dengeli MAE 76.04 kWp-dən 29.83 kWp-ə süni şəkildə enmişdir (%60.8 saxta yaxşılaşma).
3. **H3 Hipotezi (Toplama R2 > R1): REDDEDİLDİ.**
   - Xarici testdəki 6 toplama sualında R1 doğruluğu 3/6 (%50.0), R2 doğruluğu 3/6 (%50.0) çıxmış; protokol tələbi olan R2 > R1 şərti ödənməmişdir.

---

## 7. Məhdudiyyətlər və Metodoloji Qeydlər

1. **Kiçik Nümunə Ölçüsü**: Ümumi envanter 153 qeyd, ana regresyon 151 qeyd və 76 müstəqil tesis qrupundan ibarətdir. Bu məlumat həcmi böyük modelləri təlim etmək üçün deyil, az verili rejimdə diqqət memarlıqlarının davranışını laboratoriya şəraitində ölçmək üçündür.
2. **Post-Hoc Xarakter**: `ablation_*.csv` və `rag_error_decomp.csv` araşdırmaları nəticələr dondurulduqdan sonra aparılan post-hoc (təsviri) analizlərdir; ilkin H1 və H3 hipotez qərarlarını dəyişmir.
3. **Q18 LLM Sapması (Halüsinasiya)**: Q18 toplama sualında alət qatı düzgün 62.0 kWp cəmini hesablamış, LLM kanıt olaraq 29.0 və 33.0 kWp maddələrini yazsa da, yekun cümlədə araca tabe olmayaraq 52.0 kWp iddia etmişdir.
4. **R2 Atıf Göstəricilərinin Aşağı Olması**: R2 sistemində atıf dəqiqliyi R1-dən aşağıdır (0.378 vs 0.679; doğru cavablarda 0.308 vs 0.708). Çünki R2 aracı ümumiləşdirilmiş cəm çıxararkən sənəd və sətir nömrələrini çox vaxt yanıta əlavə etməmişdir.
5. **Çəkimser Cavab Biçimləri**: Qızıl standarta görə cavab verilə bilməyən suallarda modeldən tam olaraq `"ÇEKİMSER"` tələb olunur. Günlükdə tam biçim xaricində içində "çekimser" sözü keçən 16 cavab həm tam, həm də boş (gevşək) meyarlarla qiymətləndirilmişdir.
6. **Sızıntı Diaqnozu Skoru Etibarsızdır**: Təsadüfi satır bölməsi ilə əldə edilən 29.83 kWp MAE dəyəri elmi bir nailiyyət deyil; məlumat sızıntısının yaratdığı metodoloji xətanın nümunəsidir.

---

## 8. Dondurulma Sübutu (Chain of Custody)

Tədqiqatın elmi dürüstlüyü Git tarixinə deyil, [docs/protocol_freeze.txt](docs/protocol_freeze.txt) sənədindəki kriptoqrafik SHA-256 heşlərinə və UTC vaxt möhürlərinə əsaslanır:
- **Protokol v1.0 Kilitlənməsi**: `2026-10-23T07:18:54Z`
- **splits.csv v1.1 Kilitlənməsi**: `2026-10-23T07:30:37Z` | SHA-256: `99bd8090449f359cc75cab57ee9e41dfe115df07e9898277c4d204444fdbecad`
- **Dərin Öyrənmə Şərtnaməsi v1.2**: `2026-10-23T08:05:25Z`
- **Altın Test Sualları v1.3**: `2026-10-23T10:35:00Z` | SHA-256: `1ae296e07bf38cdbc024063f56cf81afcdf0c7517914ca2a9206345bc85083fd`
- **RAG Dev Seçimi v1.4**: `2026-10-23T11:54:20Z` | SHA-256: `5bfd98e78f6276527b49b9cbcd3fcf086ed6e1537d6b11bd5bc640e159166972`
