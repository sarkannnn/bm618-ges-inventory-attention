# BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM

Bu araştırma projesi, İstanbul Büyükşehir Belediyesi (İBB) açık veri portalında yayımlanan 2022–2025 Güneş Enerjisi Santralleri (GES) resmi envanteri üzerinde az verili rejimde (N=151 ana kayıt, 76 bağımsız tesis grubu) iki temel bilimsel soruyu deneysel ve metodolojik olarak incelemektedir:
1. **Regresyon ve Temsil Kapasitesi**: Sınırlı sayıda kurumsal envanter kaydından santralin kurulu gücünü tahmin ederken doğrusal tabuler model (TF-IDF + Ridge) ile derin öğrenme ve dikkat (attention) mimarileri (Maskeli Ortalama Embedding, Tek Sorgulu Attention Pooling, 2 Bloklu Pre-LN Transformer) arasında genelleme üstünlüğü var mıdır?
2. **Kanıta Dayalı Soru-Cevap (RAG)**: Bələdiyye envanteri üzerinde sıfır bağlamlı parametrik LLM (R0), standart BM25 getirmeli RAG (R1) ve kısıtlı SQL/Pandas araç planlayıcılı RAG (R2) sistemlerinin sayısal toplama, atıf doğruluğu ve çekimserlik güvenilirliği nasıl değişmektedir?

---

## 1. Projenin Amacı ve Girdi-Hedef Yapısı

- **Girdi Değişkenleri (3 Temel Özellik)**:
  - `name_clean`: Tesis adı (metin semantiği ve sözcük eşleştirmesi).
  - `district_clean`: Bulunduğu ilçe (coğrafi kategori).
  - `file_year`: Resmi envanter dosya yılı (2022, 2023, 2024, 2025). *(Not: Dosya yılı santralin devreye alınma/kurulum tarihi değil, kurumsal envanter kesit yılıdır).*
- **Hedef Değişken (Target)**:
  - `power_kwp`: Santralin resmi kurulu gücü (kWp cinsinden sürekli değer; model eğitiminde $z = \log(1 + \text{power\_kwp})$ logaritmik dönüşümü kullanılır).

---

## 2. Dizin Yapısı

```
bm618-ges-inventory-attention/
├── data/                       # Denetlenmiş veriler, bölmeler ve bilgi tabanı
│   ├── audited_inventory.csv   # Denetlenmiş 153 kayıt (151 ana regresyon kaydı)
│   ├── splits.csv              # Resmi dondurulmuş 5 katlı GroupKFold bölmesi
│   ├── inner_splits.csv        # 3 katlı iç hiperparametre arama bölmeleri
│   ├── kb_docs.jsonl           # RAG bilgi tabanı belgeleri (153 belge, 76 tesis grubu)
│   └── gold_questions.jsonl    # Altın soru seti (10 dev, 20 test sorusu)
├── docs/                       # Resmi dondurulmuş protokol ve sözleşmeler
│   ├── protokol.md             # Araştırma protokolü ve hipotez şartnamesi
│   ├── protocol_freeze.txt     # Resmi SHA-256 özetleri ve UTC dondurma kayıtları
│   └── rag_secim.md            # RAG dev seçim manifestosu (prompt ve k seçimi)
├── figures/                    # Görsel grafikler ve dikkat haritaları
├── logs/                       # Yürütme günlükleri
│   ├── data_audit.log          # Veri denetim günlüğü
│   └── rag_runs.jsonl          # Tüm RAG çağrılarının dondurulmuş tam günlüğü
├── raw/                        # İBB portalından indirilen ham Excel dosyaları (2022-2025)
├── results/                    # Resmi sonuç tabloları ve model ağırlıkları
│   ├── comparison_pooled.csv   # B0, B1, D1, D2, D3 havuzlanmış dış test metrikleri
│   ├── comparison_bootstrap.csv# 1000 yinelemeli eşleştirilmiş bootstrap farkları
│   ├── leakage_diagnostic.csv  # Gruplu vs Rastgele satır bölmesi sızıntı tanısı
│   ├── ablation_metrics.csv    # Attention ve Konum/Yıl ablasyon metrikleri
│   ├── ablation_bootstrap.csv  # Ablasyon bootstrap farkları
│   ├── rag_metrics.csv         # R0, R1, R2 dış test başarım metrikleri
│   ├── rag_soru_bazli.csv      # 20 test sorusunun sistemler bazında yanıt ve atıf ayrıntıları
│   ├── rag_error_decomp.csv    # Toplama soruları post-hoc hata ayrıştırması
│   ├── rapor_tablolari.md      # Tüm rapor tablolarının otomatik konsolide çıktısı
│   └── weights/                # 5 kat ve 3 tohum için eğitilmiş PyTorch ağırlıkları (*.pt)
├── src/                        # Tüm yürütme, modelleme ve değerlendirme modülleri
└── tests/                      # Otomatikleştirilmiş doğrulama ve birim testleri
```

---

## 3. Sistem Gereksinimleri ve Kurulum

- **Python Sürümü**: `Python 3.14.2` (64-bit)
- **Kütüphanelerin Kurulumu**:
  ```bash
  pip install -r requirements.txt
  ```
  *(Kurulan temel paketler: `torch==2.14.1`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.4`, `scipy==1.18.1`, `openpyxl==3.1.5`, `requests==2.34.2`)*

- **Yerel LLM (RAG Değerlendirmesi İçin)**:
  - Kod harici bulut API'si (OpenAI, Gemini, Anthropic vb.) kullanmaz.
  - Yerel `Ollama` servisi üzerinde `qwen2.5:3b-instruct` modeli çalıştırılır (`http://localhost:11434/api/chat`).
  - Herhangi bir API anahtarı veya gizli ortam değişkeni (env var) gerekmez.

---

## 4. Deneylerin Yürütülme Sırası (İşlem Hattı)

Tüm sonuçları ham veriden itibaren sıfırdan yeniden üretmek için yürütme sırası:

```bash
# 1. Ham Excel dosyalarının denetimi, temizliği ve kalite bayraklarının atanması
python src/audit.py

# 2. Varlık çözümleme ve tesis gruplaması (Entity Matching)
python src/match_entities.py

# 3. 5 katlı GroupKFold dış ve iç bölmelerin deterministik üretimi
python src/splits.py

# 4. Temel modellerin çalıştırılması (B0 Medyan, B1 TF-IDF + Ridge)
python src/baselines.py

# 5. Derin öğrenme modellerinin iç arama, eğitim ve dış test tahmini (D1, D2, D3)
python src/run_main_experiment.py

# 6. Modeller arası karşılaştırma ve eşleştirilmiş bootstrap analizi
python src/compare.py

# 7. Sızıntı tanısı, hata katkısı analizi ve token silme attention analizi
python src/diagnostics.py

# 8. Ablasyon deneylerinin yürütülmesi (Attention ve Konum/Yıl)
python src/ablation.py

# 9. RAG bilgi tabanı ve altın soru setinin oluşturulması
python src/build_gold_kb.py

# 10. RAG dev seçimi ve dondurma kaydı
python src/rag_systems.py dev
python src/rag_eval.py secim
python src/rag_systems.py freeze

# 11. RAG dış test sorularının tek seferlik çalıştırılması ve değerlendirilmesi
python src/rag_systems.py test
python src/rag_eval.py rapor

# 12. RAG toplama soruları post-hoc hata ayrıştırması
python src/rag_error_decomp.py

# 13. Rapor tablolarının otomatik olarak Markdown formatında derlenmesi
python src/make_tables.py
```

Bütünlük ve birim testleri için:
```bash
python tests/test_models.py
python tests/test_rag.py
python tests/test_gold.py
```

---

## 5. Tohum Protokolü ve Bölme Güvenliği

- **Tohumlar (Seeds)**:
  - İç hiperparametre aramasında sabit tohum: `42`.
  - Derin öğrenme modelleri dış test katlarında 3 bağımsız tohumla eğitilmiştir: **`17`**, **`42`**, **`2026`**.
- **Bölme Yöntemi**:
  - 5 katlı tesis bazlı `GroupKFold`. Aynı tesis grubuna (`entity_group_id`) ait farklı yıllardaki kayıtlar kesinlikle hem eğitim hem test katına bölünemez.
- **Kriptografik Bütünlük**:
  - `data/splits.csv` SHA-256 Özeti: `99bd8090449f359cc75cab57ee9e41dfe115df07e9898277c4d204444fdbecad`

---

## 6. Temel Sonuçlar ve Hipotez Kararları

Resmi sonuçlar [results/rapor_tablolari.md](results/rapor_tablolari.md) dosyasından aynen aktarılmıştır:

### Ana Regresyon Modelleri Performansı (Havuzlanmış 5 Dış Kat)
| Model Kodu ve Açıklaması          | Tesis Dengeli MAE (kWp) | Kayıt Düzeyi MAE (kWp) | RMSE (kWp) | R² | Log MAE | İmzalı Hata (kWp) |
| :-------------------------------- | ----------------------: | ---------------------: | ---------: | --: | ------: | ----------------: |
| B0: Sabit Medyan Temel Model      |                   86.68 |                  89.07 |     141.91 | -0.11 |    1.07 |            -42.70 |
| B1: TF-IDF + Ridge                |               **76.04** |                  80.14 |     136.23 | -0.02 |    0.79 |            -38.49 |
| D1: Maskeli Ortalama Embedding    |            82.35 ± 3.47 |           86.38 ± 2.32 |     150.59 | -0.25 |    0.92 |            -38.00 |
| D2: Tek Sorgulu Attention Pooling |            79.58 ± 8.08 |           81.74 ± 6.62 |     129.66 |  0.07 |    0.93 |            -30.57 |
| D3: 2 Bloklu Pre-LN Transformer   |            79.03 ± 6.83 |           81.88 ± 6.00 |     134.93 | -0.01 |    0.91 |            -37.80 |

### Hipotez Değerlendirmeleri
1. **H1 Hipotezi: REDDEDİLDİ.**
   - D2 ve D3'ün tesis dengeli MAE'si B1'den düşük değildir (D2−B1 = +3,54; D3−B1 = +2,99 kWp) ve bütün farkların %95 aralığı sıfırı içerir (D2 − B1: `[-8.54, +14.88]`, D3 − B1: `[-9.46, +13.58]`).
   - Bu nedenle H1 protokoldaki birinci koşula göre reddedilir; bu veriyle modeller arasında fark ayırt edilemedi. B1'in B0'dan farkı da (`-10,64 kWp`, `[-22.85, +4.64]`) sıfırı içerdiğinden B1 üstünlüğü de gösterilmemiştir.
2. **H2 Hipotezi (Veri Sızıntısı Tanısı): KABUL EDİLDİ.**
   - Tesis gruplaması gözetilmeksizin rastgele satır bölmesi uygulandığında test kümesindeki kayıtların %72.2'si eğitimde görülen tesislerden gelmiş ve tesis dengeli MAE 76.04 kWp'den 29.83 kWp'ye (%60.8 sahte iyileşme) inmiştir. Az verili envanterlerde gruplu bölmenin metodolojik bir zorunluluk olduğu ispatlanmıştır.
3. **H3 Hipotezi (Toplama R2 > R1): REDDEDİLDİ.**
   - Dış testteki 6 toplama sorusunda R1 doğruluğu 3/6 (%50.0), R2 doğruluğu 3/6 (%50.0) elde edilmiş; protokol gereği aranan R2 > R1 üstünlük şartı sağlanamamıştır.

---

## 7. Kısıtlar ve Metodolojik Notlar

1. **Küçük Örneklem Sınırı**: Toplam resmi envanter 153 kayıt, ana regresyon kümesi 151 kayıt ve 76 bağımsız tesis grubundan oluşmaktadır. Bu veri hacmi derin ağları eğitmek için değil; küçük veri rejiminde attention mekanizmalarının davranışını kontrollü biçimde sınamak içindir.
2. **Post-Hoc Analiz Niteliği**: `ablation_*.csv` ve `rag_error_decomp.csv` incelemeleri sonuçlar dondurulduktan sonra yapılan post-hoc (betimsel) analizlerdir; birincil H1 ve H3 hipotez kararlarını değiştirmez.
3. **Q18 LLM Sapması (Halüsinasyon)**: Q18 toplama sorusunda kısıtlı araç doğru 62.0 kWp toplamını üretmiş, LLM kanıt maddelerinde 29.0 ve 33.0 kWp satırlarını doğru listelemiş; ancak sonuç cümlesinde araca uymayarak 52.0 kWp iddia etmiştir (saf LLM sapması).
4. **R2 Atıf Metriklerinin Düşüklüğü**: R2 sisteminde atıf doğruluğu R1'e kıyasla düşüktür (0.378 vs 0.679; yalnızca doğru yanıtlarda da 0.308 vs 0.708). R2 aracı genel toplamı özetlerken Excel satır numaralarını yanıtta sıklıkla ihmal etmiştir.
5. **Çekimserlik Biçimleri**: Protokole göre yanıtlanamaz sorularda modelden tam olarak `"ÇEKİMSER"` ifadesi beklenir. Günlükte tam biçim dışında içinde "çekimser" sözcüğü geçen 16 yanıt hem katı (tam biçim) hem gevşek kuralla raporlanmıştır.
6. **Sızıntı Tanısı Skoru Geçersizdir**: Rastgele satır bölmesinden elde edilen 29.83 kWp MAE değeri geçerli bir model başarımı değil; veri sızıntısının yol açtığı yanılsamanın tanı kanıtıdır.

---

## 8. Dondurma Kanıtı (Chain of Custody)

Araştırmanın güvenilirliği Git geçmişine değil, [docs/protocol_freeze.txt](docs/protocol_freeze.txt) belgesindeki kriptografik SHA-256 özetlerine ve kesin UTC zaman damgalarına dayanır:
- **Protokol v1.0 Dondurma**: `2026-10-23T07:18:54Z`
- **splits.csv v1.1 Dondurma**: `2026-10-23T07:30:37Z` | SHA-256: `99bd8090449f359cc75cab57ee9e41dfe115df07e9898277c4d204444fdbecad`
- **Derin Öğrenme Şartnamesi v1.2**: `2026-10-23T08:05:25Z`
- **Altın Test Soruları v1.3**: `2026-10-23T10:35:00Z` | SHA-256: `1ae296e07bf38cdbc024063f56cf81afcdf0c7517914ca2a9206345bc85083fd`
- **RAG Dev Seçimi v1.4**: `2026-10-23T11:54:20Z` | SHA-256: `5bfd98e78f6276527b49b9cbcd3fcf086ed6e1537d6b11bd5bc640e159166972`
