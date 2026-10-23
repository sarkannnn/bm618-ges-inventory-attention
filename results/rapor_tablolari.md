# BM618 Araştırma Raporu Tabloları
> **Otomasyon Notu**: Bu doküman `src/make_tables.py` tarafından `results/` ve `data/` altındaki kaynak dosyalardan dinamik olarak derlenmiştir. Elle kopyalama yapılmamıştır.

---

## 1. Veri Kümesi ve Dış Çapraz Doğrulama Dağılımı Özeti

İstanbul Büyükşehir Belediyesi (İBB) açık veri portalında yayımlanan 2022–2025 Güneş Enerjisi Santralleri (GES) resmi envanter kayıtları denetlenmiş ve doğrulanmıştır. Toplam 153 resmi kayıt bulunmakta olup, 2024 yılındaki Ömerli İtfaiye çelişkili çifti (REC_2024_R13 / REC_2024_R15) dışlandığında ana regresyon veri kümesi **151 kayıt** ve **76 bağımsız tesis grubundan** (`ENT_001` - `ENT_076`) oluşmaktadır.

### Tablo 1a: Dış Çapraz Doğrulama (5 Katlı GroupKFold) Kat Dağılımı
| Dış Kat (Fold)     | Tesis Sayısı | Kayıt Sayısı | Toplam Güç (kWp) | Ort. Güç (kWp) | Min Güç (kWp) | Maks Güç (kWp) |
| :----------------: | -----------: | -----------: | ---------------: | -------------: | ------------: | -------------: |
|       Kat 0        |           15 |           31 |           3938.6 |          127.1 |           5.0 |          550.0 |
|       Kat 1        |           15 |           31 |           2819.0 |           90.9 |           7.0 |          428.0 |
|       Kat 2        |           16 |           29 |           3079.1 |          106.2 |           6.6 |          314.0 |
|       Kat 3        |           15 |           30 |           3873.6 |          129.1 |           3.0 |          484.0 |
|       Kat 4        |           15 |           30 |           3360.2 |          112.0 |           5.0 |          574.2 |
| **Toplam / Genel** |       **76** |      **151** |      **17070.5** |      **113.0** |       **3.0** |      **574.2** |

### Tablo 1b: Dosya Yılları ve Regresyon Kapsamı Dağılımı
| Dosya Yılı       | Toplam Kayıt | Ana Regresyon Kaydı | Dışlanan (Çelişki) | Toplam Güç (kWp) | Tesis Grubu Sayısı |
| :--------------: | -----------: | ------------------: | -----------------: | ---------------: | -----------------: |
|       2022       |           20 |                  20 |                  0 |           3834.8 |                 19 |
|       2023       |           11 |                  11 |                  0 |            984.7 |                 11 |
|       2024       |           46 |                  44 |                  2 |           3833.0 |                 44 |
|       2025       |           76 |                  76 |                  0 |           8418.0 |                 75 |
| **Genel Toplam** |      **153** |             **151** |              **2** |      **17070.5** |             **76** |

---

## 2. Ana Regresyon Modelleri Karşılaştırması

Tüm modeller 5 katlı gruplu çapraz doğrulamada (GroupKFold) dış test katları üzerinde değerlendirilmiş ve tahminler havuzlanmıştır (N=151 kayıt, 76 tesis). Derin öğrenme modelleri (D1, D2, D3) için 3 tohum (17, 42, 2026) ortalaması ve standart sapması verilmiştir.

### Tablo 2: Havuzlanmış Dış Test Katları Model Performansı
| Model Kodu ve Açıklaması          | Tesis Dengeli MAE (kWp) | Kayıt Düzeyi MAE (kWp) | RMSE (kWp)     | R²           | Log MAE     | İmzalı Hata (kWp) | Log İmzalı Hata |
| :-------------------------------- | ----------------------: | ---------------------: | -------------: | -----------: | ----------: | ----------------: | --------------: |
| B0: Sabit Medyan Temel Model      |                   86.68 |                  89.07 |         141.91 |        -0.11 |        1.07 |            -42.70 |            0.21 |
| B1: TF-IDF + Ridge                |                   76.04 |                  80.14 |         136.23 |        -0.02 |        0.79 |            -38.49 |           -0.07 |
| D1: Maskeli Ortalama Embedding    |            82.35 ± 3.47 |           86.38 ± 2.32 |  150.59 ± 5.30 | -0.25 ± 0.09 | 0.92 ± 0.03 |    -38.00 ± 24.17 |    -0.13 ± 0.27 |
| D2: Tek Sorgulu Attention Pooling |            79.58 ± 8.08 |           81.74 ± 6.62 |  129.66 ± 8.02 |  0.07 ± 0.11 | 0.93 ± 0.08 |    -30.57 ± 17.27 |     0.10 ± 0.19 |
| D3: 2 Bloklu Pre-LN Transformer   |            79.03 ± 6.83 |           81.88 ± 6.00 | 134.93 ± 10.27 | -0.01 ± 0.15 | 0.91 ± 0.07 |    -37.80 ± 18.38 |     0.03 ± 0.25 |

---

## 3. Eşleştirilmiş Bootstrap Karşılaştırmaları ve H1 Hipotez Kararı

Modeller arasındaki tesis dengeli MAE farkları, 76 tesis kümesi üzerinden 1.000 yinelemeli eşleştirilmiş bootstrap yöntemiyle (küme düzeyinde yeniden örnekleme) sınanmış ve %95 güven aralıkları yüzdelik (percentile) aralığı ile hesaplanmıştır.

### Tablo 3: Modeller Arası Tesis Dengeli MAE Farkları ve %95 Güven Aralıkları (Yüzdelik / Percentile)
| Model Karşılaştırması (A − B) | Tesis Dengeli MAE Farkı (kWp) | %95 Bootstrap Güven Aralığı (Yüzdelik) | Sıfırı İçeriyor mu? | İstatistiksel Karar                        |
| :---------------------------: | ----------------------------: | :------------------------------------: | :-----------------: | :----------------------------------------- |
|            B1 − B0            |                        -10.64 |            [-22.85, +4.64]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |
|            D1 − B1            |                          6.31 |            [-6.77, +18.31]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |
|            D2 − B1            |                          3.54 |            [-8.54, +14.88]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |
|            D3 − B1            |                          2.99 |            [-9.46, +13.58]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |
|            D2 − D1            |                         -2.77 |            [-16.24, +9.86]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |
|            D3 − D1            |                         -3.32 |            [-17.74, +8.09]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |
|            D3 − D2            |                         -0.55 |             [-8.84, +7.70]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |

> **H1 Hipotez Kararı: REDDEDİLDİ.**
> D2 ve D3'ün tesis dengeli MAE'si B1'den düşük değildir (D2−B1 = +3,54; D3−B1 = +2,99 kWp) ve bütün farkların %95 aralığı sıfırı içerir. Bu nedenle H1 protokoldaki birinci koşula göre reddedilir; bu veriyle modeller arasında fark ayırt edilemedi. B1'in B0'dan farkı da (−10,64 kWp) sıfırı içerdiğinden B1 üstünlüğü de gösterilmemiştir.

---

## 4. Sızıntı Tanısı (Leakage Diagnostic) ve H2 Hipotez Kararı

Aynı tesis grubuna ait farklı yıllardaki kayıtların eğitim ve test kümelerine kontrolsüz dağıtılmasının yaratacağı yanıltıcı başarı (veri sızıntısı), gruplu GroupKFold ile rastgele satır bölmesi (random split) karşılaştırılarak test edilmiştir.

### Tablo 4: Gruplu GroupKFold ve Rastgele Satır Bölmesi Karşılaştırması
| Bölme Yöntemi               | Tesis Dengeli MAE (kWp) | Kayıt Düzeyi MAE (kWp) | Ortak Tesis Oranı (Train-Test) | Etiket / Tanı Durumu               |
| :-------------------------- | ----------------------: | ---------------------: | :----------------------------: | :--------------------------------- |
| gruplu GroupKFold (GEÇERLİ) |                   76.04 |                  80.14 |             0.000              | geçerli skor                       |
| rastgele satır bölmesi      |                   29.83 |                  30.99 |             0.722              | SIZINTI TANISI, geçerli skor değil |

> **H2 Hipotez Kararı: KABUL EDİLDİ.**
> Tesis bazlı gruplama yapılmaksızın rastgele satır bölmesi uygulandığında, test kümesindeki kayıtların %72.2'si eğitim kümesinde yer alan tesislerden gelmiş ve tesis dengeli MAE 76.04 kWp'den 29.83 kWp'ye (%60.8 düşüş) yapay olarak inmiştir. Bu durum, az verili GES envanterinde gruplu bölmenin metodolojik zorunluluk olduğunu ispatlamıştır.

---

## 5. Ablasyon Deneyleri (Attention Mekanizması ve Konum/Yıl Girdileri)

D2 mimarisinde dikkat ağırlıklı havuzlama (Tek Sorgulu Attention Pooling) ile eşit ağırlıklı ortalama havuzlama; D3 mimarisinde (2 Bloklu Pre-LN Transformer) ise koordinat (enlem/boylam) ve dosya yılı girdilerinin kaldırılması ablasyon deneyleriyle incelenmiştir.

### Tablo 5a: Ablasyon Modelleri Kat Bazlı ve Havuzlanmış Test MAE Değerleri
| Ablasyon Modeli | Tesis Dengeli MAE (kWp) | Kayıt MAE (kWp) | RMSE (kWp) | R²     | Kat 0  | Kat 1 | Kat 2 | Kat 3  | Kat 4 |
| :-------------- | ----------------------: | --------------: | ---------: | -----: | -----: | ----: | ----: | -----: | ----: |
| D2_ref_adim7    |                   75.59 |           78.52 |     136.89 | -0.033 |  96.37 | 62.23 | 53.78 |  88.69 | 78.33 |
| D2_esit_agirlik |                   84.42 |           87.16 |     138.60 | -0.059 | 107.82 | 61.52 | 58.35 | 101.42 | 94.74 |
| D3_ref_adim7    |                   76.75 |           79.77 |     137.02 | -0.035 |  73.95 | 69.40 | 68.03 |  88.38 | 84.59 |
| D3_konumsuz     |                   72.12 |           75.22 |     128.95 |  0.083 |  69.79 | 69.37 | 64.84 |  87.84 | 69.24 |
| D3_yilsiz       |                   77.41 |           80.35 |     134.43 |  0.004 |  82.26 | 69.19 | 62.66 |  95.35 | 78.58 |

### Tablo 5b: Ablasyon Eşleştirilmiş Bootstrap Farkları (%95 Güven Aralığı, Yüzdelik / Percentile)
| Ablasyon Karşılaştırması (A − B) | Tesis Dengeli MAE Farkı (kWp) | %95 Güven Aralığı [Alt, Üst] (Yüzdelik) | Sıfırı İçeriyor mu? | Sonuç                                      |
| :------------------------------- | ----------------------------: | :-------------------------------------: | :-----------------: | :----------------------------------------- |
| D2_esit_agirlik − D2_ref_adim7   |                         +8.83 |             [-1.64, +20.23]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |
| D3_konumsuz − D3_ref_adim7       |                         -4.63 |             [-10.81, +1.54]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |
| D3_yilsiz − D3_ref_adim7         |                         +0.66 |              [-7.93, +8.81]             |         Evet        | Fark ayırt edilemedi (aralık 0'ı içeriyor) |

---

## 6. RAG Dış Test Metrikleri (R0, R1, R2 Karşılaştırması)

Dış test kümesinde yer alan 20 soru üzerinde üç sistem mimarisi sınanmıştır:
- **R0**: Bağlamsız parametrik LLM (Qwen2.5:3b-instruct, sıfır bağlam)
- **R1**: BM25 Retrieval (k=5) + LLM doğrudan yanıt
- **R2**: LLM Planlayıcı + Kısıtlı Aritmetik/Sorgu Aracı + LLM Yanıtlayıcı

### Tablo 6: Dış Test Kümesi (20 Soru) RAG Başarım Metrikleri
| Değerlendirme Ölçütü            | R0: Parametrik LLM (Sıfır Bağlam) | R1: BM25 Retrieval + LLM | R2: Planlayıcı + Kısıtlı Araç |
| :------------------------------ | :-------------------------------: | :----------------------: | :---------------------------: |
| doğru yanıt                     |            3/20 = 0.150           |      16/20 = 0.800       |         17/20 = 0.850         |
| tekil                           |            0/7 = 0.000            |       7/7 = 1.000        |          7/7 = 1.000          |
| toplama                         |            0/6 = 0.000            |       3/6 = 0.500        |          3/6 = 0.500          |
| çelişki                         |            0/3 = 0.000            |       2/3 = 0.667        |          3/3 = 1.000          |
| yanıtlanamaz                    |            3/4 = 0.750            |       4/4 = 1.000        |          4/4 = 1.000          |
| Recall@k                        |            uygulanamaz            |          1.000           |             1.000             |
| sayısal doğruluk (toplama)      |            0/6 = 0.000            |       3/6 = 0.500        |          3/6 = 0.500          |
| sayısal ort. mutlak sapma (kWp) |            uygulanamaz            |          20.200          |             6.000             |
| atıf doğruluğu                  |            uygulanamaz            |          0.679           |             0.378             |
| kanıt kapsamı                   |               0.000               |          0.594           |             0.344             |
| doğru çekimserlik (tam biçim)   |            3/4 = 0.750            |       4/4 = 1.000        |          4/4 = 1.000          |
| doğru çekimserlik (gevşek)      |            4/4 = 1.000            |       4/4 = 1.000        |          4/4 = 1.000          |
| gereksiz ret                    |           16/16 = 1.000           |       2/16 = 0.125       |          1/16 = 0.062         |
| H3_toplama_R2_gt_R1             |                 -                 |            -             |             hayır             |

> **H3 Hipotez Kararı: REDDEDİLDİ.**
> Toplama sorularında R2 doğru yanıt oranı 3/6 (%50.0) olup, R1 doğru yanıt oranı olan 3/6 (%50.0) değerini aşamamıştır. Protokol uyarınca R2 > R1 koşulu sağlanamadığından H3 reddedilmiştir.

---

## 7. RAG Toplama Soruları Post-Hoc Hata Ayrıştırması

> [!NOTE]
> **POST-HOC, H3 KARARINI DEĞİŞTİRMEZ.**
> Bu analiz, H3 hipotez testi protokol gereği sonuçlandıktan sonra, R2 ve R1 sistemlerinin 6 toplama sorusundaki hata kaynaklarını (planlama, araç, LLM sadakati) aydınlatmak amacıyla mevcut günlükler üzerinden derlenmiş betimsel bir incelemedir.

### Tablo 7a: R2 (Planlayıcı + Araç) Hata Ayrıştırma Tablosu (Test Toplama n=6)
| Soru ID | Altın (kWp) | Plan Kümesi Doğru? | Araç Toplamı (kWp) | Araç Doğru? | LLM Değeri | LLM == Araç? | Doğruluk | Hata Sınıfı                             |
| :-----: | ----------: | :----------------: | -----------------: | :---------: | :--------: | :----------: | :------: | :-------------------------------------- |
|   Q17   |        13.2 |       Hayır        |                0.0 |    Hayır    |  ÇEKİMSER  |    Hayır     |  YANLIŞ  | planlayıcı hatası                       |
|   Q18   |        62.0 |        Evet        |               62.0 |     Evet    |    52.0    |    Hayır     |  YANLIŞ  | LLM sapması                             |
|   Q19   |        43.0 |       Hayır        |              149.0 |    Hayır    |   129.0    |    Hayır     |  YANLIŞ  | planlayıcı hatası, ardından LLM sapması |
|   Q20   |        62.0 |        Evet        |               62.0 |     Evet    |    62.0    |     Evet     |  DOĞRU   | Doğru                                   |
|   Q21   |       179.0 |        Evet        |              179.0 |     Evet    |   179.0    |     Evet     |  DOĞRU   | Doğru                                   |
|   Q22   |       156.0 |        Evet        |              156.0 |     Evet    |   156.0    |     Evet     |  DOĞRU   | Doğru                                   |

### Tablo 7b: R1 (BM25 Retrieval + LLM) Hata Ayrıştırma Tablosu (Test Toplama n=6)
| Soru ID | Altın (kWp) | BM25 Kanıtı Kapsadı mı? | LLM Yazdığı Değer   | Doğruluk | Hata Sınıfı                   |
| :-----: | ----------: | :---------------------: | :-----------------: | :------: | :---------------------------- |
|   Q17   |        13.2 |       Evet (%100)       |         13.2        |  DOĞRU   | Doğru                         |
|   Q18   |        62.0 |       Evet (%100)       |       ÇEKİMSER      |  YANLIŞ  | gereksiz ret                  |
|   Q19   |        43.0 |       Evet (%100)       |        126.0        |  YANLIŞ  | bağlam süzme / toplama hatası |
|   Q20   |        62.0 |       Evet (%100)       |         62.0        |  DOĞRU   | Doğru                         |
|   Q21   |       179.0 |       Evet (%100)       |        179.0        |  DOĞRU   | Doğru                         |
|   Q22   |       156.0 |       Evet (%100)       | 78.0 kWp | 78.0 kWp |  YANLIŞ  | toplama yapmadı / eksik biçim |

### Tablo 7c: R2 ve R1 Hata Kaynakları Sayım Özeti
| Sistem | Ölçülen Adım / Durum        | Doğru / Başarılı Sayısı | Oran (%) | Açıklama / Teşhis                                                                 |
| :----: | :-------------------------- | :---------------------: | :------: | :-------------------------------------------------------------------------------- |
|   R2   | Planlayıcı Doğru Küme Süzme |          4 / 6          |  66.7%   | Q18, Q20, Q21, Q22 tam kanıt kümesini süzdü.                                      |
|   R2   | Araç Doğru Toplam Üretme    |          4 / 6          |  66.7%   | Q18 (62.0), Q20 (62.0), Q21 (179.0), Q22 (156.0) doğru toplam üretti.             |
|   R2   | LLM Araç Çıktısına Sadakat  |          3 / 6          |  50.0%   | Q20, Q21, Q22 araca uydu; Q18 ve Q19'da araç toplamından saptı.                   |
|   R2   | Nihai Sayısal Doğruluk      |          3 / 6          |  50.0%   | Hata: 1 Plan Hatası (Q17), 1 LLM Sapması (Q18), 1 Plan Hatası+LLM Sapması (Q19).  |
|   R1   | BM25 Altın Kanıt Kapsama    |          6 / 6          |  100.0%  | BM25 ilk-5 listesinde tüm altın kanıt kayıtları eksiksiz yer aldı.                |
|   R1   | Nihai Sayısal Doğruluk      |          3 / 6          |  50.0%   | Hata: 1 Gereksiz Ret (Q18), 1 Bağlam Süzme Hatası (Q19), 1 Toplama Yapmama (Q22). |

### Post-Hoc Hata Ayrıştırması Değerlendirmesi:
1. **Araç Katmanının Potansiyeli**: R2 planlayıcısı ve araç katmanı 6 sorunun 4'ünde (%66.7 - Q18, Q20, Q21, Q22) altın kanıtla tam eşleşen kayıtları süzmüş ve altın değere tam eşit doğru toplamı üretmiştir.
2. **LLM Sapması (Halüsinasyon)**: Q18'de araç toplamı 62.0 kWp doğru olarak LLM'e verildiği ve yanıttaki kanıt maddelerinde 29.0 kWp ve 33.0 kWp açıkça yazıldığı halde, LLM araca uymayarak yanıtta 52.0 kWp iddia etmiştir (saf LLM sapması).
3. **Planlayıcı Hataları**: Q17'de planlayıcı aşırı kısıtlayıcı bir ad filtresi (`deprem parkı tuvalet çatisı GES`) ürettiği için araç 0 satır döndürmüş ve sistem çekimser kalmıştır. Q19'da ise planlayıcı OR mantığıyla `'İtfaiye'` kelimesini filtreye ekleyerek aynı ilçedeki alakasız Kumburgaz kaydını da dahil etmiş (araç 149.0 kWp), ardından LLM bu araç toplamına da uymayıp 129.0 kWp yazmıştır.
4. **Sonuç**: Araç katmanı %66.7 doğru toplam üretmesine karşın, LLM'in araca itaatsizliği ve planlayıcının hassas süzme zaafları nedeniyle R2'nin nihai doğruluğu %50.0'de (3/6) kalarak R1 ile eşitlenmiştir.
