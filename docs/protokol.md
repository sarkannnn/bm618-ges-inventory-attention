# BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
## Dondurulmuş Araştırma Protokolü ve Deney Manifestosu (Frozen Protocol)

**Belge Sürümü:** 1.1 (Dondurulmuş / Frozen)  
**Tarih:** 2026-10-23  
**Ders:** BM618 — İleri Makine Öğrenmesi / Doğal Dil İşleme  
**Sürüm Değişiklik Özeti (v1.1):** 
1. Bölme manifestosu `data/splits.csv` (kayıt başına tek satır, 153 satır) ve `data/inner_splits.csv` (iç katlar, 612 satır) olarak iki yapısal dosyaya ayrıştırıldı; kat atamaları v1.0 ile %100 özdeş korundu.
2. H1, H3 ve H4 hipotezlerinin önceden tanımlanmış yanlışlama koşulları katılaştırıldı ve filtrelenmiş metriklerle güncellendi.
3. Kat dağılım tablosu güncellendi.

**Araştırma Problemi:** “Az sayıda ve yıllar arasında tekrar eden tesis kaydında, attention tabanlı bir model hangi koşullarda basit yöntemlerden daha yararlıdır; aynı veri üzerinde bir dil modeli hangi sorulara güvenilir biçimde yanıt verebilir?”  
**Bağlayıcı İlke:** Amaç yapay olarak en yüksek skoru elde etmek değil; matematiksel türetme, doğru veri temsili, sızıntısız değerlendirme, kontrollü deney ve kanıta dayalı bilimsel yorum üretmektir. Transformer’ın basit modelleri geçemediğini sağlam bir deneyle göstermek tam puanlık başarılı bir sonuçtur.

---

### 1. Veri Sözlüğü (Data Dictionary) ve Model Girdi Sözleşmesi

Tüm modeller ve analizler aşağıdaki kanonik veri şeması üzerinde çalışır. Model girdi sözleşmesi katıdır ve ihlal edilemez.

| Alan Adı | Veri Türü | Kaynak | Açıklama | Model Girdisi Olabilir mi? | Gerekçe / Kısıt |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`record_id`** | String | Türetilmiş | `REC_{file_year}_R{excel_row:02d}` | **HAYIR** | Yapay indeks; tesisin fiziksel gücünü temsil edemez, ezberlemeye yol açar. |
| **`source_file`** | String | Sistem | Orijinal Excel dosya adı | **HAYIR** | Meta veri; model girdisi olamaz. |
| **`sheet`** | String | Sistem | Orijinal Excel çalışma sayfası adı | **HAYIR** | Meta veri; model girdisi olamaz. |
| **`excel_row`** | Integer | Ham Dosya | Excel içindeki satır numarası | **HAYIR** | Sıra numarası doğrudan sızıntı kaynağıdır. |
| **`file_year`** | Integer / Cat | Ham Dosya | Dosya yılı / Kaynak sürümü (2022–2025) | **EVET** | **Zorunlu Girdi (1/3).** "Kurulum yılı" değil, kategorik kaynak sürümüdür. |
| **`district_raw`** | String | Ham Dosya | Excel'deki ham ilçe metni | **HAYIR** | Yazım hatası ve boşluk içerir; yerine temiz ilçe kullanılır. |
| **`district_clean`**| String / Cat | Denetlenmiş | Normalize edilmiş ilçe adı | **EVET** | **Zorunlu Girdi (2/3).** Tesisin coğrafi bağlamı. |
| **`name_raw`** | String | Ham Dosya | Excel'deki ham tesis adı metni | **HAYIR** | Ham format; yerine temiz isim kullanılır. |
| **`name_clean`** | String / Text | Denetlenmiş | Standartlaştırılmış tesis adı | **EVET** | **Zorunlu Girdi (3/3).** Tokenize edilecek ana metin girdisi. |
| **`power_kwp`** | Float | Ham Dosya | Kurulu Güç (kWp) | **HEDEF (Target)** | **KESİNLİKLE HAYIR.** Tahmin edilecek bağımlı değişkendir. Girdiye alınması doğrudan hedef sızıntısıdır. |
| **`status_raw`** | String | Ham Dosya | Ham durum alanı (Aktif/Pasif vb.) | **HAYIR** | Model girdisi yapılamaz. |
| **`status_clean`** | String / Cat | Denetlenmiş | Standardize durum (Aktif, Pasif vb.) | **HAYIR** | Yönerge kuralı: Durum alanı ana model regresyonuna alınmayacaktır. |
| **`entity_group_id`**| String | Denetlenmiş | `ENT_001` – `ENT_076` varlık kimliği | **HAYIR** | Yalnızca GroupKFold bölmesinde kullanılır; modele verilmesi sızıntıdır. |
| **`quality_flag`** | String | Denetlenmiş | Kalite ve denetim bayrakları | **HAYIR** | Denetim alanı; modele verilemez. |
| **`in_main_regression`**| Boolean | Denetlenmiş | Ana regresyona dahil edilme durumu | **HAYIR** | Filtre bayrağıdır; modele verilemez. |
| **`match_confidence`**| String | Denetlenmiş | Eşleşme güven düzeyi (yüksek/orta/düşük)| **HAYIR** | Değerlendirme meta verisidir. |

> [!IMPORTANT]
> **KATI GİRDİ SÖZLEŞMESİ:**
> Regresyon modellerine (B0, B1, D1, D2, D3) verilebilecek alanlar **YALNIZCA VE KESİNLİKLE** şunlardır:
> 1. Tesis Adı (`name_clean`)
> 2. İlçe (`district_clean`)
> 3. Dosya Yılı (`file_year`)  
> Güç, Durum, `record_id`, satır numarası, `entity_group_id` ve güçten türetilen hiçbir ifade model girdisi **OLAMAZ**.

---

### 2. Ham Veri Denetimi ve Tesis Eşleştirme Protokolü

1. **Ham Veri Bütünlüğü:**
   - Ham `.xlsx` dosyaları `raw/` altında salt okunur olarak saklanır, asla değiştirilmez.
   - 2022: 20 geçerli kayıt, toplam 3.834,8 kWp (1 başlık, 1 boş satır dışlandı).
   - 2023: 11 geçerli kayıt, toplam 984,7 kWp (üst başlık, tablo başlığı, alt birim ve boş satırlar dışlandı).
   - 2024: 46 geçerli kayıt, toplam 3.887,0 kWp (1 başlık, 1 boş satır dışlandı).
   - 2025: 76 geçerli kayıt, toplam 8.418,0 kWp (1 başlık, 1 birim satırı dışlandı).
   - Toplam: **153 geçerli envanter kaydı**; tüm dışlamalar Excel satır numaralarıyla `logs/data_audit.log` içindedir.

2. **Eşleştirme İlkeleri (Entity Deduplication):**
   - Eşleştirme kararlarında güç (`power_kwp`) ve durum (`status_clean`) değerleri **kesinlikle kullanılmamıştır**.
   - Anahtar: `İlçe + Normalize Edilmiş Ad + Etap / Alt Birim No`.
   - Alt birimler korunmuştur: Maltepe Orhangazi Şehir Parkı `Su Deposu Üzeri (1)`, `(2)`, `(3)`, `Atletizm Pisti`, `Şeflik` ve `Tenis Alanı` gerekçesiz birleştirilmemiş, 6 ayrı bağımsız varlık grubu olarak kodlanmıştır.
   - Sızıntıyı önleme amaçlı düşük güvenli birleştirmeler:
     - `Bakırköy İPA ve Florya Atatürk Kent Ormanı` (2022 bileşik kaydı $\rightarrow$ 2024-2025 ayrık kayıtlar): `ENT_001`
     - `Bakırköy Ataköy Yerleşkesi` (Kadın Aile $\rightarrow$ Koordinasyon Etap 2 $\rightarrow$ Sosyal Hizmetler): `ENT_002`
     - `Başakşehir Engelliler Merkezi` (1. Etap, 2. Etap, Etap 2, Genel Tesis): `ENT_011`
     - `Kumburgaz İtfaiye` (2023 Silivri $\rightarrow$ 2024-2025 Büyükçekmece idari sınır kayması): `ENT_024`

3. **Çelişki Yönetimi, Karantina Protokolü ve Soru-Cevap Çelişki Tanımı:**
   - **Çelişki Tanımı:** Bilgi tabanında ve soru-cevap setinde çelişki (conflict), şu üç kesin ampirik durumdan biri olarak tanımlanır:
     1. *Aynı tesis ve aynı dosya yılı içinde birden fazla farklı kayıt bulunması (Aynı tesis-yıl çifti):* (ör. 2024 Çekmeköy Ömerli İtfaiye satır 13'te 29.0 kWp, satır 15'te 25.0 kWp).
     2. *Aynı tesisin farklı dosya yıllarında idari sınır / ilçe veya resmi kurum adı değişimi:* (ör. Kumburgaz İtfaiye'nin 2023 dosyasında Silivri ilçesi, 2024 dosyasında Büyükçekmece ilçesi olarak kaydedilmesi; Ataköy yerleşkesinin 2022'de 'Kadın ve Aile Hizmetleri' iken 2025'te 'Sosyal Hizmetler Daire Başkanlığı' olarak adlandırılması).
     3. *Ham dosyalarda aynı tesis için çelişkili / azalan güç değeri kaydedilmesi:* (ör. Kartal Yakacık İtfaiye tesisinin 2024 dosyasında 33.0 kWp iken 2025 dosyasında 29.0 kWp olarak daha düşük listelenmesi; Florya Atatürk Kent Ormanı'nın 2022'de 20.0 kWp iken 2024'te 7.0 kWp olarak kaydedilmesi).
   - **Metodolojik Dondurma İlkesi:** Çelişki tanımı soru yazımından sonra netleştirildi, LLM çalıştırılmadan önce donduruldu.
   - **Saf Çelişki Kısıtı ve Test Dağılımı (7/6/3/4):** Başakşehir Engelliler Merkezi 1. ve 2. Etap kayıtları çelişki değil ayrı alt birimlerdir. Veride yeterli sayıda yapay olmayan saf çelişki bulunmadığından Q24 sorusu 'tekil' türe dönüştürülmüş; test seti dağılımı **7 tekil, 6 toplama, 3 çelişki, 4 yanıtlanamaz (7/6/3/4)** olarak dondurulmuştur. Ataköy yerleşkesinde (Q07) yalnızca kurum adı değişimi çelişki olarak değerlendirilmiş, kapasite artışları çelişki sayılmamıştır.
   - 2024 Çekmeköy Ömerli İtfaiye kayıtları (Excel 13. satırda 29 kWp, 15. satırda 25 kWp) aynı yıl içinde çözülemeyen güç çelişkisidir.
   - Bu iki kayıt otomatik silinmemiş ve ortalaması alınmamıştır. Ana regresyonda karantinaya alınmış (`in_main_regression = False`), ham bilgi tabanında çelişki etiketiyle korunmuştur.
   - Değerlendirme bütünlüğü gereği bu iki kayıt, grubun 2025 kaydıyla (`REC_2025_R26`, 28 kWp) birlikte aynı değerlendirme grubunda (`ENT_036`) ve aynı dış katta (Kat 2) tutulmuştur.

---

### 3. Dondurulmuş Bölme Manifestosu (Frozen Splits Manifest — v1.1)

- **Dosya Yapısı:**
  - `data/splits.csv` (153 satır — kayıt başına tek satır): `record_id, entity_group_id, outer_test_fold, in_main_regression`
  - `data/inner_splits.csv` (612 satır — her dış katın eğitim kayıtları): `outer_fold, record_id, inner_fold`
- **Gruplama Yöntemi:** `entity_group_id` üzerinden 5 dış katlı `GroupKFold`.
- **İç Doğrulama:** Her dış eğitim kümesinde 3 katlı `GroupKFold`.
- **Atama Kuralı:** Atama güce, duruma veya hedefe bakılmadan, sabit tohum ve deterministik sıralama ile yalnız grup büyüklüğüne (kayıt sayısı) göre yapılmıştır.
- **Dış Test Prensibi:** Hiçbir tesis grubunun herhangi bir sürümü dış eğitim ile dış test arasında paylaşılamaz. Her grup tam olarak bir dış test katında yer alır.

#### Dış Kat ve İç Kat Dağılım Tablosu

| Dış Kat (`outer_test_fold`) | Test Grup Sayısı | Test Kayıt Sayısı | Ana Regresyon (`in_main_regression=True`) | Karantina Kayıtları (`in_main_regression=False`) | Dış Eğitim Grup Sayısı | Dış Eğitim Kayıt Sayısı (`inner_splits.csv`) | İç Doğrulama Kat Dağılımı (`inner_fold`: 0, 1, 2) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 15 | 31 | 31 | 0 | 61 | 122 | 3-katlı GroupKFold (20, 20, 21 grup) |
| **Fold 1** | 15 | 31 | 31 | 0 | 61 | 122 | 3-katlı GroupKFold (20, 20, 21 grup) |
| **Fold 2** | 16 | 31 | 29 | 2 (`ENT_036` Ömerli) | 60 | 122 | 3-katlı GroupKFold (20, 20, 20 grup) |
| **Fold 3** | 15 | 30 | 30 | 0 | 61 | 123 | 3-katlı GroupKFold (20, 20, 21 grup) |
| **Fold 4** | 15 | 30 | 30 | 0 | 61 | 123 | 3-katlı GroupKFold (20, 20, 21 grup) |
| **Toplam** | **76 grup** | **153 kayıt** | **151 kayıt** | **2 kayıt** | — | **612 satır** | — |

---

### 4. Araştırma Hipotezleri ve Önceden Tanımlanmış Yanlışlama Ölçütleri

Deneyler başlamadan önce dondurulan kesin hipotezler ve onları yanlışlayacak ampirik gözlemler:

| Hipotez No | Hipotez İfadesi | Beklenen Yön / Mekanizma | Önceden Belirlenmiş Kesin Yanlışlama Ölçütü |
| :--- | :--- | :--- | :--- |
| **H1** | Attention tabanlı modeller (D3: Pre-LN Transformer; ayrıca D2: Attention Pooling), basit Ridge (B1) ve küçük MLP (D1) modellerine kıyasla görülmemiş tesislerde birincil metrik olan tesis dengeli MAE'yi anlamlı biçimde düşürür. | Tokenlar arasındaki veya havuzlamadaki dikkat ağırlıkları tesis adı içindeki anahtar sözcüklere odaklanarak kapasite kestirimini iyileştirir. | **D3'ün tesis dengeli MAE'si hem B1'den hem D1'den düşük değilse, ya da D3−B1 ve D3−D1 farklarının eşleştirilmiş bootstrap %95 aralığı sıfırı içeriyorsa H1 reddedilir.** *(D2 için aynı koşul ayrıca raporlanır.)* |
| **H2** | Rastgele satır bölmesi (Row-level Split), gruplu bölmeye (GroupKFold) göre yapay olarak daha düşük test hatası verir; bu fark veri sızıntısının (data leakage) doğrudan kanıtıdır. | Aynı tesisin farklı yıllardaki sürümleri hem eğitim hem testte yer aldığında model tesisi ezberler (hedef sızıntısı gerçekleşir). | **Rastgele satır bölmesiyle eğitilen B1'in MAE değeri, gruplu B1'in MAE değerinden istatistiksel olarak anlamlı biçimde düşük çıkmazsa (veya eşit/yüksek çıkarsa) H2 reddedilir.** |
| **H3** | RAG sistemine deterministik hesap aracı eklenmesi (R2), standart metin getirmeli RAG'e (R1) kıyasla toplama, sayım ve karşılaştırma sorularında doğru yanıt oranını artırır. | LLM'ler metin getirmeden güç toplamı alırken ilk k belge kısıtına takılır veya aritmetik hata yapar; hesap aracı tüm filtrelenmiş veriyi kodla hatasız hesaplar. | **R1'den R2'ye geçişin etkisi yalnız toplama/sayım/karşılaştırma sorularındaki doğru yanıt oranıyla sınanır; R2 bu sorularda R1'den yüksek değilse H3 reddedilir.** *(Doğru çekimserlik ve gereksiz ret oranı ayrı ölçütlerdir, H3'ün parçası değildir.)* |
| **H4** *(Opsiyonel)* | Dosya yılı bilgisinin model girdisinden çıkarılması (D3 ablasyonu), D3'ün test performansını anlamlı ölçüde değiştirmez. | Tesisin kurulu kapasitesi ismine ve ilçesine içkindir; dosya yılı kurulum tarihi olmadığından kapasite kestiriminde ayırt edici bilgi taşımaz. | **D3'ten dosya yılı çıkarılınca tesis dengeli MAE farkının eşleştirilmiş bootstrap %95 aralığı sıfırı dışlıyorsa H4 reddedilir. p-değeri kullanılmaz.** |

---

### 5. Değerlendirme Metrikleri ve İstatistiksel Analiz

1. **Birincil Metrik (Primary Metric): Tesis Dengeli MAE ($MAE_{balanced}$)**
   $$MAE_{balanced} = \frac{1}{|\mathcal{G}|} \sum_{g \in \mathcal{G}} \left( \frac{1}{|g|} \sum_{i \in g} |y_i - \hat{y}_i| \right)$$
   - Burada $\mathcal{G}$ bağımsız tesis grupları kümesidir.
   - **Gerekçe:** Çok sayıda yılı olan tesislerin (ör. 4 kayıtlı İPA veya Engelliler Merkezi) metriği domine etmesini engeller; her fiziksel tesise eşit ağırlık verir.

2. **İkincil Metrikler (Secondary Metrics):**
   - Kayıt Düzeyinde MAE (kWp): $MAE = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
   - Kök Ortalama Kare Hata (RMSE, kWp): $RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$
   - Belirleme Katsayısı ($R^2$): $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
   - Log-Ölçek MAE: Hedef $\log(1 + y)$ dönüşümü üzerinden hesaplanan hata.

3. **İstatistiksel Karşılaştırma ve Bootstrap Güven Aralığı:**
   - İki modelin ($M_A$ ve $M_B$) hata farkı ($MAE_A - MAE_B$) için, birleştirilmiş 5 dış kat tahminleri üzerinde **tesis grupları düzeyinde 1.000 tekrarlı eşleştirilmiş bootstrap (paired bootstrap)** uygulanacaktır.
   - %95 güven aralığı ($CI_{95\%}$) raporlanacaktır.
   - **Küçük Örneklem Sınırı:** Bu aralık sabit bölmeler ve veri kümesi koşulundaki belirsizliği yansıtır; 76 bağımsız tesisin oluşturduğu küçük örneklem boyutu nedeniyle sonuçlar aşırı genellenemez.

---

### 6. Hiperparametre Arama Protokolü ve Hesap Bütçesi

- **Kural:** Dış test puanına bakarak ASLA model, epoch veya hiperparametre seçilmeyecektir.
- **Model Seçimi:** Tüm model ve hiperparametre seçimi iç eğitim katlarında (3-katlı iç GroupKFold) yapılacak; en düşük iç doğrulama hatasını veren konfigürasyon seçilip dış eğitim kümesinde yeniden eğitilecektir.

#### Aday Hiperparametre Izgarası

| Model | Aday Hiperparametreler | Aday Sayısı | Sabit Değerler |
| :--- | :--- | :---: | :--- |
| **B0 (Medyan)** | Yok (Eğitim kümesi güç medyanı) | 1 | — |
| **B1 (TF-IDF + Ridge)** | $\alpha \in \{0.1, 1.0, 10.0, 100.0\}$ | En fazla 4 | Sublinear TF, ngram (1,2) |
| **D1 (Küçük MLP)** | Öğrenme oranı $\in \{0.001, 0.0003\} \times$ Dropout $\in \{0.0, 0.2\}$ | 4 | Max 150 epoch, patience 15, weight decay $10^{-4}$, batch 16 |
| **D2 (Attention Pooling)**| Öğrenme oranı $\in \{0.001, 0.0003\} \times$ Dropout $\in \{0.0, 0.2\}$ | 4 | Max 150 epoch, patience 15, weight decay $10^{-4}$, batch 16 |
| **D3 (Pre-LN Transformer)**| Öğrenme oranı $\in \{0.001, 0.0003\} \times$ Dropout $\in \{0.0, 0.2\}$ | 4 | $d=32, h=4, d_{ff}=64$, 2 blok, Pre-LN, weight decay $10^{-4}$, batch 16 |

- **Tohum Protokolü:**
  - İç model araması tek sabit tohumla (Tohum: 42) yürütülür.
  - Seçilen en iyi dış modeller üç sabit tohumla eğitilir: **17, 42, 2026**.
  - Tohumlar bağımsız yeni tesis gibi sayılamaz; tohum ortalaması ve standart sapması ayrı raporlanır.

---

### 7. Bilimsel Sınırlar ve Geçerlilik Kapsamı

1. **Zaman Serisi Değildir:** Dosya yılı (2022–2025) santrallerin işletmeye giriş veya şebeke bağlantı tarihi değildir. Dosyalardaki kayıtlar o yıl derlenen kurumsal envanter kesitidir. Bu nedenle bu çalışma "yıllık kapasite artış tahmini" veya "gelecek yıl enerji üretimi tahmini" olarak sunulamaz.
2. **Kapasite Kestirimi Amacı:** Yapılan çalışma, eldeki eksik veya yeni listelenen bir envanter kaydının mevcut kurulu gücünün (kWp) ad ve ilçe semantiğinden kestirilebilirliğini sınayan bir envanter tamamlama / alan kestirimi görevidir.
3. **Küçük Örneklem Sınırı:** Toplam 153 kayıt ve 76 bağımsız tesis, büyük dil modellerini veya devasa Transformer mimarilerini eğitmek için değil; küçük veri rejiminde attention mekanizmasının davranışını, sızıntı kontrolünü ve kanıta dayalı soru-cevap doğruluğunu ölçmek için kontrollü bir laboratuvar görevi görmektedir.

---

### 8. Dondurulmuş Protokol v1.2 Eki: Derin Öğrenme Modelleri (D1, D2, D3) Eğitim ve Mimari Şartnamesi

Bu ek, Adım 6 kapsamında D1, D2 ve D3 modellerinin mimari parametrelerini, eğitim protokolünü, epoch taşıma kuralını ve optimizasyon şartnamesini herhangi bir model eğitilmeden önce dondurur.

1. **Optimizasyon ve Kayıp Fonksiyonu:**
   - **Optimizer:** `AdamW` (`torch.optim.AdamW`), `weight_decay = 1e-4`.
   - **Batch Size:** 16 (DataLoader `shuffle=True` eğitimde, `shuffle=False` doğrulamada).
   - **Kayıp Fonksiyonu:** `MSELoss` (hedef $z = \log(1 + 	ext{power\_kwp})$ üzerinden).
   - **Epoch Limiti ve Sabır:** En fazla 150 epoch, `patience = 15`.

2. **Erken Durdurma ve Model Seçim Ölçütü:**
   - Erken durdurma ve en iyi yapılandırma seçimi **iç doğrulama Tesis Dengeli MAE** ($	ext{kWp}$ cinsinden, $\hat{y} = \max(0, \exp(\hat{z}) - 1)$ ters dönüşümüyle hesaplanır) metriğine göre yapılır. Bu kural B1 ile tam tutarlıdır.

3. **Epoch Taşıma Kuralı:**
   - Seçilen en iyi $(	ext{lr}, 	ext{dropout})$ yapılandırması için, 3 iç fold doğrulamadaki en iyi epoch sayılarının aritmetik ortalaması alınır ve en yakın tam sayıya yuvarlanır:
     $$\text{epoch}^* = \text{round}\left(\frac{1}{3} \sum_{j=0}^2 \text{best\_epoch}_j\right)$$
   - Dış eğitim kümesinde model bu sabit $\text{epoch}^*$ sayısıyla baştan eğitilir; dış test kümesinde model yalnızca bir kez tahmin üretir. Dış test puanına göre hiçbir karar alınmaz.

4. **Tohum (Seed) Protokolü:**
   - **İç Arama Tohumu:** Sabit 42. (Model ağırlık ilklendirmesi ve DataLoader karıştırması).
   - **Dış Nihai Modeller:** 17, 42 ve 2026 tohumlarıyla ayrı ayrı eğitilir (ortalama ve standart sapma raporlanır).

5. **Mimari ve Konum Embedding'i Kuralları:**
   - **Sözlük:** Her (iç ve dış) eğitim kümesinden bağımsız olarak öğrenilir (veri sızıntısı engellenir).
   - **Ortak Embedding:** `nn.Embedding(vocab_size, 32, padding_idx=PAD_IDX)`.
   - **Konum Embedding'i:** D3'te öğrenilen konumsal embedding (`nn.Embedding(max_len, 32)`) token embedding'ine toplanarak eklenir. D1 ve D2 modellerinde konumsal embedding KESİNLİKLE YOKTUR.
   - **Ortak Regresyon Başlığı:** D1, D2 ve D3 modellerinin tamamında birebir aynı regresyon başlığı kullanılır:
     $$\text{Linear}(32, 32) \longrightarrow \text{ReLU} \longrightarrow \text{Dropout}(p) \longrightarrow \text{Linear}(32, 1)$$

6. **Model Mimarileri:**
   - **D1 (Maskeli Ortalama Havuzlama):** PAD tokenları hariç tutularak geçerli token vektörlerinin ortalaması alınır $\to$ Ortak Başlık.
   - **D2 (Bahdanau Attention Havuzlama):** $e_i = v^T \tanh(W h_i + b)$ ($W \in \mathbb{R}^{32 \times 32}, b \in \mathbb{R}^{32}, v \in \mathbb{R}^{32}$). PAD konumları softmax öncesi maskelenir. $r = \sum \alpha_i h_i \to$ Ortak Başlık.
   - **D3 (Pre-LN Transformer Encoder):** Girişte `[CLS]` tokenı; 2 Pre-LN encoder bloğu ($d=32, h=4, d_{ff}=64$); son `LayerNorm(32)`; `[CLS]` çıktısı $\to$ Ortak Başlık. (PyTorch `nn.MultiheadAttention` kullanılmaz; `src/attention.py` temel tensör modülü kullanılır).

7. **D3 Konum Embedding Tablosu Boyutu ve Etkin Parametre Kararı:**
   - D3 modelinin öğrenilen konum embedding tablosu boyutu tüm katlarda sabit $64 \times 32$ ($2.048$ parametre) olarak dondurulmuştur (`nn.Embedding(64, 32)`).
   - Parametre raporunda tablonun toplam parametre sayısı ($2.048$) ve eğitim/test dizilerinin fiilen ulaştığı azami uzunluğa göre kullanılan konumlara ait 'etkin parametre sayısı' ($L_{\text{used}} \times 32$) ayrı ayrı raporlanır.

### 9. Dondurulmuş Protokol v1.2.2 Sapma ve Düzeltme Kaydı (İlklendirme Tohumu Düzeltmesi)

- **Tarih:** 2026-10-23
- **Tespit Edilen Hata:** Adım 7 ilk uygulamasında (v1), `train_neural_net` ve `train_fixed_epochs` fonksiyonları içinde `torch.manual_seed(seed)` çağrısı yapıldığı için, model nesnesi (`instantiate_model`) fonksiyon çağrısından önce oluşturulduğunda ağırlık ilklendirmesi rastgele kalmış; yalnızca DataLoader karıştırması tohuma bağlanmıştı.
- **Düzeltme:** Protokol v1.2 Madde 4 şartnamesi uyarınca ("Model ağırlık ilklendirmesi ve DataLoader karıştırması"), `src/run_main_experiment.py` ve `src/ablation.py` içinde `torch.manual_seed(seed)` ve `np.random.seed(seed)` çağrıları model nesnesi OLUŞTURULMADAN ÖNCE çağrılmıştır. İç arama tohumu sabit 42'dir; dış modeller 17, 42, 2026 tohumlarıyla eğitilmektedir. Dış eğitilen tüm modellerin ağırlıkları `results/weights/{model}_fold{fold}_seed{seed}.pt` altına kaydedilmiş, attention analizi bu ağırlıklarla yürütülmüştür.
- **Değişmeyen Kurallar:** Hiçbir hiperparametre ızgarası (lr, dropout), model mimarisi, seçim/durdurma ölçütü (iç tesis dengeli MAE), epoch taşıma kuralı veya dış kat bölmesi kesinlikle DEĞİŞMEMİŞTİR.
