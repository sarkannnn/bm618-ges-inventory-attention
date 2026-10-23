# RAG Prompt ve k Seçimi (Adım 9b) — YALNIZ 10 dev sorusu

Bu belge, 20 test sorusu çalıştırılmadan ÖNCE yazılmış ve SHA-256'sı `docs/protocol_freeze.txt` içine eklenmiştir. Test soruları (Q11–Q30) bu aşamada hiçbir sistemle çalıştırılmamıştır.

## Seçim

**Seçilen prompt: P1, k = 5**

```json
{"prompt": "P1", "k": 5}
```

Seçim ölçütü (dev günlüğü okunmadan ÖNCE sabitlendi): (prompt, k) çiftleri arasından R1 ve R2'nin dev doğru yanıt sayıları toplamı (en çok 20) en yüksek olan seçilir. Eşitlikte sırasıyla: (1) R1+R2 atıf doğruluğu ortalaması yüksek olan, (2) küçük k, (3) küçük prompt numarası. Yalnız 10 dev sorusu kullanılır.

## Dev sonuçları (10 dev soru; R1 ve R2; doğru yanıt sayısı / 10)

| prompt | k | R1 doğru | R2 doğru | R1+R2 | R1 atıf | R2 atıf | R1 gereksiz ret | R2 gereksiz ret | R1 çekimserlik | R2 çekimserlik |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 3 | 8/10 | 8/10 | 16/20 | 0.500 | 0.333 | 1/8 | 2/8 | 2/2 | 2/2 |
| P1 **(seçilen)** | 5 | 7/10 | 9/10 | 16/20 | 0.444 | 0.429 | 2/8 | 1/8 | 2/2 | 2/2 |
| P1 | 10 | 5/10 | 9/10 | 14/20 | 0.429 | 0.286 | 1/8 | 1/8 | 2/2 | 2/2 |
| P2 | 3 | 4/10 | 7/10 | 11/20 | 0.857 | 0.857 | 1/8 | 1/8 | 0/2 | 1/2 |
| P2 | 5 | 2/10 | 8/10 | 10/20 | 0.738 | 0.893 | 1/8 | 1/8 | 0/2 | 1/2 |
| P2 | 10 | 4/10 | 6/10 | 10/20 | 0.746 | 0.952 | 1/8 | 1/8 | 0/2 | 1/2 |
| P3 | 3 | 6/10 | 5/10 | 11/20 | 0.375 | 1.000 | 4/8 | 2/8 | 2/2 | 0/2 |
| P3 | 5 | 7/10 | 7/10 | 14/20 | 0.452 | 0.893 | 1/8 | 1/8 | 1/2 | 0/2 |
| P3 | 10 | 6/10 | 7/10 | 13/20 | 0.442 | 0.905 | 2/8 | 1/8 | 2/2 | 1/2 |

## Dev soru bazlı doğruluk (seçilen yapılandırma)

| soru | tür | R1 | R2 |
|---|---|---|---|
| Q01 | tekil | ✓ | ✓ |
| Q02 | tekil | ✓ | ✓ |
| Q03 | tekil | ✓ | ✓ |
| Q04 | toplama | ✗ | ✓ |
| Q05 | toplama | ✓ | ✓ |
| Q06 | toplama | ✓ | ✓ |
| Q07 | çelişki | ✗ | ✗ |
| Q08 | çelişki | ✗ | ✓ |
| Q09 | yanıtlanamaz | ✓ | ✓ |
| Q10 | yanıtlanamaz | ✓ | ✓ |

## Yalnız getirme (LLM'siz) dev Recall@k — yanıtlanabilir 8 dev soru

| k | Recall@k |
|---|---|
| 3 | 0.938 |
| 5 | 0.938 |
| 10 | 0.938 |

## Sabit bileşenler

- Model: `qwen2.5:3b-instruct` (Q4_K_M), Ollama, CPU. Ayarlar: `{"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 400}`
- Sistem mesajı (tüm sistemlerde aynı, Ollama'nın varsayılan Qwen mesajı değiştirilir):

> Yalnızca Türkçe yanıt ver. Yalnızca verilen kanıta dayan. Dosya yılı ve Excel satırı belirt. Çelişki varsa iki değeri de yaz. Kanıt yoksa tam olarak ÇEKİMSER yaz.

- Prompt varyantları (yalnız kullanıcı mesajındaki talimat farklıdır):

  - **P1** — Yalın: kanıt + soru, ek talimat yok
    - talimat: (yok)
    - R2 ek talimat: (yok)
  - **P2** — Açık yanıt biçimi: eşleşen kayıttan değeri kWp olarak yaz, parantezli atıf (ÇEKİMSER kuralı yalnız sistem mesajında)
    - talimat: Soruyu yukarıdaki kanıta göre yanıtla. Kanıtta sorudaki dosya yılı ve ilçe veya tesis adıyla eşleşen kayıt varsa değeri kWp olarak yaz ve her değer için dosya yılını ve Excel satırını parantez içinde belirt, örneğin (2022 dosyası, Excel satırı 12).
    - R2 ek talimat:  Araç sonucu varsa toplamı ve sayıyı aynen araç sonucundan al, kendin hesaplama; araç sonucundaki filtrelenen satır sayısını yanıtında belirt.
  - **P3** — Adım adım: önce eşleşen satırları seç, sonra topla ve atıf yap (ÇEKİMSER kuralı yalnız sistem mesajında)
    - talimat: Şu adımları izle: (1) Kanıttaki satırlardan dosya yılı ve ilçesi veya tesis adı sorudakiyle uyuşanları seç. (2) Toplam isteniyorsa seçtiğin tüm satırların gücünü topla. (3) Aynı tesis için farklı değerler veren satırlar varsa her iki değeri de yaz. (4) Yanıtta dosya yılını ve Excel satırını belirt.
    - R2 ek talimat:  Araç sonucu verildiyse toplam ve sayım için yalnızca araç sonucunu kullan (kendin toplama yapma) ve filtrelenen satır sayısını yanıtında belirt.

- Aday k değerleri: 3, 5, 10. Getirme: kendi yazdığımız BM25 (k1=1.5, b=0.75), sorgu = soru metni.

