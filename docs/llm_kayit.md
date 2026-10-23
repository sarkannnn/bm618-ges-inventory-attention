# LLM Kaydı (Adım 9b — RAG)

## Model ve çalışma ortamı

| Öğe | Değer |
|---|---|
| Model | `qwen2.5:3b-instruct` (aile: qwen2, 3.1 B parametre) |
| Nicemleme | Q4_K_M (`/api/show` ile doğrulandı) |
| Model kimliği (`ollama list`) | `357c53fb659c`, 1,9 GB |
| Ollama sürümü | 0.40.9 (`/api/version` ile doğrulandı) |
| Uç nokta | `http://localhost:11434/api/chat` (Python `requests`, `stream=false`) |
| Çalışma biçimi | `ollama ps`: **%100 CPU**, bağlam 8192, runner `llamacpp` |
| Donanım | Intel tümleşik GPU (kullanılmadı), 15,7 GB RAM, CPU 3,54 GHz |
| `keep_alive` | 60 dk (yalnız modeli bellekte tutar, çıktıyı etkilemez) |

## Sabit ayarlar (tüm sistemlerde ve tüm çağrılarda aynı)

```json
{"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 400}
```

Ollama'nın varsayılan Qwen sistem mesajı (`You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`)
her çağrıda `messages[0]` (rol `system`) olarak gönderilen aşağıdaki mesajla **değiştirilir**.
`tests/test_rag.py` test (d) bunu sahte HTTP ile doğrular: model adı, ayarlar ve `messages[0]` birebir aynıdır.

## Sistem mesajı (R0, R1, R2 ve R2 planlayıcısı için aynı)

> Yalnızca Türkçe yanıt ver. Yalnızca verilen kanıta dayan. Dosya yılı ve Excel satırı belirt. Çelişki varsa iki değeri de yaz. Kanıt yoksa tam olarak ÇEKİMSER yaz.

## Kullanıcı mesajı şablonu

```
KANIT:
<R0: "(Kanıt verilmedi.)" | R1/R2: BM25 ilk-k satırı>
<satır biçimi: - Dosya yılı: Y | Excel satırı: N | İlçe: İ | Tesis: T | Güç: G kWp>

[yalnız R2]
ARAÇ SONUCU (kod hesapladı):
<işlem | filtre | filtrelenen satır sayısı | (topla ise) toplam | filtrelenen satırlar>

SORU: <soru>

<seçilen varyantın talimatı (+ R2 ise araç ek talimatı)>
```

Bağlama **çelişki etiketi, kalite bayrağı, varlık grubu (ENT_xxx), kayıt kodu ve durum sütunu konmaz**
(altın etiketin sızmaması için). Yalnız dosya yılı, Excel satırı, ilçe, tesis adı ve güç görünür.

## Prompt varyantları (yalnız talimat farklı)

- **P1 (SEÇİLEN)** — Yalın: ek talimat yok. R2 ek talimatı yok.
- **P2** — Talimat: "Soruyu yukarıdaki kanıta göre yanıtla. Kanıtta sorudaki dosya yılı ve ilçe veya tesis adıyla eşleşen kayıt varsa değeri kWp olarak yaz ve her değer için dosya yılını ve Excel satırını parantez içinde belirt, örneğin (2022 dosyası, Excel satırı 12)." R2 ek: "Araç sonucu varsa toplamı ve sayıyı aynen araç sonucundan al, kendin hesaplama; araç sonucundaki filtrelenen satır sayısını yanıtında belirt."
- **P3** — Talimat: "Şu adımları izle: (1) Kanıttaki satırlardan dosya yılı ve ilçesi veya tesis adı sorudakiyle uyuşanları seç. (2) Toplam isteniyorsa seçtiğin tüm satırların gücünü topla. (3) Aynı tesis için farklı değerler veren satırlar varsa her iki değeri de yaz. (4) Yanıtta dosya yılını ve Excel satırını belirt." R2 ek: "Araç sonucu verildiyse toplam ve sayım için yalnızca araç sonucunu kullan (kendin toplama yapma) ve filtrelenen satır sayısını yanıtında belirt."

Seçim: **P1, k = 5** (`docs/rag_secim.md`, yalnız 10 dev sorusu).

## R2 araç planlayıcısı

Ayrı bir LLM çağrısı; aynı model, aynı ayarlar, **aynı sistem mesajı**; ek olarak Ollama `format: "json"` (yalnız çıktı biçimini kısıtlar,
örnekleme ayarlarını değiştirmez). Planlayıcı promptu `src/rag_systems.py` içindeki `PLAN_PROMPTU` sabitidir
(biçim, kurallar ve altın sette bulunmayan uydurma Pendik / Esenler / Maltepe örnekleri). Plan, `src/rag_tools.py`
tarafından beyaz listeye göre doğrulanır; yalnız `filtrele`, `say`, `topla` çalışır, LLM kod çalıştıramaz.

## Getirme

Kendi yazdığımız Okapi BM25 (k1 = 1,5; b = 0,75), sorgu = soru metni. Türkçe küçük harf: İ→i, I→ı; sözcük kökü: ilk 8 harf.
Dizine yalnız ilçe, tesis adı ve dosya yılı girer. Aday k: {3, 5, 10}.

## Sapmalar ve dürüstlük notları

1. **P2/P3 talimatları dev ızgarasından önce iki kez yeniden yazıldı.** İlk yazımda "ÇEKİMSER" sözcüğünü geçiren talimatlar modeli
   her soruda ÇEKİMSER yazmaya itti (3 uydurma soruyla yapılan deneme, `scratch/rag_prompt_probe.py`). Bu denemelerde
   **hiçbir dev veya test sorusu kullanılmadı**; yalnız uydurma Sancaktepe/Ümraniye/Üsküdar soruları kullanıldı.
   Resmî dev ızgarası yalnız son metinlerle çalıştırıldı ve logs/rag_runs.jsonl'a yazıldı.
2. `requests` kütüphanesi kurulu değildi, `pip install requests` ile kuruldu.
3. Planlayıcı çağrısında `format: "json"` kullanıldı (yukarıda).
4. Seçim ölçütü (R1+R2 dev doğru yanıt toplamı; eşitlikte R1+R2 atıf ortalaması, küçük k, küçük prompt no) dev günlüğü
   okunmadan önce `src/rag_eval.py` içine yazılmıştı. P1/k=3 ve P1/k=5 her ikisi 16/20 aldı; eşitlik atıf ortalamasıyla kırıldı (0,437 > 0,417).
5. Seçim belgesinin SHA-256'sı `docs/protocol_freeze.txt` içinde (Sürüm 1.4) ve test koşusu bu kayıttan sonra yapıldı;
   günlük sırası `tests/test_rag.py` test (b) ile doğrulanır.
6. Yanıt değerlendirmesi otomatik kurallıdır (`src/rag_eval.py`); ayrıştırıcı hataları elle sınıflama sırasında ayrıca kontrol edilmiştir.
