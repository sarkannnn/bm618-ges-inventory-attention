"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 9b — RAG dizini (src/rag_index.py)

data/kb_docs.jsonl üzerinde kendi yazdığımız BM25 dizini (rank_bm25 gerekmez).
Bu dizin regresyon ortamından AYRIDIR: yalnızca kb_docs.jsonl okunur; audited_inventory,
splits.csv, inner_splits.csv vb. regresyon dosyalarına dokunulmaz.

Türkçe küçük harf kuralı: İ -> i, I -> ı (Python'un varsayılan lower() davranışı DEĞİL).
Basit kök bulma: her sözcüğün ilk 8 harfi (Türkçe ekler için kaba önek kökü; 8 harf,
Sultangazi / Sultanbeyli gibi çiftleri ayırmak için seçildi).

Dizine yalnızca ilçe, tesis adı ve dosya yılı girer. Çelişki etiketi, kalite bayrağı ve
varlık grubu (ENT_xxx) dizine ve LLM bağlamına GİRMEZ (altın etiket sızıntısı olmaması için).
"""

import os
import re
import sys
import json
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

KB_PATH = os.path.join(PROJECT_ROOT, "data", "kb_docs.jsonl")

STEM_LEN = 8
BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tr_lower(text):
    """Türkçe küçük harf: İ -> i, I -> ı, sonra standart lower()."""
    if text is None:
        return ""
    return str(text).replace("İ", "i").replace("I", "ı").lower()


def tokenize(text, stem=True):
    """Türkçe küçük harfe çevirir, sözcüklere ayırır, isteğe bağlı 8 harflik önek kökü uygular."""
    tokens = _TOKEN_RE.findall(tr_lower(text))
    if stem:
        return [t[:STEM_LEN] for t in tokens]
    return tokens


def load_kb_docs(path=None):
    path = path or KB_PATH
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def doc_index_text(doc):
    """Dizine girecek metin: ilçe (temiz + ham), tesis adı, dosya yılı."""
    parts = [doc.get("district_clean", ""), doc.get("district_raw", ""),
             doc.get("name_clean", ""), str(doc.get("file_year", ""))]
    return " ".join(str(p) for p in parts if p)


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75). Skoru 0 olan belgeler döndürülmez."""

    def __init__(self, docs):
        self.docs = docs
        self.ids = [d["record_id"] for d in docs]
        self.doc_tokens = [tokenize(doc_index_text(d)) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n_docs = len(docs)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.df = {}
        self.tf = []
        for toks in self.doc_tokens:
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query_terms, doc_idx):
        s = 0.0
        tf = self.tf[doc_idx]
        dl = self.doc_len[doc_idx]
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avgdl)
            s += self.idf(term) * f * (BM25_K1 + 1.0) / denom
        return s

    def search(self, query, k):
        """[(record_id, skor), ...] skora göre azalan, eşitlikte record_id'ye göre artan (deterministik)."""
        seen = []
        for t in tokenize(query):
            if t not in seen:
                seen.append(t)
        scored = []
        for i in range(self.n_docs):
            sc = self.score(seen, i)
            if sc > 0.0:
                scored.append((self.ids[i], sc))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def build_index(path=None):
    return BM25Index(load_kb_docs(path))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    idx = build_index()
    print(f"BM25 dizini hazır: {idx.n_docs} belge, {len(idx.df)} ayrı terim, ortalama uzunluk {idx.avgdl:.2f}")
