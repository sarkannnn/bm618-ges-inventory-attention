"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Adım 9b — RAG hesap aracı (src/rag_tools.py)

İzinli işlemler YALNIZ: filtrele, say, topla (pandas, kb_docs.jsonl üzerinden).
LLM keyfî kod ÇALIŞTIRAMAZ: LLM yalnız şu biçimde bir JSON planı üretir
    {"islem": "topla", "ilce": ..., "yil": ..., "ad_icerir": ...}
Kod bu planı doğrular (beyaz liste) ve yalnızca üç sabit fonksiyondan birini çağırır.
Bu modülde eval / exec / compile / os.system / subprocess / pandas.eval / DataFrame.query YOKTUR;
LLM'den gelen metinler yalnızca düz alt dize eşleştirmesinde (regex=False) kullanılır.
"""

import os
import re
import sys
import json

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_index import tr_lower, load_kb_docs, KB_PATH

ALLOWED_OPS = ("filtrele", "say", "topla")
ALLOWED_KEYS = ("islem", "ilce", "yil", "ad_icerir")
MAX_LIST_ITEMS = 5
MAX_TEXT_LEN = 80
MAX_ROWS_SHOWN = 30
_NULL_WORDS = ("", "null", "none", "yok", "-", "belirtilmemiş")


class PlanError(ValueError):
    """Plan beyaz listeye uymuyor; araç ÇALIŞTIRILMAZ."""


def _norm_text_list(value, field):
    """Metin ya da metin listesi -> temiz metin listesi veya None."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise PlanError(f"'{field}' metin, metin listesi veya null olmalı.")
    out = []
    for it in items:
        if it is None:
            continue
        if not isinstance(it, str):
            raise PlanError(f"'{field}' öğeleri metin olmalı.")
        s = it.strip()
        if tr_lower(s) in _NULL_WORDS:
            continue
        if len(s) > MAX_TEXT_LEN:
            raise PlanError(f"'{field}' öğesi en fazla {MAX_TEXT_LEN} karakter olabilir.")
        out.append(s)
    if len(out) > MAX_LIST_ITEMS:
        raise PlanError(f"'{field}' en fazla {MAX_LIST_ITEMS} öğe içerebilir.")
    return out or None


def validate_plan(plan):
    """Planı beyaz listeye göre doğrular, normalleştirilmiş sözlük döndürür; aksi halde PlanError."""
    if not isinstance(plan, dict):
        raise PlanError("Plan bir JSON nesnesi (sözlük) olmalı.")
    extra = [k for k in plan.keys() if k not in ALLOWED_KEYS]
    if extra:
        raise PlanError(f"İzin verilmeyen anahtar(lar): {extra}")
    islem = plan.get("islem")
    if not isinstance(islem, str) or islem not in ALLOWED_OPS:
        raise PlanError(f"İzin verilmeyen işlem: {islem!r}. İzinli işlemler: {list(ALLOWED_OPS)}")

    yil = plan.get("yil")
    if isinstance(yil, bool):
        raise PlanError("'yil' sayı olmalı.")
    if isinstance(yil, str):
        if tr_lower(yil.strip()) in _NULL_WORDS:
            yil = None
        elif re.fullmatch(r"\d{4}", yil.strip()):
            yil = int(yil.strip())
        else:
            raise PlanError("'yil' 4 haneli sayı olmalı.")
    if isinstance(yil, float):
        if yil != int(yil):
            raise PlanError("'yil' tam sayı olmalı.")
        yil = int(yil)
    if yil is not None:
        if not isinstance(yil, int) or not (1900 <= yil <= 2100):
            raise PlanError("'yil' 1900-2100 aralığında tam sayı olmalı.")

    return {
        "islem": islem,
        "ilce": _norm_text_list(plan.get("ilce"), "ilce"),
        "yil": yil,
        "ad_icerir": _norm_text_list(plan.get("ad_icerir"), "ad_icerir"),
    }


def parse_plan_text(text):
    """LLM çıktısından JSON nesnesini çıkarır; yalnızca json.loads kullanılır (kod çalıştırılmaz)."""
    if text is None:
        raise PlanError("Boş plan metni.")
    s = text.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            raise PlanError("Plan metninde JSON nesnesi bulunamadı.")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise PlanError(f"Geçersiz JSON: {e}")
    return obj


def load_kb_df(path=None):
    """kb_docs.jsonl -> pandas DataFrame (yalnızca araç için gerekli sütunlar)."""
    docs = load_kb_docs(path or KB_PATH)
    df = pd.DataFrame([{
        "record_id": d["record_id"],
        "yil": int(d["file_year"]),
        "excel_row": int(d["excel_row"]),
        "ilce": d["district_clean"],
        "ad": d["name_clean"],
        "guc_kwp": float(d["power_kwp"]),
        "durum": d.get("status_clean", ""),
    } for d in docs])
    df["_ilce_l"] = df["ilce"].map(tr_lower)
    df["_ad_l"] = df["ad"].map(tr_lower)
    return df


def _apply_filter(df, plan):
    """Tüm filtreler VE ile birleşir; liste içi öğeler VEYA ile birleşir. Düz alt dize eşleşmesi."""
    mask = pd.Series(True, index=df.index)
    if plan["yil"] is not None:
        mask &= (df["yil"] == plan["yil"])
    if plan["ilce"] is not None:
        wanted = {tr_lower(x).strip() for x in plan["ilce"]}
        mask &= df["_ilce_l"].isin(wanted)
    if plan["ad_icerir"] is not None:
        sub = pd.Series(False, index=df.index)
        for frag in plan["ad_icerir"]:
            sub |= df["_ad_l"].str.contains(tr_lower(frag).strip(), regex=False)
        mask &= sub
    return df[mask].sort_values(["yil", "excel_row"])


def _rows_of(sub):
    rows = []
    for r in sub.head(MAX_ROWS_SHOWN).itertuples(index=False):
        rows.append({"record_id": r.record_id, "yil": int(r.yil), "excel_row": int(r.excel_row),
                     "ilce": r.ilce, "ad": r.ad, "guc_kwp": float(r.guc_kwp), "durum": r.durum})
    return rows


def _op_filtrele(sub):
    return {"toplam_kwp": None}


def _op_say(sub):
    return {"toplam_kwp": None}


def _op_topla(sub):
    return {"toplam_kwp": round(float(sub["guc_kwp"].sum()), 2)}


# Yalnız bu üç işlem çağrılabilir; başka hiçbir işlev adı çözülmez.
TOOL_FUNCS = {"filtrele": _op_filtrele, "say": _op_say, "topla": _op_topla}


def run_tool(plan, df=None):
    """Planı doğrular ve izinli işlemi çalıştırır. İzinsiz plan -> PlanError (hiçbir şey çalışmaz)."""
    vp = validate_plan(plan)
    if df is None:
        df = load_kb_df()
    sub = _apply_filter(df, vp)
    extra = TOOL_FUNCS[vp["islem"]](sub)
    return {
        "islem": vp["islem"],
        "filtre": {"ilce": vp["ilce"], "yil": vp["yil"], "ad_icerir": vp["ad_icerir"]},
        "satir_sayisi": int(len(sub)),
        "toplam_kwp": extra["toplam_kwp"],
        "filtrelenen_idler": [str(x) for x in sub["record_id"].tolist()],
        "kayitlar": _rows_of(sub),
        "kesildi": bool(len(sub) > MAX_ROWS_SHOWN),
    }


def _filtre_text(f):
    return (f"ilçe={f['ilce'] if f['ilce'] else '-'}, yıl={f['yil'] if f['yil'] is not None else '-'}, "
            f"ad_içerir={f['ad_icerir'] if f['ad_icerir'] else '-'}")


def format_tool_result(result):
    """LLM bağlamına konacak Türkçe araç sonucu metni (kapsam: filtrelenen satır sayısı dahil)."""
    head = f"işlem: {result['islem']} | filtre: {_filtre_text(result['filtre'])} | filtrelenen satır sayısı: {result['satir_sayisi']}"
    if result["toplam_kwp"] is not None:
        head += f" | toplam kurulu güç: {result['toplam_kwp']} kWp"
    if result["satir_sayisi"] == 0:
        return head + " (eşleşen kayıt yok)"
    lines = [head, "Filtrelenen satırlar:"]
    for r in result["kayitlar"]:
        lines.append(f"- Dosya yılı: {r['yil']} | Excel satırı: {r['excel_row']} | İlçe: {r['ilce']} | "
                     f"Tesis: {r['ad']} | Güç: {r['guc_kwp']} kWp")
    if result["kesildi"]:
        lines.append(f"(İlk {MAX_ROWS_SHOWN} satır gösterildi; hesaplama tüm {result['satir_sayisi']} satır üzerinden yapıldı.)")
    return "\n".join(lines)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    d = load_kb_df()
    print(format_tool_result(run_tool({"islem": "topla", "ilce": "Bakırköy", "yil": 2022, "ad_icerir": None}, d)))
