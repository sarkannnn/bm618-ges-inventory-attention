"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Metin ve Varlık Anahtarı Normalizasyon Modülü (src/normalize.py)
"""

import re
from typing import Tuple

TURKISH_LOWER_MAP = {'İ': 'i', 'I': 'ı'}
TURKISH_UPPER_MAP = {'i': 'İ', 'ı': 'I'}


def turkish_lower(text: str) -> str:
    """Turkish-aware lowercase conversion."""
    if not text:
        return ""
    s = "".join(TURKISH_LOWER_MAP.get(c, c) for c in str(text))
    return s.lower().strip()


def turkish_upper(text: str) -> str:
    """Turkish-aware uppercase conversion."""
    if not text:
        return ""
    s = "".join(TURKISH_UPPER_MAP.get(c, c) for c in str(text))
    return s.upper().strip()


def extract_sub_unit(name_lower: str) -> str:
    """
    Extracts distinct sub-units, sections, or stages to prevent improper merges.
    Orhangazi Şehir Parkı: Su Deposu Üzeri (1), (2), (3), Atletizm Pisti, Şeflik, Tenis Alanı.
    Etaplar: Etap 1, Etap 2 vb.
    """
    # Orhangazi Şehir Parkı sub-units
    if 'su deposu üzeri (1)' in name_lower or 'su deposu üzeri 1' in name_lower or 'su deposu 1' in name_lower:
        return 'su_deposu_1'
    if 'su deposu üzeri (2)' in name_lower or 'su deposu üzeri 2' in name_lower or 'su deposu 2' in name_lower:
        return 'su_deposu_2'
    if 'su deposu üzeri (3)' in name_lower or 'su deposu üzeri 3' in name_lower or 'su deposu 3' in name_lower:
        return 'su_deposu_3'
    if 'atletizm pisti' in name_lower:
        return 'atletizm_pisti'
    if 'şeflik' in name_lower:
        return 'seflik'
    if 'tenis alanı' in name_lower:
        return 'tenis_alani'

    # Stages (Etap)
    if '1. etap' in name_lower or 'etap 1' in name_lower or '1.etap' in name_lower:
        return 'etap_1'
    if '2. etap' in name_lower or 'etap 2' in name_lower or '2.etap' in name_lower or '(2)' in name_lower:
        return 'etap_2'

    return ''


def normalize_facility_name(name: str) -> str:
    """
    Normalizes facility name by removing common institutional suffixes and boilerplate words:
    - Güneş Enerji Santrali / GES
    - Bina / Binası / Çatısı / İstasyonu
    - İBB / Müdürlüğü vb.
    Preserves core entity identity tokens.
    """
    s = turkish_lower(name)
    
    # Strip standalone acronyms and annotations
    s = s.replace('(ges)', ' ').replace('(traker)', ' ').replace('(millet bahçesi)', ' ')
    
    # Common boilerplate terms
    boilerplate = [
        'güneş enerji santrali', 'güneş enerji santralleri',
        'güneş enerji sistemi', 'güneş enerji sistemleri',
        'bina çatısı', 'bina', 'binası', 'çatısı', 'çatı',
        'müdürlüğü', 'müdürlük', 'istasyonu',
        'tesis', 'tesisi', 'tesisleri', 'ibb', 'ges'
    ]
    for b in boilerplate:
        s = re.sub(r'\b' + re.escape(b) + r'\b', ' ', s)

    # Clean punctuation and multiple whitespaces
    s = re.sub(r'[\(\)\-\/\,\.]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def generate_entity_key(district_clean: str, name_clean: str) -> Tuple[str, str, str]:
    """
    Generates deterministic tuple: (district_key, normalized_name, sub_unit)
    """
    dist_key = turkish_lower(district_clean)
    s_low = turkish_lower(name_clean)
    sub_unit = extract_sub_unit(s_low)
    norm_name = normalize_facility_name(name_clean)
    return dist_key, norm_name, sub_unit
