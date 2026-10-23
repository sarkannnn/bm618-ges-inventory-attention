"""
BM618: Az Verili GES Envanterinde Attention, Transformer ve Kanıta Dayalı LLM
Görev 1.2 — Tesis Eşleştirme, Varlık Gruplama ve Çelişki Yönetimi (Entity Matching Module)
"""

import os
import sys
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import pandas as pd

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from normalize import (
    turkish_lower,
    turkish_upper,
    extract_sub_unit,
    normalize_facility_name,
    generate_entity_key
)

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')


class UnionFind:
    """Disjoint Set Union (DSU) with path compression and union by rank."""
    def __init__(self, elements: List[str]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, i: str) -> str:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: str, j: str) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False


def run_entity_matching(audited_csv_path: str, output_log_dir: str) -> pd.DataFrame:
    """
    Executes Görev 1.2 Entity Matching & Grouping:
    - Never uses power or status for matching decisions.
    - Preserves sub-units (Su Deposu 1/2/3, Atletizm Pisti, Şeflik, Tenis).
    - Clusters multi-stage / compound names to prevent train/test leakage:
      (Ataköy, İPA/Florya Atatürk Kent Ormanı, Başakşehir Engelliler Merkezi).
    - Links cross-district facility (Kumburgaz İtfaiye: 2023 Silivri, 2024-2025 Büyükçekmece).
    - Quarantines 2024 Çekmeköy Ömerli İtfaiye duplicate power conflicts (in_main_regression=False).
    - Assigns deterministic entity_group_id (ENT_001..ENT_XXX).
    - Writes logs/entity_match_log.csv and updates data/audited_inventory.csv.
    """
    if not os.path.exists(audited_csv_path):
        raise FileNotFoundError(f"Denetlenmiş envanter dosyası bulunamadı: {audited_csv_path}")

    df = pd.read_csv(audited_csv_path)
    records = df.to_dict('records')
    rec_ids = [r['record_id'] for r in records]
    uf = UnionFind(rec_ids)

    # Detailed pairwise merge logs and record-level match annotations
    match_events = []
    record_annotations = {}

    def register_match(rec_id1: str, rec_id2: str, rule: str, conf: str, just: str):
        uf.union(rec_id1, rec_id2)
        match_events.append({
            'record_id_1': rec_id1,
            'record_id_2': rec_id2,
            'match_rule': rule,
            'match_confidence': conf,
            'justification': just
        })

    # =========================================================================
    # 1. KURAL: İdari Sınır Değişimi Olan Tesis (Kumburgaz İtfaiye)
    # 2023 Silivri <-> 2024-2025 Büyükçekmece
    # =========================================================================
    kumburgaz_recs = [r['record_id'] for r in records if 'Kumburgaz İtfaiye' in r['name_clean']]
    for r_other in kumburgaz_recs[1:]:
        register_match(
            kumburgaz_recs[0], r_other,
            rule='CROSS_DISTRICT_FACILITY_MATCH',
            conf='düşük',
            just="Kumburgaz İtfaiye tesisi 2023'te Silivri, 2024-2025'te Büyükçekmece idari sınırında etiketlenmiştir; hedef/grup sızıntısını önlemek için aynı grupta birleştirildi."
        )

    # =========================================================================
    # 2. KURAL: Bakırköy Ataköy Birim ve Etap Değişimi
    # Kadın ve Aile (2022) -> Kadın Koordinasyon Etap 2 (2023) -> Sosyal Hizmetler (2025)
    # =========================================================================
    atakoy_recs = [r['record_id'] for r in records if r['district_clean'] == 'Bakırköy' and 'Ataköy' in r['name_clean']]
    for r_other in atakoy_recs[1:]:
        register_match(
            atakoy_recs[0], r_other,
            rule='MULTI_STAGE_NAME_CHANGE_GROUP',
            conf='düşük',
            just="Ataköy yerleşkesinde yıllar içinde birim adı ve etap değişimi (Kadın ve Aile -> Koordinasyon Etap 2 -> Sosyal Hizmetler); eğitim/test sızıntısını önlemek için aynı grupta tutuldu."
        )

    # =========================================================================
    # 3. KURAL: Bakırköy İPA ve Florya Atatürk Kent Ormanı
    # 2022'de birleşik yazılan isim, 2024 ve 2025'te ayrışmıştır
    # =========================================================================
    ipa_florya_recs = [r['record_id'] for r in records if r['district_clean'] == 'Bakırköy' and any(k in r['name_clean'] for k in ['Florya', 'Planlama Ajansı'])]
    for r_other in ipa_florya_recs[1:]:
        register_match(
            ipa_florya_recs[0], r_other,
            rule='COMPOUND_NAME_SPLIT_GROUP',
            conf='düşük',
            just="2022'de birleşik yazılan İPA - Florya Atatürk Kent Ormanı kaydı, 2024-2025'te iki ayrı satır olarak listelenmiştir; aynı fiziksel yerleşkede sızıntıyı önlemek için tek grupta toplandı."
        )

    # =========================================================================
    # 4. KURAL: Başakşehir Engelliler Merkezi Etapları
    # 1. Etap (2022), 2. Etap (2022), (2) (2023), Genel Tesis (2025)
    # =========================================================================
    engelliler_recs = [r['record_id'] for r in records if r['district_clean'] == 'Başakşehir' and 'Engelliler Merkezi' in r['name_clean']]
    for r_other in engelliler_recs[1:]:
        register_match(
            engelliler_recs[0], r_other,
            rule='MULTI_STAGE_FACILITY_GROUP',
            conf='düşük',
            just="Başakşehir Engelliler Merkezi 1. Etap, 2. Etap ve 2025 genel kaydı; etaplar arasında hedef sızıntısını önlemek için tek grupta tutuldu."
        )

    # =========================================================================
    # 5. KURAL: Çekmeköy Ömerli İtfaiye Çelişkisi ve Karantina
    # 2024 Excel 13 (29 kWp) ve Excel 15 (25 kWp), 2025 Excel 26 (28 kWp)
    # =========================================================================
    omerli_recs = [r['record_id'] for r in records if r['district_clean'] == 'Çekmeköy' and 'Ömerli İtfaiye' in r['name_clean']]
    for r_other in omerli_recs[1:]:
        register_match(
            omerli_recs[0], r_other,
            rule='SAME_FACILITY_YEAR_CONFLICT_GROUP',
            conf='yüksek',
            just="2024 Ömerli İtfaiye kayıtları (Excel 13 ve 15) ve 2025 kaydı aynı fiziksel tesis grubundadır; 2024 kayıtları çelişki nedeniyle karantinaya alınmıştır."
        )

    # =========================================================================
    # 6. KURAL: Maltepe Orhangazi Şehir Parkı Alt Birim Eşleştirmesi
    # Su Deposu Üzeri (1), (2), (3), Atletizm Pisti, Şeflik, Tenis Alanı
    # Kesinlikle birbiriyle birleştirilmez; 2024 ve 2025 birebir eşleşir.
    # =========================================================================
    maltepe_subunits = ['su_deposu_1', 'su_deposu_2', 'su_deposu_3', 'atletizm_pisti', 'seflik', 'tenis_alani']
    for su in maltepe_subunits:
        su_recs = [
            r['record_id'] for r in records
            if r['district_clean'] == 'Maltepe' and extract_sub_unit(turkish_lower(r['name_clean'])) == su
        ]
        if len(su_recs) > 1:
            for r_other in su_recs[1:]:
                register_match(
                    su_recs[0], r_other,
                    rule='EXACT_SUB_UNIT_MATCH',
                    conf='yüksek',
                    just=f"Maltepe Orhangazi Şehir Parkı '{su}' alt birimi 2024 ve 2025 yıllarında alt birim kimliği korunarak birebir eşleşti."
                )

    # =========================================================================
    # 7. KURAL: Özel İsim ve Yerleşke Eşleştirmeleri (Sözcük ve Kapsam Doğrulamalı)
    # =========================================================================
    # Esenler Kemer Spor Kompleksi:
    r_kemer_22 = [r['record_id'] for r in records if r['district_clean'] == 'Esenler' and r['file_year'] == 2022 and 'Kemer Kamp' in r['name_clean']][0]
    r_kemer_25 = [r['record_id'] for r in records if r['district_clean'] == 'Esenler' and r['file_year'] == 2025 and 'Kemer Spor' in r['name_clean']][0]
    register_match(
        r_kemer_22, r_kemer_25,
        rule='SIMPLIFIED_GENERIC_NAME_UNIQUE_IN_DISTRICT',
        conf='orta',
        just="Esenler Spor Kompleksi (Kemer Kamp Eğitim Merkezi) 2022 kaydı ile Kemer Spor Kompleksi 2025 kaydı aynı tesistir."
    )

    # Esenler Turgut Reis İtfaiye:
    r_tr_23 = [r['record_id'] for r in records if r['district_clean'] == 'Esenler' and r['file_year'] == 2023 and 'Turgut Reis' in r['name_clean']][0]
    r_tr_25 = [r['record_id'] for r in records if r['district_clean'] == 'Esenler' and r['file_year'] == 2025 and 'Turgut Reis' in r['name_clean']][0]
    register_match(
        r_tr_23, r_tr_25,
        rule='EXACT_FACILITY_MATCH',
        conf='yüksek',
        just="Esenler Turgut Reis İtfaiye tesisi 2023 ve 2025 kayıtları birebir eşleşti."
    )

    # Kağıthane Cemal Kamacı Spor Kompleksi:
    r_ck_22 = [r['record_id'] for r in records if r['district_clean'] == 'Kağıthane' and r['file_year'] == 2022 and 'Cemal Kamacı' in r['name_clean']][0]
    r_ck_25 = [r['record_id'] for r in records if r['district_clean'] == 'Kağıthane' and r['file_year'] == 2025 and 'Cemal Kamacı' in r['name_clean']][0]
    register_match(
        r_ck_22, r_ck_25,
        rule='MULTI_STAGE_FACILITY_GROUP',
        conf='orta',
        just="Kağıthane Cemal Kamacı Spor Kompleksi (2022'de 2. Etap, 2025'te tesis geneli); sızıntıyı önlemek için aynı grupta toplandı."
    )

    # Ataşehir Kayışdağı Darulacize:
    r_darul_22 = [r['record_id'] for r in records if r['district_clean'] == 'Ataşehir' and r['file_year'] == 2022 and 'Darulacize' in r['name_clean']][0]
    r_darul_25 = [r['record_id'] for r in records if r['district_clean'] == 'Ataşehir' and r['file_year'] == 2025 and 'Darulacize' in r['name_clean']][0]
    register_match(
        r_darul_22, r_darul_25,
        rule='EXACT_FACILITY_MATCH',
        conf='yüksek',
        just="Ataşehir Kayışdağı Darulacize yerleşkesi 2022 ve 2025 kayıtları birebir eşleşti."
    )

    # Zeytinburnu Topkapı Deprem Parkı Tuvaleti:
    r_topkapi_23 = [r['record_id'] for r in records if r['district_clean'] == 'Zeytinburnu' and r['file_year'] == 2023 and 'Topkapı' in r['name_clean']][0]
    r_topkapi_25 = [r['record_id'] for r in records if r['district_clean'] == 'Zeytinburnu' and r['file_year'] == 2025 and 'Topkapı' in r['name_clean']][0]
    register_match(
        r_topkapi_23, r_topkapi_25,
        rule='EXACT_FACILITY_MATCH',
        conf='yüksek',
        just="Zeytinburnu Topkapı Deprem Parkı Tuvaleti 2023 ve 2025 kayıtları birebir eşleşti."
    )

    # =========================================================================
    # 8. KURAL: İlçe İçi Normalize Edilmiş Ad ve Tekil Genel Ad Eşleştirmesi
    # =========================================================================
    already_matched_ids = set()
    for ev in match_events:
        already_matched_ids.add(ev['record_id_1'])
        already_matched_ids.add(ev['record_id_2'])

    for d, grp in df.groupby('district_clean'):
        cand = [r for idx, r in grp.iterrows() if r['record_id'] not in already_matched_ids]
        if len(cand) <= 1:
            continue

        norm_dict = defaultdict(list)
        for r in cand:
            norm = normalize_facility_name(r['name_clean'])
            d_low = turkish_lower(d)
            if norm.startswith(d_low):
                norm = norm[len(d_low):].strip()
            norm_dict[norm].append(r)

        for norm_key, r_list in norm_dict.items():
            if len(r_list) > 1:
                for r_other in r_list[1:]:
                    is_generic = norm_key in ['spor kompleksi', 'yüzme havuzu', 'itfaiye', 'mezarlığı hizmet ve gasilhane', 'kültür']
                    conf = 'orta' if is_generic else 'yüksek'
                    rule = 'SIMPLIFIED_GENERIC_NAME_UNIQUE_IN_DISTRICT' if is_generic else 'EXACT_FACILITY_MATCH'
                    just = f"{d} ilçesinde '{norm_key}' tesisi yıllar arasında eşleşti (Güven: {conf})."
                    register_match(r_list[0]['record_id'], r_other['record_id'], rule, conf, just)

    # =========================================================================
    # 9. Kümeleme ve Belirleyici Grup Kimliklerinin (ENT_001..ENT_XXX) Atanması
    # Sıralama: İlk ortaya çıkış yılı ve Excel satırına göre deterministik
    # =========================================================================
    raw_clusters = defaultdict(list)
    for idx, r in df.iterrows():
        root = uf.find(r['record_id'])
        raw_clusters[root].append(r)

    # Sort clusters chronologically by min(file_year, excel_row)
    sorted_cluster_roots = sorted(
        raw_clusters.keys(),
        key=lambda root_id: (
            min(r['file_year'] for r in raw_clusters[root_id]),
            min(r['excel_row'] for r in raw_clusters[root_id])
        )
    )

    group_id_map = {}
    for idx, root_id in enumerate(sorted_cluster_roots, start=1):
        group_id_map[root_id] = f"ENT_{idx:03d}"

    # Build final annotations per record
    final_records = []
    match_log_rows = []

    # Map each record's merge events to get confidence and rules
    rec_event_map = defaultdict(list)
    for ev in match_events:
        rec_event_map[ev['record_id_1']].append(ev)
        rec_event_map[ev['record_id_2']].append(ev)

    for idx, r in df.iterrows():
        r_id = r['record_id']
        root = uf.find(r_id)
        ent_group_id = group_id_map[root]
        cluster_members = raw_clusters[root]
        group_size = len(cluster_members)

        # Quarantined conflict check
        # Çekmeköy Ömerli İtfaiye 2024 (REC_2024_R13, REC_2024_R15)
        is_quarantined = (r_id in ['REC_2024_R13', 'REC_2024_R15'])
        in_main_regression = (not is_quarantined)

        # Confidence and rule assignment
        if group_size == 1:
            match_conf = 'tekil'
            match_rule = 'SINGLETON_NO_MATCH'
            justification = f"{r['district_clean']} ilçesinde tekil envanter kaydı; başka bir yıl ile eşleşmedi."
        else:
            # If any event in the cluster is 'düşük', entire cluster confidence is 'düşük' (conservative principle)
            # If any event is 'orta', confidence is 'orta'
            cluster_events = []
            for m in cluster_members:
                cluster_events.extend(rec_event_map.get(m['record_id'], []))
            
            confs = [e['match_confidence'] for e in cluster_events]
            if 'düşük' in confs:
                match_conf = 'düşük'
            elif 'orta' in confs:
                match_conf = 'orta'
            else:
                match_conf = 'yüksek'

            rules = list(set([e['match_rule'] for e in cluster_events]))
            match_rule = ";".join(rules) if rules else 'EXACT_FACILITY_MATCH'
            justs = list(set([e['justification'] for e in cluster_events]))
            justification = "; ".join(justs)

        matched_with = [m['record_id'] for m in cluster_members if m['record_id'] != r_id]
        matched_with_str = ",".join(matched_with) if matched_with else "YOK"

        # Update row dict
        r_dict = dict(r)
        r_dict['entity_group_id'] = ent_group_id
        r_dict['in_main_regression'] = in_main_regression
        r_dict['match_confidence'] = match_conf
        final_records.append(r_dict)

        # Match log entry
        match_log_rows.append({
            'record_id': r_id,
            'file_year': r['file_year'],
            'district_clean': r['district_clean'],
            'name_clean': r['name_clean'],
            'entity_group_id': ent_group_id,
            'group_size': group_size,
            'match_confidence': match_conf,
            'match_rule': match_rule,
            'justification': justification,
            'in_main_regression': in_main_regression,
            'quarantine_reason': 'AYNI_YIL_GUC_CELISKISI_2024_OMERLI' if is_quarantined else 'OK',
            'matched_with_records': matched_with_str
        })

    # Save updated audited inventory
    updated_df = pd.DataFrame(final_records)
    updated_df.to_csv(audited_csv_path, index=False, encoding='utf-8')

    # Save match log
    os.makedirs(output_log_dir, exist_ok=True)
    match_log_path = os.path.join(output_log_dir, 'entity_match_log.csv')
    match_log_df = pd.DataFrame(match_log_rows)
    match_log_df.to_csv(match_log_path, index=False, encoding='utf-8')

    return updated_df, match_log_df


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "audited_inventory.csv")
    logs_dir = os.path.join(base_dir, "logs")

    df_out, log_out = run_entity_matching(csv_path, logs_dir)
    print("\nEntity matching execution finished successfully.")
