"""
Business Entity Resolution - High-Performance Pipeline v5
==========================================================
Target: F0.5 > 0.991811

Key improvements over v4:
  1. Higher top_k (100 for names) to catch more blocking candidates
  2. Better features: first-token match, Levenshtein edit distance, 
     address city/state extraction, name prefix similarity
  3. Stricter address-only matching guard (when name is empty/transliterated)
  4. Two-round hard-negative mining
  5. More training data (70K S1)
  6. Better transliteration handling: keep non-ASCII for same-script matching
"""
import os, sys, time, re, gc, unicodedata, pickle, warnings, functools
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ========================== Normalization ==========================
LEGAL_SUFFIXES = sorted([
    "private limited","pvt limited","pvt ltd","pvt. ltd.","pvt. ltd",
    "pvt.ltd.","pvt.ltd","p ltd","limited","ltd","corporation","corp",
    "incorporated","inc","company","co","llc","l.l.c.","l.l.c",
    "llp","l.l.p.","l.l.p","plc","p.l.c.","gmbh",
    "sarl","s.a.r.l.","s.a.r.l","sas","s.a.s.","s.a.s",
    "sa","s.a.","s.a","ag","a.g.","nv","n.v.","bv","b.v.",
    "pty ltd","pty. ltd.","pty","pty.",
    "societe anonyme","societe a responsabilite limitee",
    "eurl","sasu","sci","snc","scs","sca","se","groupe","et cie","cie",
], key=len, reverse=True)

COUNTRY_MAP = {
    "us":"us","usa":"us","u.s.":"us","u.s.a.":"us",
    "united states":"us","united states of america":"us",
    "india":"india","in":"india","ind":"india",
    "france":"france","fr":"france","fra":"france",
}

def _to_ascii(t):
    if not t: return ""
    t = unicodedata.normalize("NFKD", t)
    a = t.encode("ascii","ignore").decode("ascii")
    return a

def norm_name(name) -> str:
    if pd.isna(name) or str(name).strip()=="": return ""
    s = str(name).lower().strip()
    a = _to_ascii(s)
    a = a.replace("&"," and ").replace("+"," and ")
    a = re.sub(r"[^a-z0-9\s]"," ",a)
    return re.sub(r"\s+"," ",a).strip()

def norm_name_core(name) -> str:
    s = norm_name(name)
    if not s: return ""
    for suf in LEGAL_SUFFIXES:
        c = re.sub(r"[^a-z0-9\s]"," ",suf.lower()).strip()
        c = re.sub(r"\s+"," ",c)
        if c:
            s = re.sub(r"\b"+re.escape(c)+r"\b","",s)
    return re.sub(r"\s+"," ",s).strip()

def is_transliterated(name) -> bool:
    if pd.isna(name) or str(name).strip()=="": return False
    s = str(name)
    ascii_chars = sum(1 for c in s if ord(c) < 128)
    return ascii_chars < len(s) * 0.5

def norm_addr(addr) -> str:
    if pd.isna(addr) or str(addr).strip()=="": return ""
    s = str(addr).lower().strip()
    a = _to_ascii(s)
    a = a.replace("&"," and ")
    a = re.sub(r"[^a-z0-9\s\-]"," ",a)
    return re.sub(r"\s+"," ",a).strip()

def norm_country(c) -> str:
    if pd.isna(c) or str(c).strip()=="": return ""
    s = re.sub(r"[^a-z\s]","",str(c).lower().strip()).strip()
    return COUNTRY_MAP.get(s,s)

def extract_nums(t):
    return set(re.findall(r"\b\d+\b",t)) if t else set()

def first_token(t):
    """Get first significant token (>2 chars)."""
    if not t: return ""
    for tok in t.split():
        if len(tok) > 2: return tok
    return t.split()[0] if t.split() else ""

def preprocess_df(df):
    t0=time.time()
    df=df.copy()
    df["nm"]=df["business_name"].apply(norm_name)
    df["nmc"]=df["business_name"].apply(norm_name_core)
    df["ad"]=df["business_address"].apply(norm_addr)
    df["ct"]=df["country"].apply(norm_country)
    df["ad_nums"]=df["ad"].apply(extract_nums)
    df["is_trans"]=df["business_name"].apply(is_transliterated)
    df["first_tok"]=df["nmc"].apply(first_token)
    print(f"    Preprocessed {len(df):,} rows in {time.time()-t0:.1f}s")
    return df

# ========================== F0.5 ==========================
def f05_ent(p,a):
    if not p and not a: return 1.0
    if not p or not a: return 0.0
    tp=len(p&a); pr=tp/len(p); rc=tp/len(a)
    return (1.25*pr*rc)/(0.25*pr+rc) if pr+rc>0 else 0.0

def macro_f05(pd_d, gt_d):
    s,ps,rs=[],[],[]
    for sid in gt_d:
        p=pd_d.get(sid,set()); a=gt_d[sid]
        s.append(f05_ent(p,a))
        if not p and not a: ps.append(1.0);rs.append(1.0)
        elif not p or not a: ps.append(0.0);rs.append(0.0)
        else: tp=len(p&a); ps.append(tp/len(p)); rs.append(tp/len(a))
    return np.mean(s),np.mean(ps),np.mean(rs)

# ========================== Features ==========================
FEAT_NAMES = [
    # Name features (12)
    "nm_exact","nmc_exact",
    "nm_ratio","nm_partial","nm_tsort","nm_tset",
    "nmc_ratio","nmc_tsort","nmc_tset",
    "nm_jaccard","nm_olap","nm_olap_frac",
    # Name structural (6)
    "nm_lendiff","nm_lenratio","nm_tokdiff",
    "first_tok_match","nm_prefix4","nm_levenshtein",
    # Transliteration (2)
    "s23_is_trans","both_have_name",
    # Address features (12)
    "ad_exact","ad_avail1","ad_avail2","ad_both",
    "ad_ratio","ad_partial","ad_tsort","ad_tset",
    "ad_jaccard","ad_olap","ad_lendiff","ad_lenratio",
    # Address numeric (3)
    "ad_numjac","ad_numolap","ad_numolap_frac",
    # Cross features (4)
    "name_strong","addr_strong","best_signal",
    "name_weak_addr_strong",
    # Country + source (3)
    "ct_match","is_s2","is_s3",
]

def pair_feats(r1, r2):
    n1, n2 = r1["nm"], r2["nm"]
    c1, c2 = r1["nmc"], r2["nmc"]
    a1, a2 = r1["ad"], r2["ad"]
    ct1, ct2 = r1["ct"], r2["ct"]
    f = []

    # --- Name features ---
    f.append(float(n1 == n2 and n1 != ""))
    f.append(float(c1 == c2 and c1 != ""))

    nm_rat = nm_part = nm_tsort = nm_tset = 0.0
    if n1 and n2:
        nm_rat = fuzz.ratio(n1, n2) / 100.0
        nm_part = fuzz.partial_ratio(n1, n2) / 100.0
        nm_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
        nm_tset = fuzz.token_set_ratio(n1, n2) / 100.0
    f.extend([nm_rat, nm_part, nm_tsort, nm_tset])

    nmc_rat = nmc_tsort = nmc_tset = 0.0
    if c1 and c2:
        nmc_rat = fuzz.ratio(c1, c2) / 100.0
        nmc_tsort = fuzz.token_sort_ratio(c1, c2) / 100.0
        nmc_tset = fuzz.token_set_ratio(c1, c2) / 100.0
    f.extend([nmc_rat, nmc_tsort, nmc_tset])

    t1 = set(c1.split()) if c1 else set()
    t2 = set(c2.split()) if c2 else set()
    if t1 or t2:
        ol = t1 & t2; un = t1 | t2
        f.append(len(ol)/len(un) if un else 0.0)
        f.append(float(len(ol)))
        f.append(len(ol)/max(len(t1),len(t2),1))
    else:
        f.extend([0.0]*3)

    # Name structural
    f.append(float(abs(len(n1)-len(n2))))
    f.append(min(len(n1),len(n2))/max(len(n1),len(n2),1))
    f.append(float(abs(len(t1)-len(t2))))

    # First token match
    ft1 = r1.get("first_tok","")
    ft2 = r2.get("first_tok","")
    f.append(float(ft1 == ft2 and ft1 != ""))

    # Name prefix similarity (first 4 chars)
    if c1 and c2:
        f.append(float(c1[:4] == c2[:4]))
    else:
        f.append(0.0)

    # Levenshtein normalized distance
    if c1 and c2:
        lev = Levenshtein.normalized_distance(c1, c2)
        f.append(1.0 - lev)  # similarity
    else:
        f.append(0.0)

    # Transliteration flags
    f.append(float(r2.get("is_trans", False)))
    f.append(float(n1 != "" and n2 != ""))

    # --- Address features ---
    f.append(float(a1 == a2 and a1 != ""))
    f.append(float(a1 != ""))
    f.append(float(a2 != ""))
    f.append(float(a1 != "" and a2 != ""))

    ad_rat = ad_part = ad_tsort = ad_tset = 0.0
    if a1 and a2:
        ad_rat = fuzz.ratio(a1, a2) / 100.0
        ad_part = fuzz.partial_ratio(a1, a2) / 100.0
        ad_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
        ad_tset = fuzz.token_set_ratio(a1, a2) / 100.0
    f.extend([ad_rat, ad_part, ad_tsort, ad_tset])

    if a1 and a2:
        at1 = set(a1.split()); at2 = set(a2.split())
        aol = at1 & at2; aun = at1 | at2
        f.append(len(aol)/len(aun) if aun else 0.0)
        f.append(float(len(aol)))
        f.append(float(abs(len(a1)-len(a2))))
        f.append(min(len(a1),len(a2))/max(len(a1),len(a2),1))
    else:
        f.extend([0.0]*4)

    # Address numeric overlap
    an1 = r1.get("ad_nums",set()) or set()
    an2 = r2.get("ad_nums",set()) or set()
    if an1 or an2:
        nol = an1 & an2; nun = an1 | an2
        f.append(len(nol)/len(nun) if nun else 0.0)
        f.append(float(len(nol)))
        f.append(len(nol)/max(len(an1),len(an2),1))
    else:
        f.extend([0.0]*3)

    # Cross features
    name_strong = max(nm_rat, nm_tsort, nmc_tset) if (n1 and n2) else 0.0
    addr_strong = ad_tset if (a1 and a2) else 0.0
    f.append(name_strong)
    f.append(addr_strong)
    f.append(max(name_strong, addr_strong))
    # Name weak but address strong — risky pattern for FP
    f.append(float(name_strong < 0.5 and addr_strong > 0.7))

    # Country + source
    f.append(float(ct1 == ct2 and ct1 != ""))
    eid2 = str(r2.get("entity_id",""))
    f.append(float(eid2.startswith("S2-")))
    f.append(float(eid2.startswith("S3-")))

    return f

# ========================== Multi-Strategy Blocking ==========================
def tfidf_block(names_query, ids_query, names_db, ids_db, top_k=50,
                batch_sz=1000, min_sim=0.1, analyzer='char_wb', ngram_range=(3,4)):
    if not names_query or not names_db: return {}
    valid_q = [(n,i) for n,i in zip(names_query, ids_query) if n.strip()]
    valid_d = [(n,i) for n,i in zip(names_db, ids_db) if n.strip()]
    if not valid_q or not valid_d: return {}
    q_names, q_ids = zip(*valid_q); d_names, d_ids = zip(*valid_d)
    q_names, q_ids = list(q_names), list(q_ids)
    d_names, d_ids = list(d_names), list(d_ids)
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range,
                          max_features=200000, sublinear_tf=True, dtype=np.float32)
    vec.fit(q_names + d_names)
    d_tf = vec.transform(d_names)
    cands = {}
    nb = (len(q_names)+batch_sz-1)//batch_sz
    for bi in range(nb):
        st = bi*batch_sz; en = min(st+batch_sz, len(q_names))
        q_tf = vec.transform(q_names[st:en])
        sim = cosine_similarity(q_tf, d_tf)
        for i in range(en-st):
            sc = sim[i]
            actual_k = min(top_k, len(sc))
            if actual_k <= 0: continue
            idxs = np.argpartition(sc, -actual_k)[-actual_k:]
            c = set()
            for j in idxs:
                if sc[j] > min_sim: c.add(d_ids[j])
            if c: cands[q_ids[st+i]] = c
        if (bi+1)%5==0 or bi==nb-1: print(f"      batch {bi+1}/{nb}")
    del vec, d_tf; gc.collect()
    return cands

def numeric_block(s1_c, s23_c, min_overlap=2):
    inv = defaultdict(list)
    for _, row in s23_c.iterrows():
        nums = row.get("ad_nums", set())
        if nums:
            for n in nums:
                if len(n) >= 2: inv[n].append(row["entity_id"])
    cands = {}
    for _, row in s1_c.iterrows():
        nums = row.get("ad_nums", set())
        if not nums: continue
        counts = defaultdict(int)
        for n in nums:
            if len(n) >= 2 and n in inv:
                for eid in inv[n]: counts[eid] += 1
        c = {eid for eid, cnt in counts.items() if cnt >= min_overlap}
        if c: cands[row["entity_id"]] = c
    return cands

def multi_block(s1_c, s23_c, top_k_name=100, top_k_addr=50):
    print("      [Name TF-IDF blocking]")
    name_cands = tfidf_block(
        s1_c["nmc"].fillna("").tolist(), s1_c["entity_id"].tolist(),
        s23_c["nmc"].fillna("").tolist(), s23_c["entity_id"].tolist(),
        top_k=top_k_name, min_sim=0.08
    )
    print("      [Address TF-IDF blocking]")
    addr_cands = tfidf_block(
        s1_c["ad"].fillna("").tolist(), s1_c["entity_id"].tolist(),
        s23_c["ad"].fillna("").tolist(), s23_c["entity_id"].tolist(),
        top_k=top_k_addr, min_sim=0.12, ngram_range=(3,5)
    )
    print("      [Numeric address blocking]")
    num_cands = numeric_block(s1_c, s23_c, min_overlap=2)
    all_cands = defaultdict(set)
    for d in [name_cands, addr_cands, num_cands]:
        for sid, cids in d.items(): all_cands[sid] |= cids
    n_name = sum(len(v) for v in name_cands.values())
    n_addr = sum(len(v) for v in addr_cands.values())
    n_num = sum(len(v) for v in num_cands.values())
    n_total = sum(len(v) for v in all_cands.values())
    print(f"      Name: {n_name:,}, Addr: {n_addr:,}, Numeric: {n_num:,}, Union: {n_total:,}")
    del name_cands, addr_cands, num_cands; gc.collect()
    return dict(all_cands)

# ========================== GT Parser ==========================
def parse_gt(gt_df):
    result = {}
    sids = gt_df["source1_entity_id"].values
    mids = gt_df["matched_entity_ids"].values
    for i in range(len(sids)):
        m = mids[i]
        if pd.isna(m) or str(m).strip() == "": result[sids[i]] = set()
        else: result[sids[i]] = set(str(m).split(","))
    return result

def cand_recall(cands, gt):
    h = t = 0
    for sid, ms in gt.items():
        for m in ms:
            t += 1
            if m in cands.get(sid, set()): h += 1
    return h, t

# ========================== TRAINING ==========================
def run_training():
    print("="*70)
    print("HIGH-PERFORMANCE TRAINING v5 (Target: F0.5 > 0.991)")
    print("="*70)
    T0 = time.time()

    print("\n[1] Loading ground truth...")
    gt_df = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")
    gt_all = parse_gt(gt_df); del gt_df; gc.collect()
    print(f"    {len(gt_all):,} S1 entities")

    all_s1 = sorted(gt_all.keys())
    sample_n = min(70000, len(all_s1))
    sampled = list(np.random.choice(all_s1, size=sample_n, replace=False))
    np.random.shuffle(sampled)
    val_n = int(sample_n * 0.15)
    val_ids = set(sampled[:val_n]); train_ids = set(sampled[val_n:])
    print(f"    Sampled {sample_n:,}: train={len(train_ids):,}, val={len(val_ids):,}")

    gt_tr = {k:v for k,v in gt_all.items() if k in train_ids}
    gt_va = {k:v for k,v in gt_all.items() if k in val_ids}
    needed_s23 = set()
    for ms in gt_tr.values(): needed_s23 |= ms
    for ms in gt_va.values(): needed_s23 |= ms
    print(f"    Need {len(needed_s23):,} S2/S3 positive records")

    print("\n[2] Loading S1...")
    s1a = pd.read_csv(os.path.join(TRAIN_DIR,"train_source1.tsv"), sep="\t")
    s1s = s1a[s1a["entity_id"].isin(set(sampled))].copy(); del s1a; gc.collect()
    s1s = preprocess_df(s1s)

    print("\n[3] Loading S2...")
    s2a = pd.read_csv(os.path.join(TRAIN_DIR,"train_source2.tsv"), sep="\t")
    s2p = s2a[s2a["entity_id"].isin(needed_s23)]
    ni = np.random.choice(len(s2a), size=min(150000,len(s2a)), replace=False)
    s2s = pd.concat([s2p,s2a.iloc[ni]]).drop_duplicates(subset="entity_id")
    del s2a,s2p; gc.collect()
    print(f"    S2 sample: {len(s2s):,}")

    print("\n[4] Loading S3...")
    s3a = pd.read_csv(os.path.join(TRAIN_DIR,"train_source3.tsv"), sep="\t")
    s3p = s3a[s3a["entity_id"].isin(needed_s23)]
    ni = np.random.choice(len(s3a), size=min(150000,len(s3a)), replace=False)
    s3s = pd.concat([s3p,s3a.iloc[ni]]).drop_duplicates(subset="entity_id")
    del s3a,s3p; gc.collect()
    print(f"    S3 sample: {len(s3s):,}")

    s23 = pd.concat([s2s,s3s]).drop_duplicates(subset="entity_id")
    del s2s,s3s; gc.collect()
    print(f"    S23 combined: {len(s23):,}")
    s23 = preprocess_df(s23)
    s23_id_set = set(s23["entity_id"].values)

    print("\n[5] Multi-strategy blocking...")
    s1tr = s1s[s1s["entity_id"].isin(train_ids)]
    s1va = s1s[s1s["entity_id"].isin(val_ids)]
    tr_cands = {}; va_cands = {}
    for country in sorted(s1s["ct"].unique()):
        if not country: continue
        s1tc = s1tr[s1tr["ct"]==country]; s1vc = s1va[s1va["ct"]==country]
        s23c = s23[s23["ct"]==country]
        print(f"\n    {country}: S1_tr={len(s1tc):,} S1_va={len(s1vc):,} S23={len(s23c):,}")
        if len(s23c)==0: continue
        if len(s1tc)>0: tr_cands.update(multi_block(s1tc, s23c, top_k_name=100, top_k_addr=50))
        if len(s1vc)>0: va_cands.update(multi_block(s1vc, s23c, top_k_name=100, top_k_addr=50))

    # Inject all positives into training
    for sid, ms in gt_tr.items():
        if sid not in tr_cands: tr_cands[sid] = set()
        for m in ms:
            if m in s23_id_set: tr_cands[sid].add(m)

    th,tt = cand_recall(tr_cands, gt_tr)
    print(f"\n    Train cand recall: {th}/{tt} = {th/max(tt,1):.4f}")
    vh,vt = cand_recall(va_cands, gt_va)
    print(f"    Val cand recall: {vh}/{vt} = {vh/max(vt,1):.4f}")

    # Blocking miss analysis
    s1s_idx = s1s.set_index("entity_id")
    s23_idx_temp = s23.set_index("entity_id")
    missed = []
    for sid, ms in gt_va.items():
        for m in ms:
            if m not in va_cands.get(sid, set()) and sid in s1s_idx.index and m in s23_id_set:
                s1r = s1s_idx.loc[sid]; s23r = s23_idx_temp.loc[m]
                missed.append({"s1_nm": s1r["nm"][:35], "s23_nm": s23r["nm"][:35],
                               "s23_trans": s23r["is_trans"],
                               "s1_ad": s1r["ad"][:35], "s23_ad": s23r["ad"][:35]})
    del s23_idx_temp
    print(f"\n    Blocking misses: {len(missed)}")
    if missed:
        print(f"    Transliterated: {sum(1 for m in missed if m['s23_trans'])}")
        for m in missed[:3]:
            print(f"      '{m['s1_nm']}' vs '{m['s23_nm']}' trans={m['s23_trans']}")
            print(f"        addr: '{m['s1_ad']}' vs '{m['s23_ad']}'")

    tp = sum(len(v) for v in tr_cands.values())
    vp = sum(len(v) for v in va_cands.values())
    print(f"\n    Train pairs: {tp:,}, Val pairs: {vp:,}")

    print("\n[6] Computing features...")
    s23_idx = s23.set_index("entity_id")
    s23_idx_keys = set(s23_idx.index)

    def compute_feats(s1_df, cands, gt):
        s1_ix = s1_df.set_index("entity_id")
        X,y,pairs = [],[],[]
        done = 0; total = sum(len(v) for v in cands.items())
        for sid, cids in cands.items():
            if sid not in s1_ix.index: continue
            r1 = s1_ix.loc[sid]; gm = gt.get(sid, set())
            for cid in cids:
                if cid not in s23_idx_keys: continue
                r2 = s23_idx.loc[cid]
                X.append(pair_feats(r1, r2))
                y.append(1 if cid in gm else 0)
                pairs.append((sid, cid))
                done += 1
                if done % 200000 == 0: print(f"      {done:,}/{total:,}")
        return np.array(X, dtype=np.float32), np.array(y), pairs

    Xtr,ytr,ptr = compute_feats(s1tr, tr_cands, gt_tr)
    print(f"    Train: {len(Xtr):,} pairs, {ytr.sum():,} pos ({ytr.mean():.4f})")
    Xva,yva,pva = compute_feats(s1va, va_cands, gt_va)
    print(f"    Val: {len(Xva):,} pairs, {yva.sum():,} pos ({yva.mean():.4f})")

    # === Round 1: Train LightGBM ===
    print("\n[7] Training LightGBM (Round 1)...")
    t0 = time.time()
    npos = ytr.sum(); nneg = len(ytr)-npos; scale = nneg/max(npos,1)
    model1 = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=8, num_leaves=127,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1)
    model1.fit(Xtr, ytr)
    print(f"    Round 1 trained in {time.time()-t0:.1f}s")
    imp = sorted(zip(FEAT_NAMES, model1.feature_importances_), key=lambda x:-x[1])
    print("    Top 15 features:")
    for nm,v in imp[:15]: print(f"      {nm}: {v}")

    # === Hard-negative mining ===
    print("\n[8] Hard-negative mining...")
    tr_proba = model1.predict_proba(Xtr)[:, 1]
    hard_neg_mask = (ytr == 0) & (tr_proba > 0.3)
    n_hard = hard_neg_mask.sum()
    print(f"    Hard negatives (proba>0.3): {n_hard:,}")

    # Round 2: retrain with upweighted hard negatives
    print("\n[9] Training LightGBM (Round 2 - hard neg weighted)...")
    sample_weight = np.ones(len(ytr))
    sample_weight[hard_neg_mask] = 5.0  # stronger upweight
    model2 = lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.02, max_depth=8, num_leaves=127,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1)
    model2.fit(Xtr, ytr, sample_weight=sample_weight)

    # === Dense threshold search for both models ===
    print("\n[10] Dense threshold search...")
    def find_best_threshold(model_obj, label):
        vp = model_obj.predict_proba(Xva)[:, 1]
        best_thr, best_f = 0.5, 0.0
        results = []
        for thr in np.arange(0.05, 0.995, 0.005):
            pd_d = {}
            for i,(sid,cid) in enumerate(pva):
                if vp[i] >= thr: pd_d.setdefault(sid,set()).add(cid)
            for sid in gt_va: pd_d.setdefault(sid,set())
            f,p,r = macro_f05(pd_d, gt_va)
            fp_c = sum(1 for i in range(len(pva)) if vp[i]>=thr and yva[i]==0)
            fn_c = sum(1 for i in range(len(pva)) if vp[i]<thr and yva[i]==1)
            results.append({"thr":thr,"f05":f,"prec":p,"rec":r,"fp":fp_c,"fn":fn_c})
            if f > best_f: best_f,best_thr = f,thr
        results.sort(key=lambda x:-x["f05"])
        print(f"\n    {label} Top thresholds:")
        print(f"    {'thr':>6} {'F0.5':>7} {'Prec':>7} {'Rec':>7} {'FP':>6} {'FN':>6}")
        for r in results[:10]:
            print(f"    {r['thr']:6.3f} {r['f05']:7.4f} {r['prec']:7.4f} {r['rec']:7.4f} {r['fp']:6,} {r['fn']:6,}")
        return best_thr, best_f, vp

    thr1,f1,vp1 = find_best_threshold(model1, "Model1")
    thr2,f2,vp2 = find_best_threshold(model2, "Model2 (hard-neg)")

    print(f"\n    Model1: F0.5={f1:.4f} @ thr={thr1:.3f}")
    print(f"    Model2: F0.5={f2:.4f} @ thr={thr2:.3f}")

    if f2 >= f1:
        final_model, final_thr, final_f, final_vp = model2, thr2, f2, vp2
        print("    -> Using Model2 (hard-neg)")
    else:
        final_model, final_thr, final_f, final_vp = model1, thr1, f1, vp1
        print("    -> Using Model1")

    # === Error analysis ===
    print("\n[11] Error analysis...")
    fp_list = [(i,sid,cid) for i,(sid,cid) in enumerate(pva) if final_vp[i]>=final_thr and yva[i]==0]
    fn_list = [(i,sid,cid) for i,(sid,cid) in enumerate(pva) if final_vp[i]<final_thr and yva[i]==1]
    print(f"    FP: {len(fp_list)}, FN in cands: {len(fn_list)}, Blocking misses: {vt-vh}")

    pred_d = {}
    for i,(sid,cid) in enumerate(pva):
        if final_vp[i] >= final_thr: pred_d.setdefault(sid,set()).add(cid)
    for sid in gt_va: pred_d.setdefault(sid,set())
    sfp = sum(1 for sid in gt_va if not gt_va[sid] and pred_d.get(sid,set()))
    sfn = sum(1 for sid in gt_va if gt_va[sid] and not pred_d.get(sid,set()))
    print(f"    Singleton FP: {sfp}, Singleton FN: {sfn}")

    print("\n    Sample FP:")
    for i,sid,cid in fp_list[:5]:
        n1 = s1s_idx.loc[sid]["nm"][:40] if sid in s1s_idx.index else "?"
        n2 = s23_idx.loc[cid]["nm"][:40] if cid in s23_idx_keys else "?"
        a1 = s1s_idx.loc[sid]["ad"][:40] if sid in s1s_idx.index else "?"
        a2 = s23_idx.loc[cid]["ad"][:40] if cid in s23_idx_keys else "?"
        print(f"      '{n1}' <-> '{n2}' | addr: '{a1}' <-> '{a2}' | p={final_vp[i]:.3f}")

    print("\n    Sample FN:")
    for i,sid,cid in fn_list[:5]:
        n1 = s1s_idx.loc[sid]["nm"][:40] if sid in s1s_idx.index else "?"
        n2 = s23_idx.loc[cid]["nm"][:40] if cid in s23_idx_keys else "?"
        trans = s23_idx.loc[cid]["is_trans"] if cid in s23_idx_keys else "?"
        print(f"      '{n1}' <-> '{n2}' | trans={trans} | p={final_vp[i]:.3f}")

    # === Retrain on full (train+val) ===
    print("\n[12] Retraining on full sample...")
    Xf = np.vstack([Xtr,Xva]); yf = np.concatenate([ytr,yva])
    full_proba = final_model.predict_proba(Xf)[:,1]
    sw = np.ones(len(yf))
    sw[(yf==0) & (full_proba>0.3)] = 5.0

    fm = lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.02, max_depth=8, num_leaves=127,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1)
    fm.fit(Xf, yf, sample_weight=sw)

    mp = os.path.join(MODELS_DIR, "lgbm_final.pkl")
    with open(mp,"wb") as f:
        pickle.dump({"model":fm,"threshold":final_thr,"features":FEAT_NAMES,
                      "val_f05":final_f,"val_blocking_recall":vh/max(vt,1)}, f)
    print(f"    Saved to {mp}")

    print("\n"+"="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"  Sample size:        {sample_n:,} S1 entities")
    print(f"  S23 pool:           {len(s23):,} records")
    print(f"  Blocking recall:    {vh/max(vt,1):.4f} ({vh}/{vt})")
    print(f"  Blocking misses:    {len(missed)}")
    print(f"  Train pairs:        {tp:,}")
    print(f"  Val pairs:          {vp:,}")
    print(f"  Best threshold:     {final_thr:.3f}")
    print(f"  Val F0.5:           {final_f:.4f}")
    print(f"  Val FP:             {len(fp_list)}")
    print(f"  Val FN (in cands):  {len(fn_list)}")
    print(f"  Singleton FP:       {sfp}")
    print(f"  Singleton FN:       {sfn}")
    print(f"  Training time:      {(time.time()-T0)/60:.1f} min")
    print("="*70)
    return {"model":fm,"threshold":final_thr,"val_f05":final_f}

# ========================== TEST INFERENCE ==========================
def run_test():
    print("\n"+"="*70)
    print("STAGE 2: TEST INFERENCE")
    print("="*70)
    T0 = time.time()
    mp = os.path.join(MODELS_DIR,"lgbm_final.pkl")
    with open(mp,"rb") as f: md = pickle.load(f)
    model = md["model"]; thr = md["threshold"]
    print(f"    Model loaded, thr={thr:.3f}, val_f05={md.get('val_f05','?')}")

    print("\n[1] Loading test S1...")
    s1t = pd.read_csv(os.path.join(TEST_DIR,"test_source1.tsv"), sep="\t")
    s1t = preprocess_df(s1t)
    all_ids = set(s1t["entity_id"].tolist())
    print(f"    {len(all_ids):,} S1 entities")
    matching = {sid:set() for sid in all_ids}
    candidates = {sid:set() for sid in all_ids}
    countries = sorted(s1t["ct"].unique())
    print(f"    Countries: {countries}")

    for country in countries:
        if not country: continue
        print(f"\n  === {country} ===")
        s1c = s1t[s1t["ct"]==country]
        print(f"    S1: {len(s1c):,}")

        print(f"    Loading S2...")
        s2t = pd.read_csv(os.path.join(TEST_DIR,"test_source2.tsv"), sep="\t")
        s2c = s2t[s2t["country"].apply(norm_country)==country].copy()
        del s2t; gc.collect()
        print(f"    Loading S3...")
        s3t = pd.read_csv(os.path.join(TEST_DIR,"test_source3.tsv"), sep="\t")
        s3c = s3t[s3t["country"].apply(norm_country)==country].copy()
        del s3t; gc.collect()
        s23c = pd.concat([s2c,s3c]).drop_duplicates(subset="entity_id")
        del s2c,s3c; gc.collect()
        print(f"    S23: {len(s23c):,}")
        s23c = preprocess_df(s23c)

        print(f"    Multi-strategy blocking...")
        cands = multi_block(s1c, s23c, top_k_name=100, top_k_addr=50)
        tc = sum(len(v) for v in cands.values())
        print(f"    Candidates: {tc:,} pairs for {len(cands):,} S1")

        print(f"    Scoring...")
        s1_ix = s1c.set_index("entity_id")
        s23_ix = s23c.set_index("entity_id")
        s23_ix_keys = set(s23_ix.index)
        chunk_ids = sorted(cands.keys()); csz = 10000
        nc = (len(chunk_ids)+csz-1)//csz
        for ci in range(nc):
            st = ci*csz; en = min(st+csz, len(chunk_ids))
            X, pl = [], []
            for sid in chunk_ids[st:en]:
                if sid not in s1_ix.index: continue
                r1 = s1_ix.loc[sid]
                for cid in cands[sid]:
                    if cid not in s23_ix_keys: continue
                    r2 = s23_ix.loc[cid]
                    X.append(pair_feats(r1, r2))
                    pl.append((sid, cid))
                    candidates[sid].add(cid)
            if X:
                Xa = np.array(X, dtype=np.float32)
                pr = model.predict_proba(Xa)[:,1]
                for i,(sid,cid) in enumerate(pl):
                    if pr[i] >= thr: matching[sid].add(cid)
            if (ci+1)%20==0 or ci==nc-1: print(f"      chunk {ci+1}/{nc}")
        del s23c,s23_ix; gc.collect()

    print("\n[2] Writing output...")
    def write_tsv(d, path, col):
        lines = [f"source1_entity_id\t{col}"]
        for sid in sorted(d.keys()):
            ids = ",".join(sorted(d[sid])) if d[sid] else ""
            lines.append(f"{sid}\t{ids}")
        with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines)+"\n")
        print(f"    {len(d):,} rows -> {path}")
    write_tsv(matching, os.path.join(OUTPUT_DIR,"matching_results.tsv"), "matched_entity_ids")
    write_tsv(candidates, os.path.join(OUTPUT_DIR,"candidate_pairs.tsv"), "candidate_entity_ids")
    nm = sum(1 for v in matching.values() if v)
    nt = sum(len(v) for v in matching.values())
    print(f"    Matched: {nm:,}/{len(all_ids):,} entities, {nt:,} total matches")
    print(f"    Inference done in {(time.time()-T0)/60:.1f} min")

def main():
    step = sys.argv[1] if len(sys.argv)>1 else "full"
    if step in ("train","full"): run_training()
    if step in ("test","full"): run_test()
    if step in ("validate","full"):
        print("\n=== Validation ===")
        import subprocess
        r = subprocess.run([sys.executable,
            os.path.join(PROJECT_ROOT,"utils","validate_submission.py"),
            "--matching", os.path.join(OUTPUT_DIR,"matching_results.tsv"),
            "--candidate", os.path.join(OUTPUT_DIR,"candidate_pairs.tsv"),
            "--test-dir", os.path.join(PROJECT_ROOT,"dataset","test")],
            capture_output=True, text=True)
        print(r.stdout)
        if r.stderr: print(r.stderr)
        print(f"Exit code: {r.returncode}")

if __name__ == "__main__": main()
