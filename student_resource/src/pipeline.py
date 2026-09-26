"""
Business Entity Resolution - Memory-Efficient Pipeline
======================================================
Designed to run on 16GB RAM with 12M+ row datasets.

Strategy:
- Process each country separately (natural partition)
- Use sparse TF-IDF for blocking
- Batch feature computation
- Train on subsample, predict on full test
"""
import os
import sys
import time
import re
import gc
import unicodedata
import pickle
import warnings
import functools
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional

# Force unbuffered output for real-time progress
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack as sparse_vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR

TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# Text Normalization
# ============================================================================
LEGAL_SUFFIXES = [
    "private limited", "pvt limited", "pvt ltd", "pvt. ltd.", "pvt. ltd",
    "pvt.ltd.", "pvt.ltd", "p ltd", "limited", "ltd", "corporation", "corp",
    "incorporated", "inc", "company", "co", "llc", "l.l.c.", "l.l.c",
    "llp", "l.l.p.", "l.l.p", "plc", "p.l.c.", "gmbh",
    "sarl", "s.a.r.l.", "s.a.r.l", "sas", "s.a.s.", "s.a.s",
    "sa", "s.a.", "s.a", "ag", "a.g.", "nv", "n.v.", "bv", "b.v.",
    "pty ltd", "pty. ltd.", "pty", "pty.",
    "societe anonyme", "societe a responsabilite limitee",
    "eurl", "sasu", "sci", "snc", "scs", "sca", "se",
    "groupe", "et cie", "cie",
]

COUNTRY_MAP = {
    "us": "us", "usa": "us", "u.s.": "us", "u.s.a.": "us",
    "united states": "us", "united states of america": "us",
    "india": "india", "in": "india", "ind": "india",
    "france": "france", "fr": "france", "fra": "france",
}


def norm_unicode(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    asc = text.encode("ascii", "ignore").decode("ascii")
    return asc if len(asc) >= len(text) * 0.5 else text


def norm_name(name) -> str:
    """Basic name normalization: lowercase, ascii, remove punct, collapse spaces."""
    if pd.isna(name) or str(name).strip() == "":
        return ""
    s = str(name).lower().strip()
    s = norm_unicode(s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_name_core(name) -> str:
    """Name without legal suffixes (for blocking)."""
    s = norm_name(name)
    if not s:
        return ""
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        clean = re.sub(r"[^a-z0-9\s]", " ", suf.lower()).strip()
        clean = re.sub(r"\s+", " ", clean)
        s = re.sub(r"\b" + re.escape(clean) + r"\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_addr(addr) -> str:
    if pd.isna(addr) or str(addr).strip() == "":
        return ""
    s = str(addr).lower().strip()
    s = norm_unicode(s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_country(c) -> str:
    if pd.isna(c) or str(c).strip() == "":
        return ""
    s = str(c).lower().strip()
    s = re.sub(r"[^a-z\s]", "", s).strip()
    return COUNTRY_MAP.get(s, s)


def extract_nums(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns."""
    t0 = time.time()
    df = df.copy()
    df["nm"] = df["business_name"].apply(norm_name)
    df["nmc"] = df["business_name"].apply(norm_name_core)
    df["ad"] = df["business_address"].apply(norm_addr)
    df["ct"] = df["country"].apply(norm_country)
    df["ad_nums"] = df["ad"].apply(extract_nums)
    print(f"    Preprocessed {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


# ============================================================================
# F0.5 Evaluation
# ============================================================================
def f05_entity(pred: Set[str], actual: Set[str]) -> float:
    if not pred and not actual:
        return 1.0
    if not pred or not actual:
        return 0.0
    tp = len(pred & actual)
    p = tp / len(pred)
    r = tp / len(actual)
    if p + r == 0:
        return 0.0
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(pred_d: Dict[str, Set[str]], gt_d: Dict[str, Set[str]]):
    scores, precs, recs = [], [], []
    for sid in gt_d:
        pr = pred_d.get(sid, set())
        ac = gt_d[sid]
        scores.append(f05_entity(pr, ac))
        if not pr and not ac:
            precs.append(1.0); recs.append(1.0)
        elif not pr or not ac:
            precs.append(0.0); recs.append(0.0)
        else:
            tp = len(pr & ac)
            precs.append(tp / len(pr)); recs.append(tp / len(ac))
    return np.mean(scores), np.mean(precs), np.mean(recs)


# ============================================================================
# Feature Computation
# ============================================================================
FEAT_NAMES = [
    "nm_exact", "nmc_exact",
    "nm_ratio", "nm_partial", "nm_tsort", "nm_tset",
    "nmc_ratio", "nmc_tsort", "nmc_tset",
    "nm_jaccard", "nm_olap", "nm_olap_r1", "nm_olap_r2",
    "nm_lendiff", "nm_lenratio", "nm_tokdiff",
    "ad_exact", "ad_avail1", "ad_avail2", "ad_both",
    "ad_ratio", "ad_partial", "ad_tsort", "ad_tset",
    "ad_jaccard", "ad_olap", "ad_lendiff", "ad_lenratio",
    "ad_numjac", "ad_numolap",
    "ct_match", "is_s2", "is_s3",
]


def pair_feats(r1, r2) -> list:
    """Compute feature vector for a pair. Returns list of floats matching FEAT_NAMES."""
    n1, n2 = r1["nm"], r2["nm"]
    c1, c2 = r1["nmc"], r2["nmc"]
    a1, a2 = r1["ad"], r2["ad"]
    ct1, ct2 = r1["ct"], r2["ct"]

    f = []
    # name exact
    f.append(float(n1 == n2 and n1 != ""))
    f.append(float(c1 == c2 and c1 != ""))

    # name fuzzy
    if n1 and n2:
        f.append(fuzz.ratio(n1, n2) / 100.0)
        f.append(fuzz.partial_ratio(n1, n2) / 100.0)
        f.append(fuzz.token_sort_ratio(n1, n2) / 100.0)
        f.append(fuzz.token_set_ratio(n1, n2) / 100.0)
    else:
        f.extend([0.0, 0.0, 0.0, 0.0])

    # core name fuzzy
    if c1 and c2:
        f.append(fuzz.ratio(c1, c2) / 100.0)
        f.append(fuzz.token_sort_ratio(c1, c2) / 100.0)
        f.append(fuzz.token_set_ratio(c1, c2) / 100.0)
    else:
        f.extend([0.0, 0.0, 0.0])

    # name token overlap
    t1 = set(c1.split()) if c1 else set()
    t2 = set(c2.split()) if c2 else set()
    if t1 or t2:
        ol = t1 & t2
        un = t1 | t2
        f.append(len(ol) / len(un) if un else 0.0)
        f.append(float(len(ol)))
        f.append(len(ol) / len(t1) if t1 else 0.0)
        f.append(len(ol) / len(t2) if t2 else 0.0)
    else:
        f.extend([0.0, 0.0, 0.0, 0.0])

    # name length
    f.append(float(abs(len(n1) - len(n2))))
    f.append(min(len(n1), len(n2)) / max(len(n1), len(n2), 1))
    f.append(float(abs(len(t1) - len(t2))))

    # address
    f.append(float(a1 == a2 and a1 != ""))
    f.append(float(a1 != ""))
    f.append(float(a2 != ""))
    f.append(float(a1 != "" and a2 != ""))

    if a1 and a2:
        f.append(fuzz.ratio(a1, a2) / 100.0)
        f.append(fuzz.partial_ratio(a1, a2) / 100.0)
        f.append(fuzz.token_sort_ratio(a1, a2) / 100.0)
        f.append(fuzz.token_set_ratio(a1, a2) / 100.0)
        at1 = set(a1.split())
        at2 = set(a2.split())
        aol = at1 & at2
        aun = at1 | at2
        f.append(len(aol) / len(aun) if aun else 0.0)
        f.append(float(len(aol)))
        f.append(float(abs(len(a1) - len(a2))))
        f.append(min(len(a1), len(a2)) / max(len(a1), len(a2), 1))
        an1 = r1.get("ad_nums", set()) or set()
        an2 = r2.get("ad_nums", set()) or set()
        if an1 or an2:
            nol = an1 & an2
            nun = an1 | an2
            f.append(len(nol) / len(nun) if nun else 0.0)
            f.append(float(len(nol)))
        else:
            f.extend([0.0, 0.0])
    else:
        f.extend([0.0] * 10)

    # country
    f.append(float(ct1 == ct2 and ct1 != ""))

    # source
    eid2 = str(r2.get("entity_id", ""))
    f.append(float(eid2.startswith("S2-")))
    f.append(float(eid2.startswith("S3-")))

    return f


# ============================================================================
# TF-IDF Blocking for one country
# ============================================================================
def tfidf_block_country(s1_c, s23_c, top_k=30, batch_sz=5000, min_sim=0.15):
    """TF-IDF char-ngram blocking within a single country.
    Returns dict: s1_id -> set of candidate s23_ids.
    """
    s1_names = s1_c["nmc"].fillna("").tolist()
    s1_ids = s1_c["entity_id"].tolist()
    s23_names = s23_c["nmc"].fillna("").tolist()
    s23_ids = s23_c["entity_id"].tolist()

    if not s1_names or not s23_names:
        return {}

    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4),
                          max_features=200000, sublinear_tf=True, dtype=np.float32)
    vec.fit(s1_names + s23_names)
    s23_tfidf = vec.transform(s23_names)

    cands = {}
    n_batches = (len(s1_names) + batch_sz - 1) // batch_sz
    for bi in range(n_batches):
        st = bi * batch_sz
        en = min(st + batch_sz, len(s1_names))
        batch_tfidf = vec.transform(s1_names[st:en])
        sim = cosine_similarity(batch_tfidf, s23_tfidf)

        for i in range(en - st):
            scores = sim[i]
            if len(scores) <= top_k:
                idxs = np.arange(len(scores))
            else:
                idxs = np.argpartition(scores, -top_k)[-top_k:]
            c = set()
            for j in idxs:
                if scores[j] > min_sim:
                    c.add(s23_ids[j])
            if c:
                cands[s1_ids[st + i]] = c

        if (bi + 1) % 50 == 0 or bi == n_batches - 1:
            print(f"      batch {bi+1}/{n_batches}")

    del vec, s23_tfidf
    gc.collect()
    return cands


# ============================================================================
# Main Pipeline  
# ============================================================================
def parse_gt(gt_df):
    """Parse ground truth to dict: s1_id -> set of matched ids. (Fast vectorized version)"""
    result = {}
    sids = gt_df["source1_entity_id"].values
    mids = gt_df["matched_entity_ids"].values
    for i in range(len(sids)):
        m = mids[i]
        if pd.isna(m) or str(m).strip() == "":
            result[sids[i]] = set()
        else:
            result[sids[i]] = set(str(m).split(","))
    return result


def run_training():
    """Train model on a subsample of training data."""
    print("=" * 70)
    print("STAGE 1: TRAINING")
    print("=" * 70)
    t_start = time.time()

    # --- Load ground truth (small) ---
    print("\n[1] Loading ground truth...")
    gt_df = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")
    gt_all = parse_gt(gt_df)
    del gt_df; gc.collect()
    print(f"    {len(gt_all):,} S1 entities in ground truth")

    # --- Sample S1 entities for training ---
    all_s1 = sorted(gt_all.keys())
    sample_n = min(30000, len(all_s1))
    sampled_ids = list(np.random.choice(all_s1, size=sample_n, replace=False))
    np.random.shuffle(sampled_ids)
    val_n = int(sample_n * 0.15)
    val_ids = set(sampled_ids[:val_n])
    train_ids = set(sampled_ids[val_n:])
    print(f"    Sampled {sample_n:,}: train={len(train_ids):,}, val={len(val_ids):,}")

    gt_train = {k: v for k, v in gt_all.items() if k in train_ids}
    gt_val = {k: v for k, v in gt_all.items() if k in val_ids}

    # Collect all S2/S3 IDs we need
    needed_s23 = set()
    for matches in gt_train.values():
        needed_s23 |= matches
    for matches in gt_val.values():
        needed_s23 |= matches

    # --- Load S1 (only sampled) ---
    print("\n[2] Loading S1 training data...")
    s1_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t")
    s1_samp = s1_all[s1_all["entity_id"].isin(set(sampled_ids))].copy()
    del s1_all; gc.collect()
    s1_samp = preprocess_df(s1_samp)

    # --- Load S2 + S3, keep positives + random negatives ---
    print("\n[3] Loading S2 training data...")
    s2_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t")
    s2_pos = s2_all[s2_all["entity_id"].isin(needed_s23)]
    # Random negative sample (small to save memory)
    neg_idx = np.random.choice(len(s2_all), size=min(100000, len(s2_all)), replace=False)
    s2_neg = s2_all.iloc[neg_idx]
    s2_samp = pd.concat([s2_pos, s2_neg]).drop_duplicates(subset="entity_id")
    del s2_all, s2_pos, s2_neg; gc.collect()
    print(f"    S2 sample: {len(s2_samp):,}")

    print("\n[4] Loading S3 training data...")
    s3_all = pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t")
    s3_pos = s3_all[s3_all["entity_id"].isin(needed_s23)]
    neg_idx = np.random.choice(len(s3_all), size=min(100000, len(s3_all)), replace=False)
    s3_neg = s3_all.iloc[neg_idx]
    s3_samp = pd.concat([s3_pos, s3_neg]).drop_duplicates(subset="entity_id")
    del s3_all, s3_pos, s3_neg; gc.collect()
    print(f"    S3 sample: {len(s3_samp):,}")

    # Combine S2+S3
    s23 = pd.concat([s2_samp, s3_samp]).drop_duplicates(subset="entity_id")
    del s2_samp, s3_samp; gc.collect()
    print(f"    Combined S23: {len(s23):,}")
    s23 = preprocess_df(s23)

    # --- Blocking per country ---
    print("\n[5] Generating candidates (TF-IDF blocking)...")
    s1_train_df = s1_samp[s1_samp["entity_id"].isin(train_ids)]
    s1_val_df = s1_samp[s1_samp["entity_id"].isin(val_ids)]

    train_cands = {}
    val_cands = {}
    for country in sorted(s1_samp["ct"].unique()):
        if not country:
            continue
        s1_tc = s1_train_df[s1_train_df["ct"] == country]
        s1_vc = s1_val_df[s1_val_df["ct"] == country]
        s23_c = s23[s23["ct"] == country]
        print(f"    Country '{country}': S1_train={len(s1_tc):,}, S1_val={len(s1_vc):,}, S23={len(s23_c):,}")

        if len(s23_c) == 0:
            continue

        if len(s1_tc) > 0:
            tc = tfidf_block_country(s1_tc, s23_c, top_k=30)
            train_cands.update(tc)
        if len(s1_vc) > 0:
            vc = tfidf_block_country(s1_vc, s23_c, top_k=30)
            val_cands.update(vc)

    # Inject positive pairs into training candidates (to learn from them)
    for sid, matches in gt_train.items():
        if sid not in train_cands:
            train_cands[sid] = set()
        for mid in matches:
            if mid in s23["entity_id"].values:
                train_cands[sid].add(mid)

    # Candidate recall
    def cand_recall(cands, gt):
        hit = total = 0
        for sid, ms in gt.items():
            for m in ms:
                total += 1
                if m in cands.get(sid, set()):
                    hit += 1
        return hit, total

    th, tt = cand_recall(train_cands, gt_train)
    print(f"\n    Train candidate recall: {th}/{tt} = {th/max(tt,1):.4f}")
    vh, vt = cand_recall(val_cands, gt_val)
    print(f"    Val candidate recall: {vh}/{vt} = {vh/max(vt,1):.4f}")

    total_train_cands = sum(len(v) for v in train_cands.values())
    total_val_cands = sum(len(v) for v in val_cands.values())
    print(f"    Train candidate pairs: {total_train_cands:,}")
    print(f"    Val candidate pairs: {total_val_cands:,}")

    # --- Compute features ---
    print("\n[6] Computing features...")
    s23_idx = s23.set_index("entity_id")

    def compute_feats(s1_df, cands, gt):
        """Return X, y, pair_ids arrays."""
        s1_idx = s1_df.set_index("entity_id")
        X_rows = []
        y_rows = []
        pairs = []
        done = 0
        total = sum(len(v) for v in cands.items())
        for sid, c_ids in cands.items():
            if sid not in s1_idx.index:
                continue
            r1 = s1_idx.loc[sid]
            gt_m = gt.get(sid, set())
            for cid in c_ids:
                if cid not in s23_idx.index:
                    continue
                r2 = s23_idx.loc[cid]
                X_rows.append(pair_feats(r1, r2))
                y_rows.append(1 if cid in gt_m else 0)
                pairs.append((sid, cid))
                done += 1
                if done % 50000 == 0:
                    print(f"      {done:,}/{total:,}")
        return np.array(X_rows, dtype=np.float32), np.array(y_rows), pairs

    X_train, y_train, pairs_train = compute_feats(s1_train_df, train_cands, gt_train)
    print(f"    Train: {len(X_train):,} pairs, {y_train.sum():,} positive ({y_train.mean():.4f})")

    X_val, y_val, pairs_val = compute_feats(s1_val_df, val_cands, gt_val)
    print(f"    Val: {len(X_val):,} pairs, {y_val.sum():,} positive ({y_val.mean():.4f})")

    # --- Train LightGBM ---
    print("\n[7] Training LightGBM...")
    t0 = time.time()
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)
    print(f"    Trained in {time.time()-t0:.1f}s")

    imp = sorted(zip(FEAT_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("    Top features:")
    for name, v in imp[:10]:
        print(f"      {name}: {v}")

    # --- Threshold tuning ---
    print("\n[8] Threshold tuning on validation...")
    val_proba = model.predict_proba(X_val)[:, 1]

    best_thr, best_f05 = 0.5, 0.0
    for thr in np.arange(0.10, 0.95, 0.05):
        pred_d = {}
        for i, (sid, cid) in enumerate(pairs_val):
            if val_proba[i] >= thr:
                pred_d.setdefault(sid, set()).add(cid)
        for sid in gt_val:
            pred_d.setdefault(sid, set())
        f, p, r = macro_f05(pred_d, gt_val)
        n_pred = sum(len(v) for v in pred_d.values())
        print(f"      thr={thr:.2f}: F0.5={f:.4f} P={p:.4f} R={r:.4f} preds={n_pred:,}")
        if f > best_f05:
            best_f05, best_thr = f, thr

    print(f"\n    Best: threshold={best_thr:.2f}, F0.5={best_f05:.4f}")

    # --- Error analysis ---
    print("\n[9] Error analysis...")
    s1_samp_idx = s1_samp.set_index("entity_id")
    fp_count = fn_count = 0
    print("    False Positives (wrong merges):")
    for i, (sid, cid) in enumerate(pairs_val):
        if val_proba[i] >= best_thr and y_val[i] == 0:
            fp_count += 1
            if fp_count <= 5:
                n1 = s1_samp_idx.loc[sid]["nm"] if sid in s1_samp_idx.index else "?"
                n2 = s23_idx.loc[cid]["nm"] if cid in s23_idx.index else "?"
                print(f"      {n1[:40]} <-> {n2[:40]} p={val_proba[i]:.3f}")
    print(f"    Total FP: {fp_count}")

    print("    False Negatives (missed matches):")
    for i, (sid, cid) in enumerate(pairs_val):
        if val_proba[i] < best_thr and y_val[i] == 1:
            fn_count += 1
            if fn_count <= 5:
                n1 = s1_samp_idx.loc[sid]["nm"] if sid in s1_samp_idx.index else "?"
                n2 = s23_idx.loc[cid]["nm"] if cid in s23_idx.index else "?"
                print(f"      {n1[:40]} <-> {n2[:40]} p={val_proba[i]:.3f}")
    print(f"    Total FN in candidates: {fn_count}")
    missed = vt - vh
    print(f"    Missed by blocking: {missed}")

    # --- Retrain on all sampled data ---
    print("\n[10] Retraining on full sample...")
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    final_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg/max(n_pos,1), random_state=42, n_jobs=-1, verbose=-1,
    )
    final_model.fit(X_full, y_full)

    # Save
    model_data = {"model": final_model, "threshold": best_thr, "features": FEAT_NAMES,
                  "val_f05": best_f05}
    model_path = os.path.join(MODELS_DIR, "lgbm_final.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Model saved to {model_path}")

    print(f"\n    Training completed in {(time.time()-t_start)/60:.1f} minutes")
    return model_data


def run_test_inference():
    """Run inference on test data, processing per-country in batches."""
    print("\n" + "=" * 70)
    print("STAGE 2: TEST INFERENCE")
    print("=" * 70)
    t_start = time.time()

    # Load model
    model_path = os.path.join(MODELS_DIR, "lgbm_final.pkl")
    with open(model_path, "rb") as f:
        md = pickle.load(f)
    model = md["model"]
    threshold = md["threshold"]
    print(f"    Model loaded, threshold={threshold:.2f}")

    # Load test S1
    print("\n[1] Loading test S1...")
    s1_test = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t")
    s1_test = preprocess_df(s1_test)
    all_s1_ids = set(s1_test["entity_id"].tolist())
    print(f"    {len(all_s1_ids):,} test S1 entities")

    # Results dicts
    matching = {sid: set() for sid in all_s1_ids}
    candidates = {sid: set() for sid in all_s1_ids}

    # Process per country
    countries = sorted(s1_test["ct"].unique())
    print(f"    Countries: {countries}")

    for country in countries:
        if not country:
            continue
        print(f"\n  === Country: {country} ===")
        s1_c = s1_test[s1_test["ct"] == country]
        print(f"    S1 entities: {len(s1_c):,}")

        # Load S2 + S3 for this country
        print(f"    Loading S2 for {country}...")
        s2_test = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t")
        s2_c = s2_test[s2_test["country"].apply(norm_country) == country].copy()
        del s2_test; gc.collect()

        print(f"    Loading S3 for {country}...")
        s3_test = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t")
        s3_c = s3_test[s3_test["country"].apply(norm_country) == country].copy()
        del s3_test; gc.collect()

        s23_c = pd.concat([s2_c, s3_c]).drop_duplicates(subset="entity_id")
        del s2_c, s3_c; gc.collect()
        print(f"    S23 for {country}: {len(s23_c):,}")

        s23_c = preprocess_df(s23_c)

        # TF-IDF blocking
        print(f"    Blocking...")
        cands = tfidf_block_country(s1_c, s23_c, top_k=30, batch_sz=3000)
        total_cands = sum(len(v) for v in cands.values())
        print(f"    Candidates: {total_cands:,} pairs for {len(cands):,} S1 entities")

        # Feature computation + prediction in sub-batches
        print(f"    Computing features & predicting...")
        s1_idx = s1_c.set_index("entity_id")
        s23_idx = s23_c.set_index("entity_id")

        chunk_ids = sorted(cands.keys())
        chunk_sz = 10000
        n_chunks = (len(chunk_ids) + chunk_sz - 1) // chunk_sz

        for ci in range(n_chunks):
            st = ci * chunk_sz
            en = min(st + chunk_sz, len(chunk_ids))
            X_rows = []
            pair_list = []

            for sid in chunk_ids[st:en]:
                if sid not in s1_idx.index:
                    continue
                r1 = s1_idx.loc[sid]
                for cid in cands[sid]:
                    if cid not in s23_idx.index:
                        continue
                    r2 = s23_idx.loc[cid]
                    X_rows.append(pair_feats(r1, r2))
                    pair_list.append((sid, cid))
                    candidates[sid].add(cid)

            if X_rows:
                X = np.array(X_rows, dtype=np.float32)
                probas = model.predict_proba(X)[:, 1]
                for i, (sid, cid) in enumerate(pair_list):
                    if probas[i] >= threshold:
                        matching[sid].add(cid)

            if (ci + 1) % 20 == 0 or ci == n_chunks - 1:
                print(f"      chunk {ci+1}/{n_chunks}")

        del s23_c, s23_idx
        gc.collect()

    # Write output
    print("\n[2] Writing output files...")
    def write_tsv(d, path, col_name):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = [f"source1_entity_id\t{col_name}"]
        for sid in sorted(d.keys()):
            ids_str = ",".join(sorted(d[sid])) if d[sid] else ""
            lines.append(f"{sid}\t{ids_str}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"    Written {len(d):,} rows to {path}")

    m_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    c_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    write_tsv(matching, m_path, "matched_entity_ids")
    write_tsv(candidates, c_path, "candidate_entity_ids")

    n_matched = sum(1 for v in matching.values() if v)
    n_total = sum(len(v) for v in matching.values())
    print(f"\n    S1 with matches: {n_matched:,}/{len(all_s1_ids):,}")
    print(f"    Total matches: {n_total:,}")
    print(f"    Test inference completed in {(time.time()-t_start)/60:.1f} minutes")


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "full"

    if step in ("train", "full"):
        run_training()

    if step in ("test", "full"):
        run_test_inference()

    if step in ("validate", "full"):
        print("\n=== Validation ===")
        import subprocess
        val_script = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
        m_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
        c_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
        test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
        result = subprocess.run(
            [sys.executable, val_script, "--matching", m_path, "--candidate", c_path, "--test-dir", test_dir],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"Validator exit code: {result.returncode}")


if __name__ == "__main__":
    main()
