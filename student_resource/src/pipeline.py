"""
Business Entity Resolution - Memory-Efficient Pipeline v3
=========================================================
Optimized for 16GB RAM with 12M+ row datasets.
All lookups use sets/dicts, no O(n) scans in loops.
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
LEGAL_SUFFIXES = [
    "private limited","pvt limited","pvt ltd","pvt. ltd.","pvt. ltd",
    "pvt.ltd.","pvt.ltd","p ltd","limited","ltd","corporation","corp",
    "incorporated","inc","company","co","llc","l.l.c.","l.l.c",
    "llp","l.l.p.","l.l.p","plc","p.l.c.","gmbh",
    "sarl","s.a.r.l.","s.a.r.l","sas","s.a.s.","s.a.s",
    "sa","s.a.","s.a","ag","a.g.","nv","n.v.","bv","b.v.",
    "pty ltd","pty. ltd.","pty","pty.",
    "societe anonyme","societe a responsabilite limitee",
    "eurl","sasu","sci","snc","scs","sca","se","groupe","et cie","cie",
]
COUNTRY_MAP = {
    "us":"us","usa":"us","u.s.":"us","u.s.a.":"us",
    "united states":"us","united states of america":"us",
    "india":"india","in":"india","ind":"india",
    "france":"france","fr":"france","fra":"france",
}

def _nu(t):
    if not t: return ""
    t = unicodedata.normalize("NFKD", t)
    a = t.encode("ascii","ignore").decode("ascii")
    return a if len(a) >= len(t)*0.5 else t

def norm_name(name) -> str:
    if pd.isna(name) or str(name).strip()=="": return ""
    s = str(name).lower().strip()
    s = _nu(s)
    s = s.replace("&"," and ").replace("+"," and ")
    s = re.sub(r"[^a-z0-9\s]"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def norm_name_core(name) -> str:
    s = norm_name(name)
    if not s: return ""
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        c = re.sub(r"[^a-z0-9\s]"," ",suf.lower()).strip()
        c = re.sub(r"\s+"," ",c)
        s = re.sub(r"\b"+re.escape(c)+r"\b","",s)
    return re.sub(r"\s+"," ",s).strip()

def norm_addr(addr) -> str:
    if pd.isna(addr) or str(addr).strip()=="": return ""
    s = str(addr).lower().strip()
    s = _nu(s)
    s = s.replace("&"," and ")
    s = re.sub(r"[^a-z0-9\s\-]"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def norm_country(c) -> str:
    if pd.isna(c) or str(c).strip()=="": return ""
    s = re.sub(r"[^a-z\s]","",str(c).lower().strip()).strip()
    return COUNTRY_MAP.get(s,s)

def extract_nums(t):
    return set(re.findall(r"\b\d+\b",t)) if t else set()

def preprocess_df(df):
    t0=time.time()
    df=df.copy()
    df["nm"]=df["business_name"].apply(norm_name)
    df["nmc"]=df["business_name"].apply(norm_name_core)
    df["ad"]=df["business_address"].apply(norm_addr)
    df["ct"]=df["country"].apply(norm_country)
    df["ad_nums"]=df["ad"].apply(extract_nums)
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
    "nm_exact","nmc_exact",
    "nm_ratio","nm_partial","nm_tsort","nm_tset",
    "nmc_ratio","nmc_tsort","nmc_tset",
    "nm_jaccard","nm_olap","nm_olap_r1","nm_olap_r2",
    "nm_lendiff","nm_lenratio","nm_tokdiff",
    "ad_exact","ad_avail1","ad_avail2","ad_both",
    "ad_ratio","ad_partial","ad_tsort","ad_tset",
    "ad_jaccard","ad_olap","ad_lendiff","ad_lenratio",
    "ad_numjac","ad_numolap",
    "ct_match","is_s2","is_s3",
]

def pair_feats(r1,r2):
    n1,n2=r1["nm"],r2["nm"]
    c1,c2=r1["nmc"],r2["nmc"]
    a1,a2=r1["ad"],r2["ad"]
    ct1,ct2=r1["ct"],r2["ct"]
    f=[]
    f.append(float(n1==n2 and n1!=""))
    f.append(float(c1==c2 and c1!=""))
    if n1 and n2:
        f.append(fuzz.ratio(n1,n2)/100.0)
        f.append(fuzz.partial_ratio(n1,n2)/100.0)
        f.append(fuzz.token_sort_ratio(n1,n2)/100.0)
        f.append(fuzz.token_set_ratio(n1,n2)/100.0)
    else: f.extend([0.0]*4)
    if c1 and c2:
        f.append(fuzz.ratio(c1,c2)/100.0)
        f.append(fuzz.token_sort_ratio(c1,c2)/100.0)
        f.append(fuzz.token_set_ratio(c1,c2)/100.0)
    else: f.extend([0.0]*3)
    t1=set(c1.split()) if c1 else set()
    t2=set(c2.split()) if c2 else set()
    if t1 or t2:
        ol=t1&t2; un=t1|t2
        f.append(len(ol)/len(un) if un else 0.0)
        f.append(float(len(ol)))
        f.append(len(ol)/len(t1) if t1 else 0.0)
        f.append(len(ol)/len(t2) if t2 else 0.0)
    else: f.extend([0.0]*4)
    f.append(float(abs(len(n1)-len(n2))))
    f.append(min(len(n1),len(n2))/max(len(n1),len(n2),1))
    f.append(float(abs(len(t1)-len(t2))))
    f.append(float(a1==a2 and a1!=""))
    f.append(float(a1!=""))
    f.append(float(a2!=""))
    f.append(float(a1!="" and a2!=""))
    if a1 and a2:
        f.append(fuzz.ratio(a1,a2)/100.0)
        f.append(fuzz.partial_ratio(a1,a2)/100.0)
        f.append(fuzz.token_sort_ratio(a1,a2)/100.0)
        f.append(fuzz.token_set_ratio(a1,a2)/100.0)
        at1=set(a1.split()); at2=set(a2.split())
        aol=at1&at2; aun=at1|at2
        f.append(len(aol)/len(aun) if aun else 0.0)
        f.append(float(len(aol)))
        f.append(float(abs(len(a1)-len(a2))))
        f.append(min(len(a1),len(a2))/max(len(a1),len(a2),1))
        an1=r1.get("ad_nums",set()) or set()
        an2=r2.get("ad_nums",set()) or set()
        if an1 or an2:
            nol=an1&an2; nun=an1|an2
            f.append(len(nol)/len(nun) if nun else 0.0)
            f.append(float(len(nol)))
        else: f.extend([0.0,0.0])
    else: f.extend([0.0]*10)
    f.append(float(ct1==ct2 and ct1!=""))
    eid2=str(r2.get("entity_id",""))
    f.append(float(eid2.startswith("S2-")))
    f.append(float(eid2.startswith("S3-")))
    return f

# ========================== TF-IDF Blocking ==========================
def tfidf_block(s1_c, s23_c, top_k=30, batch_sz=5000, min_sim=0.15):
    s1_names=s1_c["nmc"].fillna("").tolist()
    s1_ids=s1_c["entity_id"].tolist()
    s23_names=s23_c["nmc"].fillna("").tolist()
    s23_ids=s23_c["entity_id"].tolist()
    if not s1_names or not s23_names: return {}
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,4),
                        max_features=200000,sublinear_tf=True,dtype=np.float32)
    vec.fit(s1_names+s23_names)
    s23_tf=vec.transform(s23_names)
    cands={}
    nb=(len(s1_names)+batch_sz-1)//batch_sz
    for bi in range(nb):
        st=bi*batch_sz; en=min(st+batch_sz,len(s1_names))
        btf=vec.transform(s1_names[st:en])
        sim=cosine_similarity(btf,s23_tf)
        for i in range(en-st):
            sc=sim[i]
            if len(sc)<=top_k: idxs=np.arange(len(sc))
            else: idxs=np.argpartition(sc,-top_k)[-top_k:]
            c=set()
            for j in idxs:
                if sc[j]>min_sim: c.add(s23_ids[j])
            if c: cands[s1_ids[st+i]]=c
        if (bi+1)%50==0 or bi==nb-1:
            print(f"      batch {bi+1}/{nb}")
    del vec,s23_tf; gc.collect()
    return cands

# ========================== GT Parser ==========================
def parse_gt(gt_df):
    result={}
    sids=gt_df["source1_entity_id"].values
    mids=gt_df["matched_entity_ids"].values
    for i in range(len(sids)):
        m=mids[i]
        if pd.isna(m) or str(m).strip()=="": result[sids[i]]=set()
        else: result[sids[i]]=set(str(m).split(","))
    return result

# ========================== TRAINING ==========================
def run_training():
    print("="*70)
    print("STAGE 1: TRAINING")
    print("="*70)
    T0=time.time()

    print("\n[1] Loading ground truth...")
    gt_df=pd.read_csv(os.path.join(TRAIN_DIR,"train_ground_truth.tsv"),sep="\t")
    gt_all=parse_gt(gt_df); del gt_df; gc.collect()
    print(f"    {len(gt_all):,} S1 entities")

    all_s1=sorted(gt_all.keys())
    sample_n=min(30000,len(all_s1))
    sampled=list(np.random.choice(all_s1,size=sample_n,replace=False))
    np.random.shuffle(sampled)
    val_n=int(sample_n*0.15)
    val_ids=set(sampled[:val_n]); train_ids=set(sampled[val_n:])
    print(f"    Sampled {sample_n:,}: train={len(train_ids):,}, val={len(val_ids):,}")

    gt_tr={k:v for k,v in gt_all.items() if k in train_ids}
    gt_va={k:v for k,v in gt_all.items() if k in val_ids}
    needed_s23=set()
    for ms in gt_tr.values(): needed_s23|=ms
    for ms in gt_va.values(): needed_s23|=ms

    print("\n[2] Loading S1...")
    s1a=pd.read_csv(os.path.join(TRAIN_DIR,"train_source1.tsv"),sep="\t")
    s1s=s1a[s1a["entity_id"].isin(set(sampled))].copy(); del s1a; gc.collect()
    s1s=preprocess_df(s1s)

    print("\n[3] Loading S2...")
    s2a=pd.read_csv(os.path.join(TRAIN_DIR,"train_source2.tsv"),sep="\t")
    s2p=s2a[s2a["entity_id"].isin(needed_s23)]
    ni=np.random.choice(len(s2a),size=min(100000,len(s2a)),replace=False)
    s2s=pd.concat([s2p,s2a.iloc[ni]]).drop_duplicates(subset="entity_id")
    del s2a,s2p; gc.collect()
    print(f"    S2 sample: {len(s2s):,}")

    print("\n[4] Loading S3...")
    s3a=pd.read_csv(os.path.join(TRAIN_DIR,"train_source3.tsv"),sep="\t")
    s3p=s3a[s3a["entity_id"].isin(needed_s23)]
    ni=np.random.choice(len(s3a),size=min(100000,len(s3a)),replace=False)
    s3s=pd.concat([s3p,s3a.iloc[ni]]).drop_duplicates(subset="entity_id")
    del s3a,s3p; gc.collect()
    print(f"    S3 sample: {len(s3s):,}")

    s23=pd.concat([s2s,s3s]).drop_duplicates(subset="entity_id")
    del s2s,s3s; gc.collect()
    print(f"    S23 combined: {len(s23):,}")
    s23=preprocess_df(s23)

    # Build set of available S23 IDs for fast lookup
    s23_id_set=set(s23["entity_id"].values)
    print(f"    S23 ID set: {len(s23_id_set):,}")

    print("\n[5] Blocking...")
    s1tr=s1s[s1s["entity_id"].isin(train_ids)]
    s1va=s1s[s1s["entity_id"].isin(val_ids)]

    tr_cands={}; va_cands={}
    for country in sorted(s1s["ct"].unique()):
        if not country: continue
        s1tc=s1tr[s1tr["ct"]==country]
        s1vc=s1va[s1va["ct"]==country]
        s23c=s23[s23["ct"]==country]
        print(f"    {country}: S1_tr={len(s1tc):,} S1_va={len(s1vc):,} S23={len(s23c):,}")
        if len(s23c)==0: continue
        if len(s1tc)>0: tr_cands.update(tfidf_block(s1tc,s23c,top_k=30))
        if len(s1vc)>0: va_cands.update(tfidf_block(s1vc,s23c,top_k=30))

    # Inject positive pairs into training (use set for O(1) lookup)
    for sid,ms in gt_tr.items():
        if sid not in tr_cands: tr_cands[sid]=set()
        for m in ms:
            if m in s23_id_set: tr_cands[sid].add(m)

    # Candidate recall
    def cr(cands,gt):
        h=t=0
        for sid,ms in gt.items():
            for m in ms:
                t+=1
                if m in cands.get(sid,set()): h+=1
        return h,t
    th,tt=cr(tr_cands,gt_tr); print(f"    Train cand recall: {th}/{tt}={th/max(tt,1):.4f}")
    vh,vt=cr(va_cands,gt_va); print(f"    Val cand recall: {vh}/{vt}={vh/max(vt,1):.4f}")
    print(f"    Train pairs: {sum(len(v) for v in tr_cands.values()):,}")
    print(f"    Val pairs: {sum(len(v) for v in va_cands.values()):,}")

    print("\n[6] Computing features...")
    s23_idx=s23.set_index("entity_id")
    s23_idx_set=set(s23_idx.index)

    def compute_feats(s1_df,cands,gt):
        s1_idx=s1_df.set_index("entity_id")
        X,y,pairs=[],[],[]
        done=0; total=sum(len(v) for v in cands.items())
        for sid,cids in cands.items():
            if sid not in s1_idx.index: continue
            r1=s1_idx.loc[sid]; gm=gt.get(sid,set())
            for cid in cids:
                if cid not in s23_idx_set: continue
                r2=s23_idx.loc[cid]
                X.append(pair_feats(r1,r2))
                y.append(1 if cid in gm else 0)
                pairs.append((sid,cid))
                done+=1
                if done%50000==0: print(f"      {done:,}/{total:,}")
        return np.array(X,dtype=np.float32),np.array(y),pairs

    Xtr,ytr,ptr=compute_feats(s1tr,tr_cands,gt_tr)
    print(f"    Train: {len(Xtr):,} pairs, {ytr.sum():,} pos ({ytr.mean():.4f})")
    Xva,yva,pva=compute_feats(s1va,va_cands,gt_va)
    print(f"    Val: {len(Xva):,} pairs, {yva.sum():,} pos ({yva.mean():.4f})")

    print("\n[7] Training LightGBM...")
    t0=time.time()
    npos=ytr.sum(); nneg=len(ytr)-npos; scale=nneg/max(npos,1)
    model=lgb.LGBMClassifier(
        n_estimators=500,learning_rate=0.05,max_depth=7,num_leaves=63,
        min_child_samples=50,subsample=0.8,colsample_bytree=0.8,
        scale_pos_weight=scale,random_state=42,n_jobs=-1,verbose=-1)
    model.fit(Xtr,ytr)
    print(f"    Trained in {time.time()-t0:.1f}s")
    imp=sorted(zip(FEAT_NAMES,model.feature_importances_),key=lambda x:-x[1])
    print("    Top features:")
    for nm,v in imp[:10]: print(f"      {nm}: {v}")

    print("\n[8] Threshold tuning...")
    vp=model.predict_proba(Xva)[:,1]
    best_thr,best_f=0.5,0.0
    for thr in np.arange(0.10,0.95,0.05):
        pd_d={}
        for i,(sid,cid) in enumerate(pva):
            if vp[i]>=thr: pd_d.setdefault(sid,set()).add(cid)
        for sid in gt_va: pd_d.setdefault(sid,set())
        f,p,r=macro_f05(pd_d,gt_va)
        n=sum(len(v) for v in pd_d.values())
        print(f"      thr={thr:.2f}: F0.5={f:.4f} P={p:.4f} R={r:.4f} n={n:,}")
        if f>best_f: best_f,best_thr=f,thr
    print(f"    Best: thr={best_thr:.2f} F0.5={best_f:.4f}")

    print("\n[9] Error analysis (sample)...")
    fp=[(i,sid,cid) for i,(sid,cid) in enumerate(pva) if vp[i]>=best_thr and yva[i]==0]
    fn=[(i,sid,cid) for i,(sid,cid) in enumerate(pva) if vp[i]<best_thr and yva[i]==1]
    s1s_idx=s1s.set_index("entity_id")
    print(f"    FP: {len(fp)}, FN in cands: {len(fn)}, missed by blocking: {vt-vh}")
    for i,sid,cid in fp[:3]:
        n1=s1s_idx.loc[sid]["nm"][:35] if sid in s1s_idx.index else "?"
        n2=s23_idx.loc[cid]["nm"][:35] if cid in s23_idx_set else "?"
        print(f"      FP: '{n1}' <-> '{n2}' p={vp[i]:.3f}")
    for i,sid,cid in fn[:3]:
        n1=s1s_idx.loc[sid]["nm"][:35] if sid in s1s_idx.index else "?"
        n2=s23_idx.loc[cid]["nm"][:35] if cid in s23_idx_set else "?"
        print(f"      FN: '{n1}' <-> '{n2}' p={vp[i]:.3f}")

    print("\n[10] Retraining on full sample...")
    Xf=np.vstack([Xtr,Xva]); yf=np.concatenate([ytr,yva])
    fm=lgb.LGBMClassifier(
        n_estimators=500,learning_rate=0.05,max_depth=7,num_leaves=63,
        min_child_samples=50,subsample=0.8,colsample_bytree=0.8,
        scale_pos_weight=scale,random_state=42,n_jobs=-1,verbose=-1)
    fm.fit(Xf,yf)
    mp=os.path.join(MODELS_DIR,"lgbm_final.pkl")
    with open(mp,"wb") as f: pickle.dump({"model":fm,"threshold":best_thr,"features":FEAT_NAMES,"val_f05":best_f},f)
    print(f"    Saved to {mp}")
    print(f"    Training done in {(time.time()-T0)/60:.1f} min")
    return {"model":fm,"threshold":best_thr,"val_f05":best_f}

# ========================== TEST INFERENCE ==========================
def run_test():
    print("\n"+"="*70)
    print("STAGE 2: TEST INFERENCE")
    print("="*70)
    T0=time.time()

    mp=os.path.join(MODELS_DIR,"lgbm_final.pkl")
    with open(mp,"rb") as f: md=pickle.load(f)
    model=md["model"]; thr=md["threshold"]
    print(f"    Model loaded, thr={thr:.2f}")

    print("\n[1] Loading test S1...")
    s1t=pd.read_csv(os.path.join(TEST_DIR,"test_source1.tsv"),sep="\t")
    s1t=preprocess_df(s1t)
    all_ids=set(s1t["entity_id"].tolist())
    print(f"    {len(all_ids):,} S1 entities")

    matching={sid:set() for sid in all_ids}
    candidates={sid:set() for sid in all_ids}

    countries=sorted(s1t["ct"].unique())
    print(f"    Countries: {countries}")

    for country in countries:
        if not country: continue
        print(f"\n  === {country} ===")
        s1c=s1t[s1t["ct"]==country]
        print(f"    S1: {len(s1c):,}")

        # Load S2 for this country only
        print(f"    Loading S2...")
        s2t=pd.read_csv(os.path.join(TEST_DIR,"test_source2.tsv"),sep="\t")
        s2c=s2t[s2t["country"].apply(norm_country)==country].copy()
        del s2t; gc.collect()

        print(f"    Loading S3...")
        s3t=pd.read_csv(os.path.join(TEST_DIR,"test_source3.tsv"),sep="\t")
        s3c=s3t[s3t["country"].apply(norm_country)==country].copy()
        del s3t; gc.collect()

        s23c=pd.concat([s2c,s3c]).drop_duplicates(subset="entity_id")
        del s2c,s3c; gc.collect()
        print(f"    S23: {len(s23c):,}")
        s23c=preprocess_df(s23c)

        print(f"    Blocking...")
        cands=tfidf_block(s1c,s23c,top_k=30,batch_sz=3000)
        tc=sum(len(v) for v in cands.values())
        print(f"    Candidates: {tc:,} pairs for {len(cands):,} S1")

        print(f"    Scoring...")
        s1_idx=s1c.set_index("entity_id")
        s23_idx=s23c.set_index("entity_id")
        s23_idx_set=set(s23_idx.index)

        chunk_ids=sorted(cands.keys())
        csz=10000; nc=(len(chunk_ids)+csz-1)//csz
        for ci in range(nc):
            st=ci*csz; en=min(st+csz,len(chunk_ids))
            X,pl=[],[]
            for sid in chunk_ids[st:en]:
                if sid not in s1_idx.index: continue
                r1=s1_idx.loc[sid]
                for cid in cands[sid]:
                    if cid not in s23_idx_set: continue
                    r2=s23_idx.loc[cid]
                    X.append(pair_feats(r1,r2))
                    pl.append((sid,cid))
                    candidates[sid].add(cid)
            if X:
                Xa=np.array(X,dtype=np.float32)
                pr=model.predict_proba(Xa)[:,1]
                for i,(sid,cid) in enumerate(pl):
                    if pr[i]>=thr: matching[sid].add(cid)
            if (ci+1)%20==0 or ci==nc-1: print(f"      chunk {ci+1}/{nc}")

        del s23c,s23_idx; gc.collect()

    print("\n[2] Writing output...")
    def write_tsv(d,path,col):
        lines=[f"source1_entity_id\t{col}"]
        for sid in sorted(d.keys()):
            ids=",".join(sorted(d[sid])) if d[sid] else ""
            lines.append(f"{sid}\t{ids}")
        with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines)+"\n")
        print(f"    {len(d):,} rows -> {path}")

    write_tsv(matching,os.path.join(OUTPUT_DIR,"matching_results.tsv"),"matched_entity_ids")
    write_tsv(candidates,os.path.join(OUTPUT_DIR,"candidate_pairs.tsv"),"candidate_entity_ids")

    nm=sum(1 for v in matching.values() if v)
    nt=sum(len(v) for v in matching.values())
    print(f"    Matched: {nm:,}/{len(all_ids):,} entities, {nt:,} total matches")
    print(f"    Inference done in {(time.time()-T0)/60:.1f} min")

def main():
    step=sys.argv[1] if len(sys.argv)>1 else "full"
    if step in ("train","full"): run_training()
    if step in ("test","full"): run_test()
    if step in ("validate","full"):
        print("\n=== Validation ===")
        import subprocess
        r=subprocess.run([sys.executable,os.path.join(PROJECT_ROOT,"utils","validate_submission.py"),
            "--matching",os.path.join(OUTPUT_DIR,"matching_results.tsv"),
            "--candidate",os.path.join(OUTPUT_DIR,"candidate_pairs.tsv"),
            "--test-dir",os.path.join(PROJECT_ROOT,"dataset","test")],
            capture_output=True,text=True)
        print(r.stdout)
        if r.stderr: print(r.stderr)
        print(f"Exit code: {r.returncode}")

if __name__=="__main__": main()
