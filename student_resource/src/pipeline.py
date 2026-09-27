"""
ULTRA-FAST Pipeline for Windows — Generates submission in ~90 min
Uses inverted-index blocking for test (100x faster than TF-IDF on large data)
TF-IDF blocking only for training (small data)
"""
import os, sys, time, re, gc, unicodedata, pickle, warnings, functools
from collections import defaultdict

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
TEST_DIR  = os.path.join(PROJECT_ROOT, "dataset", "test")
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
    return unicodedata.normalize("NFKD",t).encode("ascii","ignore").decode("ascii")

def norm_name(name):
    if pd.isna(name) or str(name).strip()=="": return ""
    a=_to_ascii(str(name).lower().strip())
    a=a.replace("&"," and ").replace("+"," and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]"," ",a)).strip()

def norm_name_core(name):
    s=norm_name(name)
    if not s: return ""
    for suf in LEGAL_SUFFIXES:
        c=re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s]"," ",suf.lower())).strip()
        if c: s=re.sub(r"\b"+re.escape(c)+r"\b","",s)
    return re.sub(r"\s+"," ",s).strip()

def is_transliterated(name):
    if pd.isna(name) or str(name).strip()=="": return False
    s=str(name); return sum(1 for c in s if ord(c)<128)<len(s)*0.5

def norm_addr(addr):
    if pd.isna(addr) or str(addr).strip()=="": return ""
    a=_to_ascii(str(addr).lower().strip()).replace("&"," and ")
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9\s\-]"," ",a)).strip()

def norm_country(c):
    if pd.isna(c) or str(c).strip()=="": return ""
    s=re.sub(r"[^a-z\s]","",str(c).lower().strip()).strip()
    return COUNTRY_MAP.get(s,s)

def extract_nums(t):
    return set(re.findall(r"\b\d+\b",t)) if t else set()

def first_token(t):
    if not t: return ""
    for tok in t.split():
        if len(tok)>2: return tok
    return t.split()[0] if t.split() else ""

def preprocess_df(df):
    t0=time.time(); df=df.copy()
    df["nm"]=df["business_name"].apply(norm_name)
    df["nmc"]=df["business_name"].apply(norm_name_core)
    df["ad"]=df["business_address"].apply(norm_addr)
    df["ct"]=df["country"].apply(norm_country)
    df["ad_nums"]=df["ad"].apply(extract_nums)
    df["is_trans"]=df["business_name"].apply(is_transliterated)
    df["first_tok"]=df["nmc"].apply(first_token)
    print(f"    Preprocessed {len(df):,} rows in {time.time()-t0:.1f}s")
    return df

# ========================== Metrics ==========================
def f05_ent(p,a):
    if not p and not a: return 1.0
    if not p or not a: return 0.0
    tp=len(p&a); pr=tp/len(p); rc=tp/len(a)
    return (1.25*pr*rc)/(0.25*pr+rc) if pr+rc>0 else 0.0

def macro_f05(pd_d,gt_d):
    return np.mean([f05_ent(pd_d.get(s,set()),gt_d[s]) for s in gt_d])

# ========================== Features (41) ==========================
FEAT_NAMES = [
    "nm_exact","nmc_exact","nm_ratio","nm_partial","nm_tsort","nm_tset",
    "nmc_ratio","nmc_tsort","nmc_tset","nm_levenshtein",
    "nm_jaccard","nm_olap","nm_olap_frac","first_tok_match","nm_prefix4",
    "nm_lendiff","nm_lenratio","nm_tokdiff",
    "s23_is_trans","both_have_name",
    "ad_exact","ad_both","ad_ratio","ad_partial","ad_tsort","ad_tset",
    "ad_lendiff","ad_lenratio","ad_jaccard","ad_olap",
    "ad_numjac","ad_numolap","ad_numolap_frac",
    "name_best","addr_best","overall_best","name_weak_addr_strong","name_addr_product",
    "ct_match","is_s2","is_s3",
]

def pair_feats(r1,r2):
    n1,n2=r1["nm"],r2["nm"]; c1,c2=r1["nmc"],r2["nmc"]; a1,a2=r1["ad"],r2["ad"]
    f=[float(n1==n2 and n1!=""),float(c1==c2 and c1!="")]
    nm_rat=nm_part=nm_tsort=nm_tset=0.0
    if n1 and n2:
        nm_rat=fuzz.ratio(n1,n2)/100; nm_part=fuzz.partial_ratio(n1,n2)/100
        nm_tsort=fuzz.token_sort_ratio(n1,n2)/100; nm_tset=fuzz.token_set_ratio(n1,n2)/100
    f+=[nm_rat,nm_part,nm_tsort,nm_tset]
    nmc_rat=nmc_tsort=nmc_tset=0.0
    if c1 and c2:
        nmc_rat=fuzz.ratio(c1,c2)/100; nmc_tsort=fuzz.token_sort_ratio(c1,c2)/100
        nmc_tset=fuzz.token_set_ratio(c1,c2)/100
    f+=[nmc_rat,nmc_tsort,nmc_tset]
    f.append(1.0-Levenshtein.normalized_distance(c1,c2) if c1 and c2 else 0.0)
    t1=set(c1.split()) if c1 else set(); t2=set(c2.split()) if c2 else set()
    if t1 or t2:
        ol=t1&t2; un=t1|t2
        f+=[len(ol)/len(un) if un else 0,float(len(ol)),len(ol)/max(len(t1),len(t2),1)]
    else: f+=[0,0,0]
    ft1=r1.get("first_tok",""); ft2=r2.get("first_tok","")
    f.append(float(ft1==ft2 and ft1!=""))
    f.append(float(c1[:4]==c2[:4]) if c1 and c2 else 0.0)
    f+=[float(abs(len(n1)-len(n2))),min(len(n1),len(n2))/max(len(n1),len(n2),1),float(abs(len(t1)-len(t2)))]
    f+=[float(r2.get("is_trans",False)),float(n1!="" and n2!="")]
    f+=[float(a1==a2 and a1!=""),float(a1!="" and a2!="")]
    ad_rat=ad_part=ad_tsort=ad_tset=0.0
    if a1 and a2:
        ad_rat=fuzz.ratio(a1,a2)/100; ad_part=fuzz.partial_ratio(a1,a2)/100
        ad_tsort=fuzz.token_sort_ratio(a1,a2)/100; ad_tset=fuzz.token_set_ratio(a1,a2)/100
    f+=[ad_rat,ad_part,ad_tsort,ad_tset]
    f+=[float(abs(len(a1)-len(a2))),min(len(a1),len(a2))/max(len(a1),len(a2),1)] if a1 and a2 else [0,0]
    if a1 and a2:
        at1=set(a1.split()); at2=set(a2.split()); aol=at1&at2; aun=at1|at2
        f+=[len(aol)/len(aun) if aun else 0,float(len(aol))]
    else: f+=[0,0]
    an1=r1.get("ad_nums",set()) or set(); an2=r2.get("ad_nums",set()) or set()
    if an1 or an2:
        nol=an1&an2; nun=an1|an2
        f+=[len(nol)/len(nun) if nun else 0,float(len(nol)),len(nol)/max(len(an1),len(an2),1)]
    else: f+=[0,0,0]
    name_best=max(nm_rat,nm_tsort,nmc_tset) if(n1 and n2) else 0
    addr_best=max(ad_rat,ad_tset) if(a1 and a2) else 0
    f+=[name_best,addr_best,max(name_best,addr_best),float(name_best<0.4 and addr_best>0.6),name_best*addr_best]
    f.append(float(r1["ct"]==r2["ct"] and r1["ct"]!=""))
    eid2=str(r2.get("entity_id","")); f+=[float(eid2.startswith("S2-")),float(eid2.startswith("S3-"))]
    return f

# ========================== Blocking ==========================
def tfidf_block(nq,iq,nd,id_,top_k=50,batch_sz=1000,min_sim=0.1,ngr=(3,4)):
    """TF-IDF blocking - use only for training (small data)."""
    if not nq or not nd: return {}
    vq=[(n,i) for n,i in zip(nq,iq) if n.strip()]
    vd=[(n,i) for n,i in zip(nd,id_) if n.strip()]
    if not vq or not vd: return {}
    qn,qi=zip(*vq); dn,di=zip(*vd)
    qn,qi,dn,di=list(qn),list(qi),list(dn),list(di)
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=ngr,max_features=200000,
                        sublinear_tf=True,dtype=np.float32)
    vec.fit(qn+dn); dtf=vec.transform(dn); cands={}
    nb=(len(qn)+batch_sz-1)//batch_sz
    for bi in range(nb):
        s=bi*batch_sz; e=min(s+batch_sz,len(qn))
        sim=cosine_similarity(vec.transform(qn[s:e]),dtf)
        for i in range(e-s):
            sc=sim[i]; k=min(top_k,len(sc))
            if k<=0: continue
            ix=np.argpartition(sc,-k)[-k:]
            c={di[j] for j in ix if sc[j]>min_sim}
            if c: cands[qi[s+i]]=c
        if (bi+1)%5==0 or bi==nb-1: print(f"      batch {bi+1}/{nb}")
    del vec,dtf; gc.collect()
    return cands

def inverted_index_block(s1_df, s23_df, max_cands_per_s1=80):
    """
    FAST inverted-index blocking for large test data.
    Creates multiple blocking keys per record and looks up via inverted index.
    O(n) per query instead of O(n*m) for cosine similarity.
    """
    t0 = time.time()
    # Build inverted index from S23
    idx = defaultdict(set)  # key -> set of entity_ids

    s23_nmc = s23_df["nmc"].values
    s23_ad = s23_df["ad"].values
    s23_ids = s23_df["entity_id"].values
    s23_nums = s23_df["ad_nums"].values
    s23_ft = s23_df["first_tok"].values

    for i in range(len(s23_df)):
        eid = s23_ids[i]
        nmc = s23_nmc[i] if isinstance(s23_nmc[i], str) else ""
        ad = s23_ad[i] if isinstance(s23_ad[i], str) else ""
        ft = s23_ft[i] if isinstance(s23_ft[i], str) else ""

        # Key 1: first 5 chars of name
        if nmc and len(nmc) >= 3:
            idx[f"n5:{nmc[:5]}"].add(eid)
        # Key 2: first 4 chars of name
        if nmc and len(nmc) >= 3:
            idx[f"n4:{nmc[:4]}"].add(eid)
        # Key 3: first 3 chars of name
        if nmc and len(nmc) >= 3:
            idx[f"n3:{nmc[:3]}"].add(eid)
        # Key 4: first token (significant word)
        if ft and len(ft) >= 3:
            idx[f"ft:{ft}"].add(eid)
        # Key 5: each significant word in name (>= 4 chars)
        if nmc:
            for w in nmc.split():
                if len(w) >= 4:
                    idx[f"w:{w}"].add(eid)
        # Key 6: each significant number in address (>= 3 digits)
        nums = s23_nums[i] if isinstance(s23_nums[i], set) else set()
        for n in nums:
            if len(n) >= 3:
                idx[f"num:{n}"].add(eid)
        # Key 7: address prefix (first 10 chars)
        if ad and len(ad) >= 5:
            idx[f"a10:{ad[:10]}"].add(eid)
        # Key 8: 4-char n-grams of name (first 3)
        if nmc and len(nmc) >= 4:
            for j in range(min(3, len(nmc)-3)):
                idx[f"ng:{nmc[j:j+4]}"].add(eid)

    print(f"      Index built: {len(idx):,} keys in {time.time()-t0:.1f}s")

    # Query
    s1_nmc = s1_df["nmc"].values
    s1_ad = s1_df["ad"].values
    s1_ids = s1_df["entity_id"].values
    s1_nums = s1_df["ad_nums"].values
    s1_ft = s1_df["first_tok"].values

    cands = {}
    for i in range(len(s1_df)):
        sid = s1_ids[i]
        nmc = s1_nmc[i] if isinstance(s1_nmc[i], str) else ""
        ad = s1_ad[i] if isinstance(s1_ad[i], str) else ""
        ft = s1_ft[i] if isinstance(s1_ft[i], str) else ""

        # Collect candidates with vote counting
        votes = defaultdict(int)

        # Name prefix keys (weighted)
        if nmc and len(nmc) >= 3:
            for eid in idx.get(f"n5:{nmc[:5]}", set()): votes[eid] += 3
            for eid in idx.get(f"n4:{nmc[:4]}", set()): votes[eid] += 2
            for eid in idx.get(f"n3:{nmc[:3]}", set()): votes[eid] += 1

        # First token
        if ft and len(ft) >= 3:
            for eid in idx.get(f"ft:{ft}", set()): votes[eid] += 2

        # Significant words
        if nmc:
            for w in nmc.split():
                if len(w) >= 4:
                    for eid in idx.get(f"w:{w}", set()): votes[eid] += 2

        # Numbers
        nums = s1_nums[i] if isinstance(s1_nums[i], set) else set()
        for n in nums:
            if len(n) >= 3:
                for eid in idx.get(f"num:{n}", set()): votes[eid] += 1

        # Address prefix
        if ad and len(ad) >= 5:
            for eid in idx.get(f"a10:{ad[:10]}", set()): votes[eid] += 3

        # N-grams
        if nmc and len(nmc) >= 4:
            for j in range(min(3, len(nmc)-3)):
                for eid in idx.get(f"ng:{nmc[j:j+4]}", set()): votes[eid] += 1

        if votes:
            # Take top candidates by vote count
            sorted_cands = sorted(votes.items(), key=lambda x: -x[1])
            cands[sid] = {eid for eid, v in sorted_cands[:max_cands_per_s1] if v >= 2}

        if (i+1) % 200000 == 0:
            print(f"      queried {i+1:,}/{len(s1_df):,}")

    print(f"      Blocking done: {sum(len(v) for v in cands.values()):,} pairs in {time.time()-t0:.1f}s")
    return cands

def numeric_block(s1c,s23c,min_ov=2):
    inv=defaultdict(list)
    for _,r in s23c.iterrows():
        for n in (r.get("ad_nums",set()) or set()):
            if len(n)>=2: inv[n].append(r["entity_id"])
    cands={}
    for _,r in s1c.iterrows():
        nums=r.get("ad_nums",set())
        if not nums: continue
        ct=defaultdict(int)
        for n in nums:
            if len(n)>=2 and n in inv:
                for e in inv[n]: ct[e]+=1
        c={e for e,cnt in ct.items() if cnt>=min_ov}
        if c: cands[r["entity_id"]]=c
    return cands

def multi_block_train(s1c,s23c):
    """TF-IDF blocking for training (small data)."""
    print("      [Name TF-IDF]")
    nc=tfidf_block(s1c["nmc"].fillna("").tolist(),s1c["entity_id"].tolist(),
                   s23c["nmc"].fillna("").tolist(),s23c["entity_id"].tolist(),top_k=50,min_sim=0.1)
    print("      [Addr TF-IDF]")
    ac=tfidf_block(s1c["ad"].fillna("").tolist(),s1c["entity_id"].tolist(),
                   s23c["ad"].fillna("").tolist(),s23c["entity_id"].tolist(),top_k=30,min_sim=0.15,ngr=(3,5))
    print("      [Numeric]")
    nuc=numeric_block(s1c,s23c)
    all_c=defaultdict(set)
    for d in [nc,ac,nuc]:
        for sid,cids in d.items(): all_c[sid]|=cids
    print(f"      N:{sum(len(v) for v in nc.values()):,} A:{sum(len(v) for v in ac.values()):,} Nu:{sum(len(v) for v in nuc.values()):,} U:{sum(len(v) for v in all_c.values()):,}")
    del nc,ac,nuc; gc.collect()
    return dict(all_c)

def multi_block_test(s1c,s23c):
    """Fast inverted-index blocking for test (large data)."""
    print("      [Inverted index blocking]")
    ii_c = inverted_index_block(s1c, s23c, max_cands_per_s1=80)
    print("      [Numeric]")
    nuc = numeric_block(s1c, s23c)
    all_c = defaultdict(set)
    for d in [ii_c, nuc]:
        for sid, cids in d.items(): all_c[sid] |= cids
    del ii_c, nuc; gc.collect()
    return dict(all_c)

def parse_gt(df):
    r={}
    sids=df["source1_entity_id"].values; mids=df["matched_entity_ids"].values
    for i in range(len(sids)):
        m=mids[i]
        r[sids[i]]=set() if pd.isna(m) or str(m).strip()=="" else set(str(m).split(","))
    return r

def cand_recall(c,g):
    h=t=0
    for s,ms in g.items():
        for m in ms:
            t+=1
            if m in c.get(s,set()): h+=1
    return h,t

# ========================== TRAINING ==========================
def run_training():
    print("="*70)
    print("ULTRA-FAST TRAINING")
    print("="*70)
    T0=time.time()

    gt_df=pd.read_csv(os.path.join(TRAIN_DIR,"train_ground_truth.tsv"),sep="\t")
    gt_all=parse_gt(gt_df); del gt_df; gc.collect()
    all_s1=sorted(gt_all.keys())
    sample_n=min(30000,len(all_s1))
    sampled=list(np.random.choice(all_s1,size=sample_n,replace=False))
    np.random.shuffle(sampled)
    val_n=int(sample_n*0.15)
    val_ids=set(sampled[:val_n]); train_ids=set(sampled[val_n:])
    print(f"    {sample_n:,} S1: train={len(train_ids):,} val={len(val_ids):,}")

    gt_tr={k:v for k,v in gt_all.items() if k in train_ids}
    gt_va={k:v for k,v in gt_all.items() if k in val_ids}
    needed=set(); 
    for ms in gt_tr.values(): needed|=ms
    for ms in gt_va.values(): needed|=ms

    s1a=pd.read_csv(os.path.join(TRAIN_DIR,"train_source1.tsv"),sep="\t")
    s1s=s1a[s1a["entity_id"].isin(set(sampled))].copy(); del s1a; gc.collect()
    s1s=preprocess_df(s1s)

    s2a=pd.read_csv(os.path.join(TRAIN_DIR,"train_source2.tsv"),sep="\t")
    s2p=s2a[s2a["entity_id"].isin(needed)]
    s2s=pd.concat([s2p,s2a.iloc[np.random.choice(len(s2a),min(80000,len(s2a)),False)]]).drop_duplicates(subset="entity_id")
    del s2a,s2p; gc.collect()

    s3a=pd.read_csv(os.path.join(TRAIN_DIR,"train_source3.tsv"),sep="\t")
    s3p=s3a[s3a["entity_id"].isin(needed)]
    s3s=pd.concat([s3p,s3a.iloc[np.random.choice(len(s3a),min(80000,len(s3a)),False)]]).drop_duplicates(subset="entity_id")
    del s3a,s3p; gc.collect()

    s23=pd.concat([s2s,s3s]).drop_duplicates(subset="entity_id"); del s2s,s3s; gc.collect()
    print(f"    S23: {len(s23):,}")
    s23=preprocess_df(s23); s23_ids=set(s23["entity_id"].values)

    s1tr=s1s[s1s["entity_id"].isin(train_ids)]; s1va=s1s[s1s["entity_id"].isin(val_ids)]
    tr_c={}; va_c={}
    for ct in sorted(s1s["ct"].unique()):
        if not ct: continue
        s1tc=s1tr[s1tr["ct"]==ct]; s1vc=s1va[s1va["ct"]==ct]; s23c=s23[s23["ct"]==ct]
        print(f"\n    {ct}: tr={len(s1tc):,} va={len(s1vc):,} s23={len(s23c):,}")
        if len(s23c)==0: continue
        if len(s1tc)>0: tr_c.update(multi_block_train(s1tc,s23c))
        if len(s1vc)>0: va_c.update(multi_block_train(s1vc,s23c))

    for sid,ms in gt_tr.items():
        if sid not in tr_c: tr_c[sid]=set()
        for m in ms:
            if m in s23_ids: tr_c[sid].add(m)

    th,tt=cand_recall(tr_c,gt_tr); vh,vt=cand_recall(va_c,gt_va)
    print(f"\n    Train: {th}/{tt}={th/max(tt,1):.4f}  Val: {vh}/{vt}={vh/max(vt,1):.4f}")

    s23_ix=s23.set_index("entity_id"); s23k=set(s23_ix.index)
    def comp(s1d,cn,gt):
        s1i=s1d.set_index("entity_id"); X,y,p=[],[],[]
        d=0
        for sid,cids in cn.items():
            if sid not in s1i.index: continue
            r1=s1i.loc[sid]; gm=gt.get(sid,set())
            for cid in cids:
                if cid not in s23k: continue
                X.append(pair_feats(r1,s23_ix.loc[cid])); y.append(1 if cid in gm else 0); p.append((sid,cid))
                d+=1
                if d%200000==0: print(f"      {d:,}")
        return np.array(X,dtype=np.float32),np.array(y),p

    print("\n    Computing features...")
    Xtr,ytr,ptr=comp(s1tr,tr_c,gt_tr); print(f"    Train: {len(Xtr):,} ({ytr.sum():,} pos)")
    Xva,yva,pva=comp(s1va,va_c,gt_va); print(f"    Val: {len(Xva):,} ({yva.sum():,} pos)")

    scale=(len(ytr)-ytr.sum())/max(ytr.sum(),1)
    print("\n    Training M1...")
    m1=lgb.LGBMClassifier(n_estimators=800,learning_rate=0.03,max_depth=8,num_leaves=127,
        min_child_samples=30,subsample=0.8,colsample_bytree=0.8,
        reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=scale,random_state=42,n_jobs=-1,verbose=-1)
    m1.fit(Xtr,ytr)

    print("    Hard-neg + M2...")
    tp=m1.predict_proba(Xtr)[:,1]; sw=np.ones(len(ytr)); sw[(ytr==0)&(tp>0.3)]=5.0
    m2=lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.02,max_depth=8,num_leaves=127,
        min_child_samples=30,subsample=0.8,colsample_bytree=0.8,
        reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=scale,random_state=42,n_jobs=-1,verbose=-1)
    m2.fit(Xtr,ytr,sample_weight=sw)

    def ev(model):
        vp=model.predict_proba(Xva)[:,1]; bf,bt=0,0.5
        for thr in np.arange(0.10,0.995,0.005):
            pd_d={}
            for i,(sid,cid) in enumerate(pva):
                if vp[i]>=thr: pd_d.setdefault(sid,set()).add(cid)
            for sid in gt_va: pd_d.setdefault(sid,set())
            f=macro_f05(pd_d,gt_va)
            if f>bf: bf,bt=f,thr
        return bt,bf

    t1,f1=ev(m1); t2,f2=ev(m2)
    print(f"    M1: F0.5={f1:.4f}@{t1:.3f}  M2: F0.5={f2:.4f}@{t2:.3f}")
    fm,ft,ff=( (m2,t2,f2) if f2>=f1 else (m1,t1,f1) )

    Xf=np.vstack([Xtr,Xva]); yf=np.concatenate([ytr,yva])
    fp_=fm.predict_proba(Xf)[:,1]; swf=np.ones(len(yf)); swf[(yf==0)&(fp_>0.3)]=5.0
    final=lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.02,max_depth=8,num_leaves=127,
        min_child_samples=30,subsample=0.8,colsample_bytree=0.8,
        reg_alpha=0.1,reg_lambda=1.0,scale_pos_weight=scale,random_state=42,n_jobs=-1,verbose=-1)
    final.fit(Xf,yf,sample_weight=swf)

    mp=os.path.join(MODELS_DIR,"lgbm_final.pkl")
    with open(mp,"wb") as f: pickle.dump({"model":final,"threshold":ft,"features":FEAT_NAMES,"val_f05":ff},f)
    print(f"\n    DONE: {(time.time()-T0)/60:.1f}min F0.5={ff:.4f} thr={ft:.3f}")

# ========================== TEST ==========================
def run_test():
    print("\n"+"="*70)
    print("TEST INFERENCE (inverted-index blocking)")
    print("="*70)
    T0=time.time()
    with open(os.path.join(MODELS_DIR,"lgbm_final.pkl"),"rb") as f: md=pickle.load(f)
    model=md["model"]; thr=md["threshold"]
    print(f"    thr={thr:.3f}")

    s1t=pd.read_csv(os.path.join(TEST_DIR,"test_source1.tsv"),sep="\t")
    s1t=preprocess_df(s1t)
    all_ids=set(s1t["entity_id"].tolist())
    matching={sid:set() for sid in all_ids}
    candidates={sid:set() for sid in all_ids}

    for country in sorted(s1t["ct"].unique()):
        if not country: continue
        print(f"\n  === {country} ===")
        s1c=s1t[s1t["ct"]==country]; print(f"    S1: {len(s1c):,}")

        # Load S2 for this country only
        print(f"    Loading S2...")
        s2t=pd.read_csv(os.path.join(TEST_DIR,"test_source2.tsv"),sep="\t")
        s2c=s2t[s2t["country"].apply(norm_country)==country].copy(); del s2t; gc.collect()
        print(f"    Loading S3...")
        s3t=pd.read_csv(os.path.join(TEST_DIR,"test_source3.tsv"),sep="\t")
        s3c=s3t[s3t["country"].apply(norm_country)==country].copy(); del s3t; gc.collect()
        s23c=pd.concat([s2c,s3c]).drop_duplicates(subset="entity_id"); del s2c,s3c; gc.collect()
        print(f"    S23: {len(s23c):,}")
        s23c=preprocess_df(s23c)

        print(f"    Blocking...")
        cands=multi_block_test(s1c,s23c)
        tc=sum(len(v) for v in cands.values())
        print(f"    Total candidates: {tc:,}")

        print(f"    Scoring...")
        s1_i=s1c.set_index("entity_id")
        s23_i=s23c.set_index("entity_id"); s23k=set(s23_i.index)
        cids=sorted(cands.keys()); csz=5000; nc=(len(cids)+csz-1)//csz
        for ci in range(nc):
            s=ci*csz; e=min(s+csz,len(cids)); X,pl=[],[]
            for sid in cids[s:e]:
                if sid not in s1_i.index: continue
                r1=s1_i.loc[sid]
                for cid in cands[sid]:
                    if cid not in s23k: continue
                    X.append(pair_feats(r1,s23_i.loc[cid])); pl.append((sid,cid))
                    candidates[sid].add(cid)
            if X:
                pr=model.predict_proba(np.array(X,dtype=np.float32))[:,1]
                for i,(sid,cid) in enumerate(pl):
                    if pr[i]>=thr: matching[sid].add(cid)
            if (ci+1)%100==0 or ci==nc-1: print(f"      chunk {ci+1}/{nc}")
        del s23c,s23_i; gc.collect()

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
    print(f"    Matched: {nm:,}/{len(all_ids):,}, {nt:,} total")
    print(f"    Inference: {(time.time()-T0)/60:.1f} min")

def main():
    step=sys.argv[1] if len(sys.argv)>1 else "full"
    if step in ("train","full"): run_training()
    if step in ("test","full"): run_test()
    if step in ("validate","full"):
        print("\n=== Validate ===")
        import subprocess
        r=subprocess.run([sys.executable,os.path.join(PROJECT_ROOT,"utils","validate_submission.py"),
            "--matching",os.path.join(OUTPUT_DIR,"matching_results.tsv"),
            "--candidate",os.path.join(OUTPUT_DIR,"candidate_pairs.tsv"),
            "--test-dir",TEST_DIR],capture_output=True,text=True)
        print(r.stdout)
        if r.stderr: print(r.stderr)

if __name__=="__main__": main()
