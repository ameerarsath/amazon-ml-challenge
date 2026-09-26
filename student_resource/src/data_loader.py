"""
Data loading utilities for Business Entity Resolution.
Loads and returns training/test data as pandas DataFrames.
"""
import pandas as pd
from typing import Tuple, Dict, Set


def load_tsv(path: str) -> pd.DataFrame:
    """Load a TSV file using tab separator."""
    return pd.read_csv(path, sep="\t")


def load_train_data(train_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all training data files.
    
    Returns:
        Tuple of (source1, source2, source3, ground_truth) DataFrames
    """
    import os
    s1 = load_tsv(os.path.join(train_dir, "train_source1.tsv"))
    s2 = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    s3 = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    gt = load_tsv(os.path.join(train_dir, "train_ground_truth.tsv"))
    return s1, s2, s3, gt


def load_test_data(test_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test data files.
    
    Returns:
        Tuple of (source1, source2, source3) DataFrames
    """
    import os
    s1 = load_tsv(os.path.join(test_dir, "test_source1.tsv"))
    s2 = load_tsv(os.path.join(test_dir, "test_source2.tsv"))
    s3 = load_tsv(os.path.join(test_dir, "test_source3.tsv"))
    return s1, s2, s3


def parse_ground_truth(gt: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse ground truth into a dict mapping S1 entity_id to set of matched entity_ids.
    
    Returns:
        Dict mapping source1_entity_id -> set of matched entity_ids (may be empty)
    """
    result = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row.get("matched_entity_ids", "")
        if pd.isna(matched) or str(matched).strip() == "":
            result[s1_id] = set()
        else:
            result[s1_id] = set(str(matched).split(","))
    return result
