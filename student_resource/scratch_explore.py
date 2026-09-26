import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd

# Load training data
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

print('=== SHAPES ===')
print(f'Source1: {s1.shape}')
print(f'Source2: {s2.shape}')
print(f'Source3: {s3.shape}')
print(f'Ground Truth: {gt.shape}')

print('\n=== COLUMNS ===')
print(f'S1: {list(s1.columns)}')
print(f'GT: {list(gt.columns)}')

print('\n=== S1 HEAD ===')
print(s1.head(5).to_string())

print('\n=== S2 HEAD ===')
print(s2.head(5).to_string())

print('\n=== S3 HEAD ===')
print(s3.head(5).to_string())

print('\n=== GT HEAD ===')
print(gt.head(10).to_string())

print('\n=== MISSING VALUES ===')
for name, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    missing = df.isnull().sum().to_dict()
    print(f'{name}: {missing}')

print('\n=== COUNTRY DISTRIBUTION ===')
col = 'country'
for name, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    counts = df[col].value_counts().to_dict()
    print(f'{name}: {counts}')

# Ground truth analysis
print('\n=== GT ANALYSIS ===')
gt['has_match'] = gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')
n_with_match = gt['has_match'].sum()
n_without = (~gt['has_match']).sum()
print(f'Total GT rows: {len(gt)}')
print(f'With matches: {n_with_match}')
print(f'Singletons (no match): {n_without}')
print(f'Singleton ratio: {n_without/len(gt):.4f}')

# Count matches per S1 entity
def count_matches(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    return len(str(x).split(','))

gt['n_matches'] = gt['matched_entity_ids'].apply(count_matches)
print(f'\nMatches per S1 entity distribution:')
print(gt['n_matches'].value_counts().sort_index().head(20).to_string())

# Source breakdown
gt_matched = gt[gt['has_match']]
all_match_ids = []
for ids in gt_matched['matched_entity_ids']:
    all_match_ids.extend(str(ids).split(','))
s2_matches = [x for x in all_match_ids if x.startswith('S2-')]
s3_matches = [x for x in all_match_ids if x.startswith('S3-')]
print(f'\nTotal match IDs: {len(all_match_ids)}')
print(f'S2 matches: {len(s2_matches)}')
print(f'S3 matches: {len(s3_matches)}')

# Name/address length stats
for name, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    name_len = df['business_name'].astype(str).str.len()
    addr_len = df['business_address'].astype(str).str.len()
    print(f'\n{name} Name len: min={name_len.min()}, mean={name_len.mean():.1f}, max={name_len.max()}')
    print(f'{name} Addr len: min={addr_len.min()}, mean={addr_len.mean():.1f}, max={addr_len.max()}')

# Test data shapes
ts1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t')
ts2 = pd.read_csv('dataset/test/test_source2.tsv', sep='\t')
ts3 = pd.read_csv('dataset/test/test_source3.tsv', sep='\t')
print(f'\n=== TEST SHAPES ===')
print(f'Test Source1: {ts1.shape}')
print(f'Test Source2: {ts2.shape}')
print(f'Test Source3: {ts3.shape}')

print('\n=== TEST COUNTRY DISTRIBUTION ===')
for name, df in [('TS1', ts1), ('TS2', ts2), ('TS3', ts3)]:
    counts = df[col].value_counts().to_dict()
    print(f'{name}: {counts}')

# Sample matched pairs to understand noise
print('\n=== SAMPLE MATCHED PAIRS ===')
sample_gt = gt_matched.head(10)
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    matched = str(row['matched_entity_ids']).split(',')
    s1_rec = s1[s1['entity_id'] == s1_id].iloc[0]
    print(f'\nS1: {s1_id}')
    print(f'  Name: {s1_rec["business_name"]}')
    print(f'  Addr: {s1_rec["business_address"]}')
    print(f'  Country: {s1_rec["country"]}')
    for mid in matched[:3]:
        mid = mid.strip()
        if mid.startswith('S2-'):
            rec = s2[s2['entity_id'] == mid]
        else:
            rec = s3[s3['entity_id'] == mid]
        if len(rec) > 0:
            rec = rec.iloc[0]
            print(f'  -> {mid}: {rec["business_name"]} | {rec["business_address"]} | {rec["country"]}')
