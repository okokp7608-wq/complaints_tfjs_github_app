import pandas as pd
import json
from pathlib import Path

SRC = Path('complaints.parquet')
OUT = Path('.')
N_PER_CLASS = 1500

cols = [
    'channel','complaint_type','region_code','staff_career','process_days',
    'internal_consult','escalated','text_length','attachment_count','satisfaction',
    'resubmit','budget_impact','priority'
]

df = pd.read_parquet(SRC, columns=cols)
samples = []
for cls, g in df.groupby('priority', observed=True):
    samples.append(g.sample(n=min(N_PER_CLASS, len(g)), random_state=42))

sample = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)
for c in sample.columns:
    if str(sample[c].dtype) == 'category':
        sample[c] = sample[c].astype(str)
sample['region_code'] = sample['region_code'].astype(str)
sample['escalated'] = sample['escalated'].astype(int)

with open(OUT / 'sampled_complaints.json', 'w', encoding='utf-8') as f:
    json.dump(sample.to_dict(orient='records'), f, ensure_ascii=False, separators=(',', ':'))

meta = {
    'source_rows': int(len(df)),
    'sample_rows': int(len(sample)),
    'sample_strategy': f'priority 클래스별 균형 샘플링, 최대 {N_PER_CLASS:,}건씩 추출',
    'target': 'priority',
    'features': [c for c in cols if c != 'priority'],
    'class_counts': sample['priority'].value_counts().to_dict(),
}
with open(OUT / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(meta)
