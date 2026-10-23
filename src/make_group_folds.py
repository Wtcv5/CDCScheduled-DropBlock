import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20260809)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    group = df.groupby('group_id', as_index=False).agg(grade_index=('grade_index', 'first'))
    if df.groupby('group_id')['grade_index'].nunique().max() != 1:
        raise ValueError('One group_id contains multiple grade labels.')
    sgkf = StratifiedGroupKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    group['fold'] = -1
    X = group['group_id'].to_numpy()
    y = group['grade_index'].to_numpy()
    groups = group['group_id'].to_numpy()
    for fold, (_, va) in enumerate(sgkf.split(X, y, groups)):
        group.loc[va, 'fold'] = fold
    fold_map = dict(zip(group['group_id'], group['fold']))
    df['fold'] = df['group_id'].map(fold_map).astype(int)
    assert df.groupby('group_id')['fold'].nunique().max() == 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(df.drop_duplicates('group_id').groupby(['fold', 'grade_text']).size())
if __name__ == '__main__':
    main()
