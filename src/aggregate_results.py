import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from metrics import ordinal_metrics

def summarize_one(model_dir: Path, num_classes: int):
    fold_dirs = sorted([p for p in model_dir.glob('fold*') if p.is_dir()], key=lambda p: int(p.name.replace('fold', '')))
    if not fold_dirs:
        return None
    fold_rows = []
    all_group = []
    all_image = []
    theta_rows = []
    history_rows = []
    for fd in fold_dirs:
        fold = int(fd.name.replace('fold', ''))
        mp = fd / 'metrics.json'
        gp = fd / 'group_predictions.csv'
        ip = fd / 'image_predictions.csv'
        hp = fd / 'history.json'
        if not (mp.exists() and gp.exists() and ip.exists()):
            continue
        m = json.loads(mp.read_text(encoding='utf-8'))
        row = {'fold': fold, **{f'group_{k}': v for k, v in m['group_metrics'].items()}, **{f'image_{k}': v for k, v in m['image_metrics'].items()}}
        fold_rows.append(row)
        g = pd.read_csv(gp)
        g['fold'] = fold
        all_group.append(g)
        im = pd.read_csv(ip)
        im['fold'] = fold
        all_image.append(im)
        for b, theta in enumerate(m.get('theta_final', []), start=1):
            theta_rows.append({'fold': fold, 'block': b, 'theta': theta})
        if hp.exists():
            h = pd.read_json(hp)
            h['fold'] = fold
            history_rows.append(h)
    if not fold_rows:
        return None
    fold_df = pd.DataFrame(fold_rows)
    group_df = pd.concat(all_group, ignore_index=True)
    image_df = pd.concat(all_image, ignore_index=True)
    pooled_group = ordinal_metrics(group_df['true'], group_df['pred'])
    pooled_image = ordinal_metrics(image_df['true'], image_df['pred'])
    cm = confusion_matrix(group_df['true'], group_df['pred'], labels=list(range(num_classes)))
    metric_names = ['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'rank_mae', 'cross_grade_error_rate_ge2']
    stats = {}
    for name in metric_names:
        col = f'group_{name}'
        vals = fold_df[col].astype(float).to_numpy()
        stats[name] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    fold_df.to_csv(model_dir / 'fold_metrics.csv', index=False, encoding='utf-8-sig')
    group_df.to_csv(model_dir / 'oof_group_predictions.csv', index=False, encoding='utf-8-sig')
    image_df.to_csv(model_dir / 'oof_image_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(theta_rows).to_csv(model_dir / 'theta_final.csv', index=False, encoding='utf-8-sig')
    if history_rows:
        pd.concat(history_rows, ignore_index=True).to_json(model_dir / 'all_history.json', orient='records', force_ascii=False, indent=2)
    summary = {'model_dir': model_dir.name, 'n_folds': int(len(fold_df)), 'pooled_group': pooled_group, 'pooled_image': pooled_image, 'fold_mean_std_group': stats, 'group_confusion_matrix': cm.tolist()}
    (model_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-root', required=True)
    ap.add_argument('--num-classes', type=int, required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.models_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        s = summarize_one(model_dir, args.num_classes)
        if s is None:
            continue
        r = {'model': model_dir.name, **{f'pooled_{k}': v for k, v in s['pooled_group'].items()}}
        for k, v in s['fold_mean_std_group'].items():
            r[f'{k}_mean'] = v['mean']
            r[f'{k}_std'] = v['std']
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(['pooled_macro_f1', 'pooled_balanced_accuracy'], ascending=False)
    df.to_csv(out / 'model_summary.csv', index=False, encoding='utf-8-sig')
    print(df.to_string(index=False) if len(df) else 'No completed models.')
if __name__ == '__main__':
    main()
