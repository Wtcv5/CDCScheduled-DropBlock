import argparse, json, platform, sys, shutil
from pathlib import Path
import pandas as pd
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--root', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    info = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'cuda_available': bool(torch.cuda.is_available()), 'cuda_device_count': int(torch.cuda.device_count()), 'cuda_devices': [], 'csv': str(args.csv), 'root': str(args.root) if args.root else None}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info['cuda_devices'].append({'index': i, 'name': p.name, 'total_memory_gb': round(p.total_memory / 1024 ** 3, 2)})
    try:
        import timm
        info['timm'] = timm.__version__
    except Exception as e:
        info['timm'] = None
        info['timm_error'] = repr(e)
    df = pd.read_csv(args.csv)
    required = {'image_path', 'group_id', 'grade_index', 'grade_text', 'fold'}
    info['missing_columns'] = sorted(required - set(df.columns))
    info['n_images'] = int(len(df))
    info['n_groups'] = int(df['group_id'].nunique()) if 'group_id' in df else None
    info['grades'] = df[['grade_index', 'grade_text']].drop_duplicates().sort_values('grade_index').to_dict('records') if {'grade_index', 'grade_text'}.issubset(df.columns) else []
    info['num_classes'] = len(info['grades'])
    if 'group_id' in df and 'fold' in df:
        leaks = int((df.groupby('group_id')['fold'].nunique() > 1).sum())
        info['groups_crossing_folds'] = leaks
    else:
        info['groups_crossing_folds'] = None
    if args.root and 'image_path' in df:
        root = Path(args.root)
        missing = 0
        for p in df['image_path'].astype(str):
            pp = Path(p)
            if not pp.is_absolute():
                pp = root / pp
            if not pp.exists():
                missing += 1
        info['missing_images'] = int(missing)
    else:
        info['missing_images'] = None
    total, used, free = shutil.disk_usage(Path(args.out).parent)
    info['disk_free_gb'] = round(free / 1024 ** 3, 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))
    fatal = []
    if info['timm'] is None:
        fatal.append('timm not available')
    if info['missing_columns']:
        fatal.append('dataset schema incomplete')
    if info['groups_crossing_folds']:
        fatal.append('group leakage across folds')
    if info['missing_images']:
        fatal.append('missing image files')
    if fatal:
        raise SystemExit('PREFLIGHT FAILED: ' + '; '.join(fatal))
if __name__ == '__main__':
    main()
