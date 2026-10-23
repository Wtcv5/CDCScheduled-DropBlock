import argparse
import json
import subprocess
import sys
from pathlib import Path


def execute(command, root, log_path=None):
    if log_path:
        with log_path.open('w', encoding='utf-8') as log:
            proc = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            if proc.wait():
                raise SystemExit(proc.returncode)
    else:
        subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-csv')
    parser.add_argument('--data-root')
    parser.add_argument('--folds')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--out')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    for field in ['data_csv', 'data_root', 'epochs']:
        value = getattr(args, field)
        if value is not None:
            cfg[field] = value
    if args.folds:
        cfg['folds'] = [int(x) for x in args.folds.split(',')]
    csv_path = Path(cfg['data_csv'])
    csv_path = csv_path if csv_path.is_absolute() else root / csv_path
    data_root = Path(cfg['data_root'])
    data_root = data_root if data_root.is_absolute() else root / data_root
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert df.groupby('group_id')['fold'].nunique().max() == 1
    labels = sorted(df['grade_index'].unique().tolist())
    assert labels == list(range(len(labels)))
    assert set(cfg['folds']).issubset(set(df['fold'].unique()))
    seeds = args.seeds or [cfg['seed']]
    if args.dry_run:
        print(json.dumps({'config': cfg, 'seeds': seeds, 'groups': int(df['group_id'].nunique()), 'images': len(df), 'classes': labels}, indent=2))
        return
    import torch
    if cfg.get('require_cuda', True) and not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    output_root = Path(args.out).resolve() if args.out else root / 'runs'
    for seed in seeds:
        output = output_root / cfg['run_name'] / f'seed_{seed}'
        logs = output / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        (output / 'models').mkdir(exist_ok=True)
        (output / 'tables').mkdir(exist_ok=True)
        (output / 'config.json').write_text(json.dumps({**cfg, 'seed': seed}, indent=2), encoding='utf-8')
        for experiment in cfg['experiments']:
            for fold in cfg['folds']:
                command = [sys.executable, 'src/train_cached_resize.py' if cfg.get('cache_resize', False) else 'src/train.py', '--csv', str(csv_path), '--root', str(data_root), '--experiment', experiment, '--num-classes', str(len(labels)), '--fold', str(fold), '--seed', str(seed), '--out', str(output / 'models')]
                fields = ['input_size', 'epochs', 'batch_size', 'workers', 'lr', 'weight_decay', 'theta_mode', 'theta_init', 'cdc_stages', 'backbone']
                for field in fields:
                    if field in cfg:
                        command += ['--' + field.replace('_', '-'), str(cfg[field])]
                if not cfg.get('pretrained', True):
                    command += ['--no-pretrained']
                if cfg.get('dropblock', False):
                    command += ['--dropblock', '--dropblock-max-prob', str(cfg['dropblock_max_prob']), '--dropblock-block-size', str(cfg['dropblock_block_size']), '--dropblock-schedule', cfg['dropblock_schedule']]
                execute(command, root, logs / f'{experiment}_fold{fold}.log')
        execute([sys.executable, 'src/aggregate_results.py', '--models-root', str(output / 'models'), '--num-classes', str(len(labels)), '--out', str(output / 'tables')], root)


if __name__ == '__main__':
    main()
