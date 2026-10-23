import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from dataset import RockMassImageDataset
from models import create_experiment_model
from ordinal import coral_loss, coral_predict, class_probs_from_cumulative
from metrics import ordinal_metrics
from cdc import get_theta_values, CentralDifferenceDWConv2d

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_eta(seconds):
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'

def make_transforms(size, random_erasing=None):
    train_tf = transforms.Compose([transforms.Resize(int(size * 1.08)), transforms.RandomCrop(size), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.04), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if random_erasing:
        train_tf.transforms.append(transforms.RandomErasing(p=random_erasing['p'], scale=random_erasing['scale'], ratio=random_erasing['ratio'], value=random_erasing.get('value', 0)))
    val_tf = transforms.Compose([transforms.Resize(int(size * 1.08)), transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return (train_tf, val_tf)

@torch.no_grad()
def predict(model, loader, device, head_type):
    model.eval()
    rows = []
    for batch in loader:
        x = batch['image'].to(device, non_blocking=True)
        y = batch['label'].numpy()
        logits = model(x)
        if head_type == 'ce':
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(1)
        else:
            probs = class_probs_from_cumulative(logits)
            pred = coral_predict(logits)
        probs = probs.cpu().numpy()
        pred = pred.cpu().numpy()
        for i in range(len(y)):
            row = {'group_id': batch['group_id'][i], 'path': batch['path'][i], 'true': int(y[i]), 'pred': int(pred[i])}
            if 'view' in batch:
                row['view'] = batch['view'][i]
            for c in range(probs.shape[1]):
                row[f'p_{c}'] = float(probs[i, c])
            rows.append(row)
    return rows

def aggregate_groups(rows, num_classes):
    df = pd.DataFrame(rows)
    prob_cols = [f'p_{c}' for c in range(num_classes)]
    aggs = {c: 'mean' for c in prob_cols}
    aggs['true'] = 'first'
    g = df.groupby('group_id', as_index=False).agg(aggs)
    g['pred'] = g[prob_cols].to_numpy().argmax(1)
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--root', default=None)
    ap.add_argument('--experiment', required=True, choices=['M0', 'M1', 'M2', 'M3'])
    ap.add_argument('--num-classes', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--input-size', type=int, default=320)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=12)
    ap.add_argument('--workers', type=int, default=0, help='DataLoader workers; Windows default 0 (main-process loading)')
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--theta-lr', type=float, default=0.001, help='Learning rate for CDC alpha/theta params; default 10x main lr because dL/dtheta is small (sigmoid compression + near-zero kernel-sum center)')
    ap.add_argument('--weight-decay', type=float, default=0.05)
    ap.add_argument('--theta-mode', choices=['fixed', 'shared', 'learnable'], default='learnable')
    ap.add_argument('--theta-init', type=float, default=0.05)
    ap.add_argument('--cdc-stages', default='0', help='e.g. 0 or 0,1 or 0,1,2,3')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--seed', type=int, default=20260809)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--local-variant', choices=['atto', 'tiny'], default='tiny')
    ap.add_argument('--backbone', default='convnextv2_tiny.fcmae_ft_in22k_in1k', help='timm model name; e.g. resnet50.a1_in1k for the comparison baseline. CDC requires a ConvNeXt-style backbone.')
    ap.add_argument('--weighted-ce', action='store_true', help='diagnostic option for imbalanced binary development data')
    ap.add_argument('--dropblock', action='store_true', help='insert ScheduledDropBlock2d between Stage 3 and Stage 4')
    ap.add_argument('--dropblock-max-prob', type=float, default=0.1)
    ap.add_argument('--dropblock-block-size', type=int, default=3)
    ap.add_argument('--dropblock-schedule', choices=['linear', 'cosine', 'constant'], default='linear')
    ap.add_argument('--gc', action='store_true', help='insert a GCNet global context block after Stage 3 (before Stage 4)')
    ap.add_argument('--gc-ratio', type=int, default=16, help='GC bottleneck reduction ratio (standard lightweight=16)')
    ap.add_argument('--random-erasing', action='store_true', help='lightweight Random Erasing on training images')
    ap.add_argument('--re-p', type=float, default=0.25, help='Random Erasing apply probability')
    ap.add_argument('--re-scale', default='0.02,0.10', help='Random Erasing area range, e.g. 0.02,0.10')
    ap.add_argument('--re-ratio', default='0.3,3.3', help='Random Erasing aspect range, e.g. 0.3,3.3')
    args = ap.parse_args()
    seed_all(args.seed + args.fold)
    if args.experiment in {'M2', 'M3'} and args.num_classes < 3:
        raise SystemExit('CORAL experiment blocked: num_classes < 3. For K=2, CORAL degenerates to a single binary threshold and cannot validate ordinal learning as a paper contribution.')
    cdc_stages = tuple((int(x) for x in args.cdc_stages.split(',') if x.strip()))
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU is required but not available. This training code is GPU-only (AMP / pin_memory / non_blocking).')
    device = torch.device('cuda')
    re_cfg = None
    if args.random_erasing:
        re_cfg = {'p': args.re_p, 'scale': tuple((float(v) for v in args.re_scale.split(','))), 'ratio': tuple((float(v) for v in args.re_ratio.split(','))), 'value': 0}
    train_tf, val_tf = make_transforms(args.input_size, random_erasing=re_cfg)
    train_ds = RockMassImageDataset(args.csv, args.root, train_tf, fold=args.fold, train=True)
    val_ds = RockMassImageDataset(args.csv, args.root, val_tf, fold=args.fold, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = create_experiment_model(args.experiment, args.num_classes, pretrained=not args.no_pretrained, cdc_stages=cdc_stages, theta_mode=args.theta_mode, theta_init=args.theta_init, local_variant=args.local_variant, model_name=args.backbone, use_dropblock=args.dropblock, dropblock_max_prob=args.dropblock_max_prob, dropblock_block_size=args.dropblock_block_size, dropblock_schedule=args.dropblock_schedule, use_gc=args.gc, gc_ratio=args.gc_ratio).to(device)
    head_type = model.head_type
    if args.weighted_ce:
        counts = np.bincount(train_ds.df['grade_index'].astype(int).to_numpy(), minlength=args.num_classes)
        weights = counts.sum() / (args.num_classes * np.maximum(counts, 1))
        ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        weights = None
        ce = torch.nn.CrossEntropyLoss()
    theta_params = [p for name, p in model.named_parameters() if 'alpha' in name]
    theta_ids = {id(p) for p in theta_params}
    base_params = [p for p in model.parameters() if id(p) not in theta_ids]
    param_groups = [{'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}]
    if theta_params:
        param_groups.append({'params': theta_params, 'lr': args.theta_lr, 'weight_decay': args.weight_decay})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        if model.dropblock is not None:
            model.dropblock.set_epoch(epoch + 1, args.epochs)
        model.train()
        total, n = (0.0, 0)
        for batch in train_loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(x)
                if head_type == 'ce':
                    loss = ce(logits, y)
                else:
                    loss = coral_loss(logits, y, args.num_classes)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(y)
            n += len(y)
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        train_loss = total / max(n, 1)
        eta_s = epoch_time / (epoch + 1) * (args.epochs - epoch - 1)
        print(f'Epoch {epoch + 1}/{args.epochs} | train_loss {train_loss:.4f} | {epoch_time:.1f}s | GPU peak {gpu_peak_mb:.0f} MB | ETA {format_eta(eta_s)}')
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_time_s': round(epoch_time, 2), 'gpu_peak_mb': round(gpu_peak_mb, 1), 'eta_s': round(eta_s, 1), 'theta_values': get_theta_values(model.backbone)})
    rows = predict(model, val_loader, device, head_type)
    image_metrics = ordinal_metrics([r['true'] for r in rows], [r['pred'] for r in rows])
    groups = aggregate_groups(rows, args.num_classes)
    group_metrics = ordinal_metrics(groups['true'], groups['pred'])
    run = Path(args.out) / f'{args.experiment}_size{args.input_size}' / f'fold{args.fold}'
    run.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run / 'final.pt')
    pd.DataFrame(rows).to_csv(run / 'image_predictions.csv', index=False, encoding='utf-8-sig')
    groups.to_csv(run / 'group_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(history).to_json(run / 'history.json', orient='records', force_ascii=False, indent=2)
    result = {'experiment': args.experiment, 'head_type': head_type, 'fold': args.fold, 'num_classes': args.num_classes, 'input_size': args.input_size, 'cdc_stages': cdc_stages, 'theta_mode': args.theta_mode, 'theta_init': args.theta_init, 'local_variant': args.local_variant, 'backbone': args.backbone, 'weighted_ce': bool(args.weighted_ce), 'dropblock': {'enabled': bool(args.dropblock), 'max_prob': args.dropblock_max_prob, 'block_size': args.dropblock_block_size, 'schedule': args.dropblock_schedule}, 'random_erasing': {'enabled': bool(args.random_erasing), 'p': args.re_p, 'scale': args.re_scale, 'ratio': args.re_ratio}, 'gc': {'enabled': bool(args.gc), 'ratio': args.gc_ratio}, 'ce_class_weights': weights.tolist() if weights is not None else None, 'theta_final': get_theta_values(model.backbone), 'image_metrics': image_metrics, 'group_metrics': group_metrics, 'group_confusion_matrix': confusion_matrix(groups['true'], groups['pred'], labels=list(range(args.num_classes))).tolist()}
    (run / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
