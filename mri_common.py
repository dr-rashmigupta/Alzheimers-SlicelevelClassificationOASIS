"""Shared, non-destructive data preparation and reporting for OASIS experiments."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import warnings
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = ('Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very mild Dementia')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parser(name, batch_size):
    p = argparse.ArgumentParser(description=name)
    p.add_argument('--data-dir', type=Path, help='Class-folder root or its parent; otherwise download imagesoasis')
    p.add_argument('--output-dir', type=Path, default=Path('runs') / name)
    p.add_argument('--manifest', type=Path, help='Reuse a previously saved split_manifest.csv')
    p.add_argument('--split-mode', choices=['patient', 'image'], default='patient')
    p.add_argument('--patient-regex', default=r'(OAS1_\d+)_', help='First capture group is the patient ID')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=batch_size)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--noise-amount', type=float, default=.02)
    p.add_argument('--no-noise', action='store_true')
    p.add_argument('--no-class-weights', action='store_true')
    p.add_argument('--no-pretrained', action='store_true', help='Random weights for smoke tests, not pretrained comparison')
    p.add_argument('--prepare-only', action='store_true', help='Validate and save splits without importing an ML framework')
    return p


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + '\n')


def discover(root):
    if root is None:
        import kagglehub
        root = Path(kagglehub.dataset_download('ninadaithal/imagesoasis'))
    root = Path(root).resolve()
    candidates = [root, root / 'Data']
    candidates += sorted(p for p in root.rglob('Data') if p.is_dir())
    root = next((p for p in candidates if all((p / c).is_dir() for c in CLASSES)), None)
    if root is None:
        raise ValueError(f'Expected all four class folders: {CLASSES}')
    rows = []
    for label, cls in enumerate(CLASSES):
        files = sorted(p for p in (root / cls).rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
        if not files:
            raise ValueError(f'No images in {cls}')
        for path in files:
            rows.append(dict(path=path.relative_to(root).as_posix(), label=label, patient_id=''))
    return root, rows


def make_splits(rows, mode, seed, patient_regex):
    """Class-stratified 70/15/15 by patient count (image counts can differ)."""
    rng = random.Random(seed)
    pattern = re.compile(patient_regex)
    if pattern.groups < 1:
        raise ValueError('--patient-regex must contain a capturing group')
    groups = {}
    patient_labels = {}
    for row in rows:
        match = pattern.search(Path(row['path']).name)
        row['patient_id'] = match.group(1) if match else ''
        if mode == 'patient' and not row['patient_id']:
            raise ValueError(f"Cannot identify patient in {row['path']}; supply --patient-regex. No automatic image fallback.")
        key = row['patient_id'] if mode == 'patient' else row['path']
        if mode == 'patient' and key in patient_labels and patient_labels[key] != row['label']:
            raise ValueError(f'Patient {key} has multiple labels; curate a consistent cohort before splitting.')
        patient_labels[key] = row['label']
        groups.setdefault(row['label'], {}).setdefault(key, []).append(row)
    for label, by_id in sorted(groups.items()):
        keys = sorted(by_id)
        rng.shuffle(keys)
        if len(keys) < 3:
            raise ValueError(f'{CLASSES[label]} has only {len(keys)} independent {mode} units; cannot populate all three splits.')
        n_val = max(1, round(.15 * len(keys)))
        n_test = max(1, round(.15 * len(keys)))
        for i, key in enumerate(keys):
            split = 'val' if i < n_val else 'test' if i < n_val + n_test else 'train'
            for row in by_id[key]:
                row['split'] = split
    return rows


def validate_manifest(root, rows, mode, patient_regex):
    pattern = re.compile(patient_regex)
    if pattern.groups < 1:
        raise ValueError('--patient-regex needs a capturing group')
    seen, owners = set(), {}
    for row in rows:
        row['label'] = int(row['label'])
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid image path: {path}')
        if path in seen:
            raise ValueError(f'Duplicate manifest path: {path}')
        seen.add(path)
        if row['split'] not in ('train', 'val', 'test') or row['label'] not in range(4):
            raise ValueError('Invalid split or label in manifest')
        if Path(row['path']).parts[0] != CLASSES[row['label']]:
            raise ValueError(f"Folder/label mismatch: {row['path']}")
        match = pattern.search(path.name)
        derived = match.group(1) if match else ''
        if row.get('patient_id', '') != derived:
            raise ValueError(f'Patient ID does not match filename: {path.name}')
        if mode == 'patient':
            if not derived:
                raise ValueError(f'Missing patient ID: {path.name}')
            assignment = (row['split'], row['label'])
            if derived in owners and owners[derived] != assignment:
                raise ValueError(f'Patient overlap or inconsistent label: {derived}')
            owners[derived] = assignment
    for split in ('train', 'val', 'test'):
        if {r['label'] for r in rows if r['split'] == split} != set(range(4)):
            raise ValueError(f'Every class must be represented in {split}')


def prepare(args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.learning_rate <= 0:
        raise ValueError('Epochs, batch size, patience and learning rate must be positive')
    if not 0 <= args.noise_amount <= 1:
        raise ValueError('Noise amount must be in [0, 1]')
    random.seed(args.seed)
    np.random.seed(args.seed)
    root, rows = discover(args.data_dir)
    if args.manifest:
        with args.manifest.open(newline='') as f:
            rows = list(csv.DictReader(f))
    else:
        rows = make_splits(rows, args.split_mode, args.seed, args.patient_regex)
    validate_manifest(root, rows, args.split_mode, args.patient_regex)
    if args.split_mode == 'image':
        warnings.warn('Image-level split: related slices may span partitions; this is not patient-independent evaluation.')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out}; use a new --output-dir')
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = out
    with (out / 'split_manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'patient_id', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    partitions = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    summary = {s: {c: sum(r['label'] == i for r in rs) for i, c in enumerate(CLASSES)} for s, rs in partitions.items()}
    summary['patient_overlap'] = {}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        ids_a = {r['patient_id'] for r in partitions[a] if r['patient_id']}
        ids_b = {r['patient_id'] for r in partitions[b] if r['patient_id']}
        summary['patient_overlap'][f'{a}_{b}'] = len(ids_a & ids_b)
    save_json(out / 'split_summary.json', summary)
    save_json(out / 'config.json', {**vars(args), 'data_root': root, 'classes': CLASSES})
    print(json.dumps(summary, indent=2))
    return root, partitions


def training_records(rows, seed, enabled=True):
    """Virtual replicas: +5 mild/moderate, +2 very mild, corrupt 25% non-demented.
    Validation/test never call this function. Originals on disk are never changed.
    """
    rng = random.Random(seed)
    normal = sorted(r['path'] for r in rows if r['label'] == 2)
    noisy_normal = set(rng.sample(normal, int(.25 * len(normal)))) if enabled else set()
    result = []
    for row in rows:
        result.append({**row, 'noise_variant': 1 if row['path'] in noisy_normal else 0})
        copies = {0: 5, 1: 5, 2: 0, 3: 2}[row['label']] if enabled else 0
        result.extend({**row, 'noise_variant': i + 1} for i in range(copies))
    return result


def load_image(root, row, size, seed, amount):
    with Image.open(root / row['path']) as image:
        array = np.array(image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR))
    variant = row.get('noise_variant', 0)
    if variant:
        key = f"{seed}:{row['path']}:{variant}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        n = int(amount * size * size)
        pixels = rng.choice(size * size, size=n, replace=False)
        array.reshape(-1, 3)[pixels] = rng.choice([0, 255], size=(n, 1))
    return array


def class_weights(rows, enabled=True):
    counts = np.bincount([r['label'] for r in rows], minlength=4)
    if np.any(counts == 0):
        raise ValueError('Missing training class')
    return (len(rows) / (4 * counts)).astype(np.float32) if enabled else np.ones(4, np.float32)


def report(out, rows, probabilities, history):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(rows), 4) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid prediction shape or nonfinite probabilities')
    labels = np.array([r['label'] for r in rows])
    predictions = probabilities.argmax(1)
    metrics = classification_report(labels, predictions, labels=list(range(4)), target_names=CLASSES, output_dict=True, zero_division=0)
    metrics['accuracy'] = float(accuracy_score(labels, predictions))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(labels, predictions))
    save_json(out / 'metrics.json', metrics)
    save_json(out / 'history.json', history)
    with (out / 'predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'patient_id', 'true_label', 'predicted_label', *CLASSES])
        for row, prediction, probs in zip(rows, predictions, probabilities):
            writer.writerow([row['path'], row['patient_id'], row['label'], int(prediction), *probs.tolist()])
    cm = confusion_matrix(labels, predictions, labels=list(range(4)))
    np.savetxt(out / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(cm / cm.sum(axis=1, keepdims=True), display_labels=CLASSES).plot(ax=ax, values_format='.2f', colorbar=False)
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for key in [metric, 'val_' + metric]:
            ax.plot(range(1, len(history[key]) + 1), history[key], label=key)
        ax.set_xlabel('Epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'training_curves.png', dpi=150)
    plt.close(fig)
    print(classification_report(labels, predictions, labels=list(range(4)), target_names=CLASSES, digits=4, zero_division=0))
