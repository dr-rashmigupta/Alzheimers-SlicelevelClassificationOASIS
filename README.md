# Reimplemented OASIS MRI experiments

Keep all five implementation files together. `baseline.py` runs VGG19;
`sota.py` runs ResNet50; `novel.py` runs ViT-B/16 with optional LIME.
`mri_common.py` shares data handling and reporting; `keras_training.py` shares
TensorFlow training. These scripts use a main guard and do nothing on import.

## Setup and run

Use Python 3.11 in a fresh environment. Install the packages in requirements.txt;
you may omit TensorFlow when using only ViT, or omit torch/torchvision/timm/LIME
when using only the CNNs. Select matching torch/torchvision builds for your platform.
Package ranges are proposed requirements, not a tested lockfile.

```bash
python -m pip install -r requirements.txt
python baseline.py --data-dir /path/to/Data --output-dir runs/vgg19
python sota.py --data-dir /path/to/Data --manifest runs/vgg19/split_manifest.csv --output-dir runs/resnet50
python novel.py --data-dir /path/to/Data --manifest runs/vgg19/split_manifest.csv --output-dir runs/vit --lime
```

Without `--data-dir`, the scripts download `ninadaithal/imagesoasis` via kagglehub.
First runs also download pretrained weights. Provide either the folder containing
all four class folders, or its parent containing `Data`. Expected class order:

| ID | Class |
|---|---|
| 0 | Mild Dementia |
| 1 | Moderate Dementia |
| 2 | Non Demented |
| 3 | Very mild Dementia |

For Colab, upload the five implementation files into the working directory, install
the requirements in a setup cell, and run the same commands with a leading `!`.
No notebook-only syntax appears inside the Python files.

## Validate the cohort first

```bash
python baseline.py --data-dir /path/to/Data --prepare-only --output-dir runs/split_check
```

Patient IDs are extracted using `(OAS1_\d+)_`. Every scan/session/slice from a
matched patient stays in one partition. Supply `--patient-regex` with one capture
group for another naming convention. This verifies filename-derived identity,
not actual clinical identity. There is no content-based duplicate detection.

Splits target 70/15/15 **patient counts within each class**; actual image ratios
can differ. Rounding reserves at least one patient for validation and test.
A class with fewer than three patients, an unknown patient ID, or a patient with
multiple class labels causes an explicit error. Inspect the real cohort and
choose an appropriate study design; do not turn slices into independent patients
just to make the split work. For explicit image-level exploratory comparisons,
`--split-mode image` is available, with a warning. Pass the same split mode when
reusing its manifest. Image mode is not an exact reproduction of the originals'
scikit-learn split and is not patient-independent evaluation.

Reuse the same manifest for all three models. Paths in it are relative to the
class-folder root, so the dataset can be relocated. Imported manifests are
validated for paths, class mapping, patient IDs, and patient partition overlap.
They may intentionally contain a subset of the dataset. Output directories must
be empty; source images are never overwritten or copied into augmented folders.

## Retained designs and intentional corrections

- VGG19: ImageNet backbone frozen, 224×224 RGB, GAP → dropout 0.3 → Dense 256
  ReLU → dropout 0.3 → four-way softmax; batch 128.
- ResNet50: 160×160 RGB, last ten backbone layers trainable, same head plus
  L2(0.001) and batch normalization; batch 64; mild geometric/brightness augmentation.
- ViT: timm `vit_base_patch16_224`, all parameters trainable, dropout 0.1,
  four outputs; batch 64; mild affine/brightness augmentation.
- All: Adam at 1e-4, up to 20 epochs, early stopping patience 7. ResNet/ViT
  reduce learning rate on a validation-loss plateau (factor 0.5, patience 4).
- Salt/pepper policy retained as virtual training records: five extra variants
  per mild/moderate image, two per very-mild image, and corruption of 25% of
  non-demented originals in memory. Class weights use these expanded training
  counts. Use `--no-noise` and/or `--no-class-weights` for ablations.
- Noise is deterministic, generated after resizing, with unique selected pixels;
  no lossy JPEG resaving. It differs numerically from the old noise implementation.
- VGG/ResNet use their application-specific ImageNet preprocessing, replacing the
  original simple 1/255 scaling. ViT obtains normalization statistics from timm.
- Reports use the actual shared class mapping. Precision/recall/F1 are multiclass
  argmax metrics, avoiding misleading threshold-based Keras precision/recall.
- Checkpoints are saved on validation improvement, including before early stopping.
  ViT epoch loss uses the correct total class-weight denominator.
- LIME uses the same inference normalization and selected test examples, without
  hardcoded paths, undefined transforms, or an unconditional CUDA requirement.
- Mixed precision is opt-in for CNNs (`--mixed-precision`) and GPU-dependent.
  ViT uses float32. Seeds are set; cross-hardware bitwise equality is not promised.

These are corrected reimplementations, not exact reproductions of the old numeric
results. Do not reuse the earlier reported scores as outputs of this version.
The name `novel.py` preserves your filename; it does not establish methodological
novelty for a standard ViT fine-tuning experiment.

## Outputs and inference

Each run produces config, split manifest/summary, class weights, training history,
metrics, test predictions, confusion matrix CSV/PNG, and training curves.
Metrics are **slice-level** even when splits are patient-disjoint. No patient-level
aggregation, confidence intervals, or external validation are implemented.
LIME generates one image per true class and metadata when requested; it is not
clinical validation or a measure of explanation faithfulness.

CNN runs save `best.weights.h5` and an inference SavedModel. The exported model
expects float RGB inputs in [0,255] at 224×224 (VGG) or 160×160 (ResNet), and
includes preprocessing. Example after a completed run:

```python
import tensorflow as tf
model = tf.saved_model.load('runs/vgg19/inference_savedmodel')
probabilities = model.serve(tf.zeros((1, 224, 224, 3), dtype=tf.float32))
```

ViT saves `best_model.pth` with weights, class order, epoch, and normalization.
Recreate with `timm.create_model('vit_base_patch16_224', pretrained=False,
num_classes=4, drop_rate=0.1)`, load `model_state_dict`, call `eval()`, and apply
RGB resize to 224×224, ToTensor, and checkpoint mean/std normalization.
These are best-model inference checkpoints, not full optimizer-resume checkpoints.

For a short training smoke test, use `--epochs 1 --batch-size 2 --no-pretrained`
on a small, patient-disjoint cohort with every class in all splits. A randomly
initialized frozen VGG backbone is only a plumbing check, not a scientific baseline.

## Verification completed in this environment

CPU-only tests cover split repeatability/disjointness, manifest tampering,
insufficient patient counts, virtual augmentation counts, deterministic noise,
source preservation, class weights, and report label mapping. All entry points
are checked for syntax, import safety, help, and prepare-only execution.
TensorFlow, PyTorch, timm, actual OASIS images, and a GPU were unavailable here;
model training, checkpoint loading/export, and LIME execution are **not runtime
validated**. Run the short smoke test in your target environment before a full study.

```bash
python -m unittest test_data.py
```

API references used for preprocessing/model configuration:
- https://keras.io/api/applications/vgg/vgg_models/
- https://keras.io/api/applications/resnet/resnet_models/
- https://huggingface.co/docs/timm/reference/models
- https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/config.py
