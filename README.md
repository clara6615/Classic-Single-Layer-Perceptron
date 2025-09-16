# MNIST Classic Perceptron

A from-scratch one-vs-rest Perceptron on classic MNIST (IDX files), with a comparison to `sklearn.linear_model.Perceptron`.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data (classic MNIST IDX)
https://github.com/cvdfoundation/mnist?tab=readme-ov-file

Download the four IDX files into `data/` (gzipped is fine):
- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz

Check:
```bash
python src/check_mnist.py
```
You should see shapes (60000, 28, 28) for train images and (10000, 28, 28) for test images, with labels 0-9.

## Preprocessing & Split
- Images flattened to 784-d vectors.
- Float32 casting; default normalization to [0,1].
- Stratified train/val split (default 90/10) with `random_state=42` for reproducibility.

### Check:
```bash
python src/eda_quick.py
python src/check_split.py
```
#### Label balance:
Open reports/label_hist_train.png. If any class bar is wildly off (e.g., zero), labels or loader are wrong.

#### Determinism:
Run python src/check_split.py twice. The printed Train/Val shapes and the final “Deterministic…” line must be identical on both runs.

## From-Scratch One-vs-Rest Perceptron
- Binary rule: if `y * score <= 0` then `w <- w + y*x` (bias included by feature augmentation).
- OVR: train 10 binary perceptrons (class `c` vs rest), predict `argmax` of scores.
- Deterministic: `random_state=42`, fixed shuffle seeds.

### Tuning
- Averaged perceptron (`average=True`) typically adds ~1–3% accuracy.
- More epochs (20–30) help OVR convergence.
- Optional margin (`margin=0.05–0.1`) can stabilize updates.

Train & evaluate:
```bash
python src/train_scratch.py
```
Run `python src/train_scratch.py` twice; the printed accuracies must match to 4 decimals (given same subsets / epochs).

## Baseline: `sklearn.linear_model.Perceptron`
We train sklearn’s multiclass perceptron with the same features/split and compare.

Run:
```bash
python src/compare_sklearn.py
```

## Error Analysis
- Per-class accuracy tables (`reports/per_class_acc_*.csv`)
- Confusion matrices (val/test)
- Grids of most-confused pairs (validation set)

Run:
```bash
python src/error_analysis.py
```

## Results (full MNIST)
| method  | train | val   | test  |
|---------|------:|------:|------:|
| scratch | 0.928 | 0.916 | 0.920 |
| sklearn | 0.871 | 0.852 | 0.864 |

Artifacts:
- `reports/accuracy_comparison.png`
- `reports/confusion_val.png`, `reports/confusion_test.png`

### Notes
- Scratch uses **averaged** perceptron; sklearn’s `Perceptron` is **not averaged** (expected gap).
- Deterministic with `np.random.seed(42)` and `random_state=42`.

## Interactive sketchpad (quick demo)
Draw a digit and run the trained perceptron on it.

```bash
python src/app_draw.py
```