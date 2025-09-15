# mnist-classic-perceptron

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

Check:
```bash
python src/eda_quick.py
python src/check_split.py
```