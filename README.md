# mnist-classic-perceptron

A from-scratch one-vs-rest Perceptron on classic MNIST (IDX files), with a comparison to `sklearn.linear_model.Perceptron`.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data (classic MNIST IDX)
Download the four IDX files into `data/` (gzipped is fine):
- train-images-idx3-ubyte(.gz)
- train-labels-idx1-ubyte(.gz)
- t10k-images-idx3-ubyte(.gz)
- t10k-labels-idx1-ubyte(.gz)

Check:
```bash
python src/check_mnist.py
```
You should see shapes (60000, 28, 28) for train images and (10000, 28, 28) for test images, with labels 0-9.