import os, numpy as np, matplotlib.pyplot as plt
from perceptron import OneVsRestPerceptron

np.random.seed(42)
os.makedirs("reports", exist_ok=True)

W = np.load("models/perceptron_scratch_weights.npz")["W"]  # (10, 785) includes bias
Wnobias = W[:, :-1]                                        # drop bias
fig = plt.figure(figsize=(10,4))
for c in range(10):
    ax = plt.subplot(2,5,c+1)
    ax.imshow(Wnobias[c].reshape(28,28), cmap="bwr")
    ax.set_title(f"class {c}")
    ax.axis("off")
fig.suptitle("Perceptron weight templates (scratch OVR)")
fig.tight_layout()
out = "reports/weight_templates.png"
fig.savefig(out, dpi=150)
print("Saved", out)
