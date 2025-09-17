# train_mlp.py
import json, math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path

# --- config
PHASE = 1  # 1 or 2
OUT = Path("docs/models/mlp_p1.json" if PHASE==1 else "docs/models/mlp_p2.json")
EPOCHS = 12
LR = 1e-3
WD = 1e-4
BS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- data: normalize to [0,1]; do NOT invert if your canvas already yields fg=1
tfm = transforms.Compose([
    transforms.ToTensor(),  # 0..1 with 1 = white pixel in MNIST (MNIST bg is black)
])
train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
test  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
train_loader = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(test,  batch_size=BS, shuffle=False, num_workers=2)

# Optional centering vector μ (mean image) in model space
with torch.no_grad():
    mu = (train.data.float() / 255.0).mean(dim=0).view(-1)

# --- model
class MLP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = x - mu if mu is not None else x
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits

class MLP2(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = x - mu if mu is not None else x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)  # logits

net = (MLP1() if PHASE==1 else MLP2()).to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)

def evaluate():
    net.eval(); correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = net(x)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred==y).sum().item()
    return correct/total

best = 0.0
for ep in range(1, EPOCHS+1):
    net.train()
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward(); opt.step()
    acc = evaluate()
    print(f"epoch {ep}: acc={acc:.4f}")
    if acc > best: best = acc

# --- export (Float32, row-major)
state = net.state_dict()
def w_and_b(layer):
    W = state[layer + ".weight"].cpu().contiguous().view(-1).float().tolist()
    b = state[layer + ".bias"].cpu().contiguous().view(-1).float().tolist()
    return W, b

payload = {"meta":{"arch":"mlp_p1" if PHASE==1 else "mlp_p2","n_features":784,"n_classes":10}}
if PHASE==1:
    W1,b1 = w_and_b("fc1"); W2,b2 = w_and_b("fc2")
    payload.update({"W1":W1,"b1":b1,"W2":W2,"b2":b2,"mu":mu.cpu().view(-1).float().tolist()})
else:
    W1,b1 = w_and_b("fc1"); W2,b2 = w_and_b("fc2"); W3,b3 = w_and_b("out")
    payload.update({"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"mu":mu.cpu().view(-1).float().tolist()})

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload))
print("wrote", OUT)
