import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class CsvDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

data = load_breast_cancer()
X_train, X_val, y_train, y_val = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

train_loader = DataLoader(CsvDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(CsvDataset(X_val, y_val), batch_size=32, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 61),
            nn.BatchNorm1d(61),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(61, 31),
            nn.BatchNorm1d(31),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(31, 1)
        )
    def forward(self, x): return self.net(x)

# X_train.shape[1] is explicitly passed to resolve the TypeError
model = MLP(X_train.shape[1])
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 40
best_val_loss = float('inf')
best_val_acc = 0.0
best_epoch = 0

for epoch in range(1, epochs + 1):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        criterion(model(bx), by).backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for bx, by in val_loader:
            out = model(bx)
            val_loss += criterion(out, by).item() * bx.size(0)
            preds = (torch.sigmoid(out) >= 0.5).float()
            correct += (preds == by).sum().item()
            
    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pt')

print(f"Params: {total_params:,} | Best val accuracy {best_val_acc:.2f} at epoch {best_epoch}")

new_model = MLP(X_train.shape[1])
new_model.load_state_dict(torch.load('best_model.pt'))
new_model.eval()

def compute_accuracy(m, loader):
    correct = 0
    with torch.no_grad():
        for bx, by in loader:
            preds = (torch.sigmoid(m(bx)) >= 0.5).float()
            correct += (preds == by).sum().item()
    return correct / len(loader.dataset)

final_mlp_acc = compute_accuracy(new_model, val_loader)
print(f"Reloaded model accuracy: {final_mlp_acc:.2f} (identical)")