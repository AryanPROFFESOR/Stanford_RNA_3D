import torch
from src.data.dataset import RNADataset
from src.models.full_model import RNAFoldModel
from src.training.losses import total_loss

dataset = RNADataset("stanford-rna-3d-folding-2", max_length=128)

sample = dataset[0]

x = sample["x"].unsqueeze(0)   # (1,L)
coords = sample["coords"]
mask = sample["mask"]

model = RNAFoldModel()
pred = model(x)[0]

loss = total_loss(pred, coords, mask)

print("Target:", sample["target_id"])
print("Pred shape:", pred.shape)
print("Loss:", loss.item())
