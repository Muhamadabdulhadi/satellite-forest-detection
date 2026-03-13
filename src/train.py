import torch
import os
from torch.utils.data import DataLoader
from forest_dataset import ForestDataset
from model import UNet

# 1. DEVICE SETUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# LOAD DATASET
dataset = ForestDataset("data/train")
print("Dataset size:", len(dataset))

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

#  LOAD MODEL
model = UNet().to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 3

#  IOU METRIC FUNCTION

def iou_score(pred, mask):

    pred = (pred > 0.5).float()

    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection

    if union == 0:
        return 0

    return (intersection / union).item()


#  TRAINING LOOP

for epoch in range(epochs):

    print(f"\nStarting Epoch {epoch+1}/{epochs}")

    total_iou = 0

    for i,(images,masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        # forward pass
        preds = model(images)

        # compute loss
        loss = criterion(preds, masks)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate IoU
        iou = iou_score(preds, masks)
        total_iou += iou

        # print progress
        if i % 20 == 0:
            print(f"Batch {i}/{len(loader)} - Loss: {loss.item():.4f} IoU: {iou:.4f}")

    avg_iou = total_iou / len(loader)

    print(f"Epoch {epoch+1} Finished - Loss: {loss.item():.4f} Avg IoU: {avg_iou:.4f}")


# SAVE MODEL

os.makedirs("outputs/models", exist_ok=True)

torch.save(model.state_dict(), "outputs/models/model.pth")

print("Model saved successfully!")