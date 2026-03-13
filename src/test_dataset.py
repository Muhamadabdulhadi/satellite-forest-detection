import sys
import os

sys.path.append(os.path.dirname(__file__))

from forest_dataset import ForestDataset

dataset = ForestDataset("data/train")

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image shape:", image.shape)
print("Mask shape:", mask.shape)