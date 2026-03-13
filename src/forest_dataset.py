import os
import cv2
import torch
from torch.utils.data import Dataset


class ForestDataset(Dataset):

    def __init__(self, folder):
        self.folder = folder
        self.images = [f for f in os.listdir(folder) if "_sat.jpg" in f]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.folder, img_name)

        mask_name = img_name.replace("_sat.jpg", "_mask.png")
        mask_path = os.path.join(self.folder, mask_name)

        # read image
        image = cv2.imread(img_path)

        # read mask
        mask = cv2.imread(mask_path, 0)

        # 🔹 Resize images to make training faster
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # normalize image
        image = image / 255.0
        mask = mask / 255.0

        # convert to torch tensors
        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return image, mask