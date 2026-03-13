import os
import cv2
import numpy as np

input_folder = "data/train"

files = os.listdir(input_folder)

for file in files:

    if "_mask.png" in file:

        path = os.path.join(input_folder,file)

        mask = cv2.imread(path,0)

        binary_mask = np.where(mask>0,1,0)

        cv2.imwrite(path,binary_mask*255)

print("Masks converted")