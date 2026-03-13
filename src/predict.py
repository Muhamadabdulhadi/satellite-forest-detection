import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from model import UNet

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet().to(device)
model.load_state_dict(torch.load("outputs/models/model.pth", map_location=device))
model.eval()

# -----------------------------
# FOREST AREA FUNCTION
# -----------------------------
def forest_percentage(mask):

    forest_pixels = np.sum(mask == 1)
    total_pixels = mask.size

    return (forest_pixels / total_pixels) * 100


# -----------------------------
# IMAGE FOLDER
# -----------------------------
folder = "data/train"

files = [f for f in os.listdir(folder) if "_sat.jpg" in f]

print("Total images found:", len(files))

# create output folder
os.makedirs("outputs/predictions", exist_ok=True)

# -----------------------------
# PROCESS MULTIPLE IMAGES
# -----------------------------
for file in files[:5]:

    img_path = os.path.join(folder, file)

    print("\nProcessing:", img_path)

    image = cv2.imread(img_path)

    if image is None:
        print("Error loading image")
        continue

    image = cv2.resize(image, (256,256))

    # -----------------------------
    # PREPROCESS IMAGE
    # -----------------------------
    input_img = image / 255.0
    input_img = torch.tensor(input_img).permute(2,0,1).unsqueeze(0).float().to(device)

    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------
    with torch.no_grad():
        pred = model(input_img)

    pred = pred.squeeze().cpu().numpy()

    # -----------------------------
    # PREDICTION STATISTICS
    # -----------------------------
    print("Prediction min:", pred.min())
    print("Prediction max:", pred.max())
    print("Prediction mean:", pred.mean())
    print("Prediction std:", pred.std())

    # -----------------------------
    # NORMALIZE FOR VISUALIZATION
    # -----------------------------
    pred_vis = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    # -----------------------------
    # STRONGER THRESHOLD
    # -----------------------------
    mask = (pred > 0.995).astype(np.uint8)

    # -----------------------------
    # DEBUG: MASK VALUES
    # -----------------------------
    unique_vals = np.unique(mask)
    print("Mask unique values:", unique_vals)

    forest_pixels = np.sum(mask == 1)
    non_forest_pixels = np.sum(mask == 0)

    print("Forest pixels:", forest_pixels)
    print("Non-forest pixels:", non_forest_pixels)

    # -----------------------------
    # FOREST COVERAGE
    # -----------------------------
    forest_area = forest_percentage(mask)

    print(f"Forest Coverage: {forest_area:.2f}%")

    # -----------------------------
    # CREATE FOREST OVERLAY
    # -----------------------------
    overlay = image.copy()
    overlay[mask == 1] = [0,255,0]

    result = cv2.addWeighted(image,0.7,overlay,0.3,0)

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------
    base_name = file.replace("_sat.jpg","")

    cv2.imwrite(
        f"outputs/predictions/{base_name}_mask.png",
        mask * 255
    )

    cv2.imwrite(
        f"outputs/predictions/{base_name}_overlay.png",
        result
    )

    cv2.imwrite(
        f"outputs/predictions/{base_name}_probability.png",
        (pred_vis * 255).astype(np.uint8)
    )

    print("Saved results for", file)

    # -----------------------------
    # VISUALIZE RESULTS
    # -----------------------------
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Satellite Image")
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Prediction Probability Map")
    plt.imshow(pred_vis, cmap="inferno")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(f"Forest Detection ({forest_area:.2f}%)")
    plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()