# Satellite Forest Detection using Deep Learning 🌲

This project uses a **U-Net deep learning model** to detect forest regions in satellite imagery.

The system predicts forest coverage and visualizes detected forest areas on satellite images.

---

## Features

- Semantic segmentation using **U-Net**
- Forest detection from satellite images
- Forest coverage percentage calculation
- Probability heatmap visualization
- Forest overlay on satellite imagery

---

## Model Architecture

The model uses a **U-Net convolutional neural network** implemented with PyTorch.

Input:
Satellite Image (256x256)

Output:
Binary forest segmentation mask.

---

## Example Output

Satellite Image → AI Prediction → Forest Overlay

Forest Coverage Example:

Forest Coverage: **72.4%**

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib

---

## Project Pipeline

Satellite Image  
↓  
U-Net Deep Learning Model  
↓  
Probability Map  
↓  
Binary Segmentation Mask  
↓  
Forest Coverage Calculation  
↓  
Visualization

---
## Run Training

python train.py

## Run Prediction

python predict.py

## Future Improvements

-Streamlit web interface
-Forest change detection
-GPU training optimization

## Output image Loc:
outputs/predictions/ :
-forest_overlay.png
-probability_map.png
-mask.png


## Installation

```bash
pip install -r requirements.txt -

