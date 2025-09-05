# Stanford Dogs Classification: Raw vs Cropped Images 🐕‍🦺

## Overview

This project tackles **fine-grained dog breed classification** using the Stanford Dogs dataset. It demonstrates an **end-to-end pipeline**, from data preprocessing to model evaluation.

Key points:

- **Data Loading & Inspection** – train, validation, and test sets
- **Modular PyTorch Training Pipeline** – AMP, learning rate scheduler, imbalance handling
- **Comparison: Raw vs Cropped Images** – two models trained:
  - **Raw images** (full image)
  - **Cropped images** (bounding boxes around dogs)
- **Evaluation Metrics** – accuracy, precision, recall, top-10 class analysis, confusion matrices

---

## Dataset Preparation

- Dataset: [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)  
- Train/validation/test split: **70% train, 15% validation, 15% test**  
- Cropped dataset created using bounding boxes from annotations  

```python
# Split dataset
split_data(IMAGES_PATH, OUTPUT_PATH)

# Crop images using annotations
cropped_img = img.crop((xmin, ymin, xmax, ymax))
cropped_img.save(os.path.join(out_cls_dir, img_file))
```
**Model Training**
- **Architecture**: ResNet-18 pre-trained  
- **Loss**: CrossEntropyLoss  
- **Optimizer**: Adam + scheduler  
- **AMP**: Mixed precision training enabled  
- **Class** Imbalance: Weighted sampler  
- **Epochs**: 10  
- **Batch** size: 32

```python
trained_model, history = train_model(
    model.to(device),
    dataloaders,
    criterion,
    optimizer,
    num_epochs=EPOCHS,
    device=device
)
```

| Model   | Test Accuracy (Local / Kaggle) | Notable High-Recall Classes | Notable High-Precision Classes |
|---------|-------------------------------|-----------------------------|--------------------------------|
| **Raw**     | 43.9% / 14.9%         | Bernese mountain dog (0.85) <br> Gordon setter (0.83) | Australian terrier (1.00, recall 0.07) <br> Golden retriever (1.00, recall 0.22) |
| **Cropped** | **56.5% / 34.3%**     | Old English sheepdog (0.92) <br> Weimaraner (0.92) <br> Clumber (0.91) | Bloodhound (1.00, recall 0.45) <br> Chow (0.96, recall 0.83) <br> Saint Bernard (0.94, recall 0.65) |

Cropping images improves both **accuracy** and **recall**, showing the importance of focusing on the object of interest.

## 🚀 Reproducibility

**1. Local environment**  
-Clone this repository:
   ```bash
   git clone https://github.com/MVasquez95/Dog-Breed-Classification.git
   cd Dog-Breed.Classification
   ```
-Run the Jupyter notebook:
```bash
    jupyter notebook notebook.ipynb
```
**2. Kaggle Notebook** 
You can run the full pipeline directly on Kaggle without local setup:
[View on Kaggle](https://www.kaggle.com/code/crowwick/dog-breed-classification-with-deep-learning?scriptVersionId=260170594).
⚠️ **Note:** Due to slight differences in package versions between local (`requirements.txt`) and Kaggle environments, the validation score may vary marginally (e.g., **Raw	43.9% / 14.9% || Cropped	56.5% / 34.3%**). 
This variation is expected and does not affect the overall conclusions.