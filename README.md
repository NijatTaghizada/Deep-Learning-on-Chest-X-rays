# Chest X-Ray Classification with DenseNet121

A deep learning project that uses a **convolutional neural network (CNN)** to classify chest X-rays as **healthy (`No Finding`)** or **unhealthy (one or more findings)**. The pipeline fine-tunes an ImageNet-pretrained DenseNet121, selects the best checkpoint using validation ROC-AUC, and evaluates it on 10,000 test images.

This README covers the CNN pipeline in `FinalCNN.ipynb`.

## Project Overview

The project asks: **Can a CNN distinguish chest X-rays labeled with no finding from those labeled with one or more pathological findings using the image alone?**

The model performs binary classification rather than identifying individual diseases. “Healthy” and “unhealthy” are shorthand for the dataset label groups; they do not establish a patient's clinical health status. Metadata supplies training labels but is not used as a model input.

## Dataset and Labels

The project uses images and metadata from the **NIH Chest X-ray Dataset**.

Dataset source: <https://www.kaggle.com/datasets/nih-chest-xrays/data>

The notebook matches image filenames to `Data_Entry_2017.csv` using these fields:

| Field | Purpose |
|---|---|
| `Image Index` | Image filename |
| `Finding Labels` | Original finding labels |
| `has_disease` | Derived binary target |

Labels are constructed as follows:

```python
df["Finding Labels"] = df["Finding Labels"].str.replace("No Finding", "")
df["has_disease"] = (df["Finding Labels"] != "").astype(int)
```

| Target | Meaning |
|---|---|
| `0` | Originally labeled `No Finding` |
| `1` | Contains one or more finding labels |

The notebook records **92,120 images** collected into `images_final/`. Metadata rows are filtered to images present in that folder, then split into training and validation sets using a **90/10 stratified split** with `random_state=42`. Test evaluation uses a separately prepared folder containing **10,000 matched images**.

```python
train_df, val_df = train_test_split(
    df_final,
    test_size=0.1,
    stratify=df_final["has_disease"],
    random_state=42,
)
```

## Image Preprocessing

Images are opened with Pillow and converted to grayscale. After transformation, the single normalized channel is repeated three times to provide the three-channel input expected by DenseNet121.

### Training

Training uses random augmentation:

- Horizontal flipping.
- Rotation up to ±10°.
- Random resized cropping to **224 × 224**, with `scale=(0.8, 1.0)`.
- Tensor conversion and normalization with mean `0.485` and standard deviation `0.229`.

```python
T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ToTensor(),
    T.Normalize([0.485], [0.229]),
])
```

### Validation and Testing

Validation and test images use deterministic preprocessing:

```python
T.Compose([
    T.Resize(240),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485], [0.229]),
])
```

The dataset classes perform channel replication after these transforms:

```python
img = Image.open(image_path).convert("L")
img = transform(img).repeat(3, 1, 1)
```

## Model Architecture

DenseNet121 is initialized with ImageNet-pretrained weights. Its original classifier is replaced by an identity layer, followed by dropout and a single linear output:

| Component | Configuration |
|---|---|
| Backbone | DenseNet121 with ImageNet weights |
| Feature vector | 1,024 features |
| Dropout | `0.5` |
| Classification layer | `Linear(1024, 1)` |
| Model output | One raw logit per image |
| Inference probability | Sigmoid applied to the logit |

```python
backbone = models.densenet121(pretrained=True)
feat_dim = backbone.classifier.in_features
backbone.classifier = nn.Identity()

model = nn.Sequential(
    backbone,
    nn.Dropout(0.5),
    nn.Linear(feat_dim, 1),
).to(DEVICE)
```

All model parameters are passed to the optimizer, so the backbone is fine-tuned alongside the classification head. Transfer learning supplies an initial set of visual features rather than requiring the network to learn every feature from scratch.

Sigmoid is applied during validation and inference; it is not part of the model definition because `BCEWithLogitsLoss` accepts raw logits.

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Input size | `224 × 224` |
| Training / validation batch size | `16` |
| Test batch size | `32` |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Dropout | `0.5` |
| Loss | Weighted `BCEWithLogitsLoss` |
| Epochs | `7` |
| Scheduler | `CosineAnnealingLR`, `T_max=7` |
| Checkpoint selection | Highest validation ROC-AUC |
| DataLoader workers | `2` |

The project report describes hyperparameter tuning on a 10,000-image subset with stratified cross-validation. The search included learning rates `{1e-3, 3e-4, 1e-4}`, dropout `{0.3, 0.5}`, weight decay `{1e-5, 1e-4}`, and batch sizes `{8, 16}`. The final notebook implements the selected configuration above; it does not include the tuning procedure.

### Class Imbalance

The loss weights positive examples using the ratio of negative to positive examples in the training split:

```python
counts = train_df["has_disease"].value_counts().to_dict()
pos_weight = torch.tensor(counts[0] / counts[1]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This adjusts the positive-class contribution to the loss according to the training class distribution. The weight can increase or decrease that contribution depending on the ratio.

### Device and Mixed Precision

The notebook selects CUDA when available and otherwise selects the CPU:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Training uses `torch.cuda.amp.autocast` and `GradScaler` for automatic mixed precision on CUDA. A CUDA-capable GPU is recommended for this workload.

### Best-Model Checkpoint

After each epoch, the model is evaluated on the validation split. A checkpoint is saved whenever validation ROC-AUC improves:

```python
if val_auc > best_auc:
    best_auc = val_auc
    torch.save(model.state_dict(), "final_best_model.pth")
```

`final_best_model.pth` therefore stores the parameters from the best validation epoch, which may differ from the final epoch.

## Evaluation and Results

The final evaluation cell rebuilds the architecture, loads `final_best_model.pth`, switches to evaluation mode, and runs inference without gradient tracking.

```python
probs = torch.sigmoid(out)
preds = (probs >= 0.5).astype(int)
```

ROC-AUC uses predicted probabilities. Accuracy, precision, recall, F1, and the confusion matrix use a **0.5 classification threshold**.

The following results are recorded in the notebook's final evaluation output and summarized in the project report. They have not been independently reproduced here.

| Metric | Score |
|---|---:|
| Accuracy | **0.7333** |
| ROC-AUC | **0.8004** |
| Precision — unhealthy class | **0.7046** |
| Recall — unhealthy class | **0.7289** |
| F1 — unhealthy class | **0.7165** |

### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy (`0`) | 0.7596 | 0.7371 | 0.7482 | 5,375 |
| Unhealthy (`1`) | 0.7046 | 0.7289 | 0.7165 | 4,625 |
| Macro average | 0.7321 | 0.7330 | 0.7324 | 10,000 |
| Weighted average | 0.7342 | 0.7333 | 0.7336 | 10,000 |

### Confusion Matrix

Rows represent actual labels; columns represent predictions.

| | Predicted Healthy | Predicted Unhealthy |
|---|---:|---:|
| **Actual Healthy** | 3,962 | 1,413 |
| **Actual Unhealthy** | 1,254 | 3,371 |

The model correctly classified **7,333 of 10,000 images**. Unhealthy-class recall was approximately **72.9%**, with **1,254 false negatives**. These errors are an important limitation of the classifier.

## Project Structure

The notebook expects the following working-directory layout. Images, metadata, and trained weights must be supplied or generated separately.

```text
project/
├── README.md
├── FinalCNN.ipynb
├── Data_Entry_2017.csv
├── images_final/
├── test_images/
│   └── images1/
└── final_best_model.pth
```

| Item | Purpose |
|---|---|
| `FinalCNN.ipynb` | Data preparation, training, validation, checkpointing, and evaluation |
| `Data_Entry_2017.csv` | Image filenames and labels |
| `images_final/` | Training and validation image pool |
| `test_images/images1/` | Images used by the final test evaluation cell |
| `final_best_model.pth` | Model state dictionary generated during training |

## Requirements

The CNN uses Python with PyTorch, torchvision, scikit-learn, pandas, NumPy, Pillow, and tqdm. JupyterLab can be used to run the notebook locally.

```bash
pip install torch torchvision scikit-learn pandas numpy pillow tqdm jupyterlab
```

The notebook's installation cells also install `torchaudio`, `timm`, and `gdown`; `gdown` is used by its Google Drive image-download cell. Its first installation cell requests CUDA 11.8 PyTorch wheels. Adjust or skip that installation cell to suit your own environment.

No exact dependency versions are pinned in the supplied notebook.

## Running the Project

### 1. Clone the Repository

Replace the placeholders with your repository details:

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies

Install the packages listed above in your Python environment.

### 3. Prepare the Metadata and Images

Place `Data_Entry_2017.csv` in the project directory. Put training and validation images directly in `images_final/`, and test images directly in `test_images/images1/`. Image filenames must match the CSV's `Image Index` values.

The notebook includes Dropbox and Google Drive archive-download cells. If those shared archives are unavailable, prepare the image folders yourself from the dataset source. Keep test images separate from the training and validation pool.

### 4. Open the Notebook

```bash
jupyter lab FinalCNN.ipynb
```

Google Colab or another compatible notebook environment can also be used, provided the expected data paths and dependencies are available.

### 5. Train the CNN

Run device setup and data preparation, then the cell titled **“Final CNN training & evaluation using best hyperparameters.”** Training runs for seven epochs and writes `final_best_model.pth` to the working directory.

### 6. Evaluate the Saved Model

Prepare the test images before running the **final evaluation cell**, which uses:

```python
TEST_DIR = Path("test_images") / "images1"
```

The notebook also contains an earlier evaluation cell pointing to `test_images_final/`. Skip that duplicate cell unless you have deliberately prepared that alternative directory. The test-download cell appears after the earlier evaluation cell, so running every cell unchanged from top to bottom can fail before test preparation is complete.

The final evaluation prints the classification report, confusion matrix, accuracy, ROC-AUC, precision, recall, and F1 score.

## Reproducibility Notes

- The train/validation split fixes `random_state=42`, but the notebook does not set global seeds for model initialization, augmentation, or DataLoader shuffling. Retraining results can vary.
- The split is stratified by image label, not grouped by patient. Patient separation across training, validation, and test sets is not established by the supplied code.
- The notebook does not explicitly verify overlap between training and test filenames. Check image and patient separation before interpreting performance as generalization to unseen patients.
- The supplied notebook produces deprecation warnings for `pretrained=` and `torch.cuda.amp`. Its code uses those original APIs; adapting it to another environment may require updates.

## Limitations

- **Binary output:** The model predicts the presence of any finding, not a specific diagnosis or the location of an abnormality.
- **Label interpretation:** `No Finding` is a dataset label and does not guarantee the absence of all disease.
- **False negatives:** The recorded test output includes 1,254 finding-positive images classified as healthy.
- **Dataset scope:** Evaluation uses images from the same underlying NIH dataset. External validation is needed to assess performance across other institutions and populations.
- **Split verification:** Patient overlap and training/test image overlap are not ruled out by the notebook's split logic.
- **Threshold choice:** A fixed threshold of 0.5 is used without a documented threshold-selection or probability-calibration procedure.

## Future Improvements

- Split and audit data at the patient level.
- Validate on independent external datasets.
- Tune the decision threshold using validation data.
- Improve unhealthy-class recall while measuring the effect on false positives.
- Extend the task to multilabel prediction of individual findings.
- Investigate Grad-CAM or other model explanation techniques.
- Evaluate probability calibration and performance across patient subgroups.
- Pin dependencies and add complete random-seed control for reproducibility.

## Intended Use

This project is intended for education and machine learning research. Its reported results do not establish clinical readiness. It should not be used to make medical decisions or replace professional interpretation of chest X-rays.

## Authors

- Nijat Taghizada
- Felix Pang
- Antonio Laloni

## Acknowledgments and References

- **NIH Chest X-ray Dataset:** <https://www.kaggle.com/datasets/nih-chest-xrays/data>
- **DenseNet:** Huang, Gao, et al. *Densely Connected Convolutional Networks*. CVPR 2017. <https://arxiv.org/abs/1608.06993>
- **Project sources:** `FinalCNN.ipynb` and `Final Project Report.docx`. The implementation and recorded evaluation output are drawn from the notebook; the hyperparameter-search description and authorship are drawn from the report.
