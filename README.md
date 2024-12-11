

# Alzheimer's Disease Classification Using ADNI MRI Dataset

This repository focuses on the classification of Alzheimer's Disease, Mild Cognitive Impairment (MCI), and Cognitive Normal (CN) conditions using MRI scans from the ADNI dataset. The project uses a convolutional neural network (CNN) to classify MRI slices effectively.

## Project Overview

The repository presents a pipeline that:
1. Preprocesses 3D MRI images by extracting the middle slice.
2. Normalizes and scales the extracted slices.
3. Trains a SimpleCNN model on the processed data.
4. Evaluates the model on unseen data.

## Dataset Preparation

### Input Data Format

Organize your `.nii` files in a folder named `nii_files`, following this naming convention:

```
<patient_id>_<class_label>.nii
```

Where `<class_label>` is:
- `CN` for Cognitive Normal
- `AD` for Alzheimer's Disease
- `MCI` for Mild Cognitive Impairment

### Preprocessing Workflow

1. **Normalization**: Scale the MRI data to `[0, 1]`.
2. **Middle Slice Extraction**: Extract the central slice of the MRI scan along the axial plane.
3. **Scaling**: Convert the slice to a grayscale image and rescale it to 0–255.

### Example Dataset Structure

```
nii_files/
    patient1_CN.nii
    patient2_AD.nii
    patient3_MCI.nii
    ...
```

## Code Structure

### Key Functions and Classes

- **`load_mri_data(nii_folder)`**  
  Preprocesses `.nii` files and maps class labels to numeric values.
  
- **`MRIDataset`**  
  A custom PyTorch Dataset class to handle MRI data with transformations (resizing, normalization, tensor conversion).

- **`SimpleCNN`**  
  A simple convolutional neural network architecture for classification.

### Model Architecture
- **Convolutional Layers**: 2 layers with ReLU activation and max pooling.
- **Fully Connected Layers**: 2 layers for classification into CN, AD, and MCI.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ADNI-MRI-Classification.git
   cd ADNI-MRI-Classification
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Place `.nii` files in the `nii_files` folder.
2. Run the training script:
   ```bash
   python AD_classfication.py
   ```
3. Monitor the training and validation metrics in the terminal.

## Results

After training, the test accuracy is displayed. Example metrics logged during training include loss and accuracy for both training and validation datasets.

### Visualization Example

```python
# Visualize Preprocessed Slices
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(data[i], cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.show()
```

## Future Work

- Implement 3D CNNs (`nn.Conv3d`) for full volumetric analysis.
- Explore advanced architectures like ResNet and EfficientNet.
- Use data augmentation to improve model generalization.


## Contact

For questions or collaborations, please contact:

**Abdul Ghafoor**  
**aghafoor.faculty@aror.edu.pk**  
