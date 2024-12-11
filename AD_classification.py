import os
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
# Step 1: Load and Preprocess MRI Data
def load_mri_data(nii_folder):
    data = []
    labels = []
    label_map = {"CN": 0, "AD": 1, "MCI": 2}  # Map classes to numeric labels

    for filename in os.listdir(nii_folder):
        if filename.endswith('.nii'):
            filepath = os.path.join(nii_folder, filename)
            
            # Load the .nii file
            nii_data = nib.load(filepath).get_fdata()
            nii_data = (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data))  # Normalize
            
            middle_slice = nii_data[:, :, nii_data.shape[2] // 2]  # Extract middle slice
            slice_image = Image.fromarray((middle_slice * 255).astype(np.uint8))  # Scale to 0-255
            
            label_str = filename.split('_')[-1].split('.')[0]
            label = label_map[label_str]  # Convert label to numeric
            
            data.append(slice_image)
            labels.append(label)

    return data, labels




# Step 2: Custom Dataset Class
class MRIDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Step 3: Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust for 224x224 input
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)  # Three output classes: CN, AD, MCI

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x



# Step 4: Load Data
nii_folder = 'nii_files'
data, labels = load_mri_data(nii_folder)

# Visualize a Few Slices
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(data[i], cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.show()


# save sample MRI Slices



# Define the output directory and file name
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
output_file = output_dir / "enhanced_combined_image_with_labels.jpg"

# Create a single image combining the 5 slices with labels
images = [data[i] for i in range(5)]  # Assuming 'data' contains the images
labels_text = [f"Label: {labels[i]}" for i in range(5)]  # Assuming 'labels' contains the label data

# Define dimensions
width, height = images[0].size  # Assuming all images have the same dimensions
padding = 20  # Space between images
label_height = 40  # Space for labels
combined_width = width * len(images) + padding * (len(images) - 1)
combined_height = height + label_height + padding
background_color = (245, 245, 245)  # Light grey background
text_color = (0, 0, 0)  # Black text

# Create the combined image
combined_image = Image.new("RGB", (combined_width, combined_height), background_color)

# Font setup for labels
try:
    font = ImageFont.truetype("arial.ttf", 16)
except IOError:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(combined_image)

# Paste each image with padding and add labels
for idx, img in enumerate(images):
    x_offset = idx * (width + padding)
    y_offset = padding
    combined_image.paste(img.convert("RGB"), (x_offset, y_offset))  # Convert to RGB
    label_text = labels_text[idx]
    # Calculate the text size using textbbox
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    label_x = x_offset + (width - text_width) // 2
    label_y = y_offset + height + 5
    draw.text((label_x, label_y), label_text, fill=text_color, font=font)

# Draw a border around the slices
border_color = (0, 0, 0)  # Black border
for idx in range(len(images)):
    x_offset = idx * (width + padding)
    y_offset = padding
    draw.rectangle(
        [x_offset, y_offset, x_offset + width, y_offset + height],
        outline=border_color,
        width=2
    )

# Save the combined image
combined_image.save(output_file)

print(f"Enhanced combined image with labels saved as {output_file}")







transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = MRIDataset(data, labels, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 5: Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the Model
def train_model(model, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss, val_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_correct / len(train_loader.dataset):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Acc: {val_correct / len(val_loader.dataset):.4f}")

train_model(model, train_loader, val_loader, num_epochs=10)

# Step 7: Evaluate on Test Data
def evaluate_model(model, test_loader):
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()

    print(f"Test Accuracy: {test_correct / len(test_loader.dataset):.4f}")

evaluate_model(model, test_loader)
