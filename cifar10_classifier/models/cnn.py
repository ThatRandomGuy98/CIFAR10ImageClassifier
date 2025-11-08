import torch
import torch.nn as nn
import torch.nn.functional as F

    
    
class CNN(nn.Module):
    """
    A simple Convolutional Neural Network for CIFAR-10 classification.
    Input: 3x32x32 images
    Output: 10 class logits
    """
    def __init__(self, dropout=0.25):
        super(CNN, self).__init__()
        
        # --- Convolutional layers ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # --- Batch Normalization layers + Pooling layer ---
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Fully Connected layers ---
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)  # 10 CIFAR-10 classes
        
        # --- Regularization ---
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # Output: [B, 32, 16, 16]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # Output: [B, 64, 8, 8]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # Output: [B, 128, 4, 4]

        # --- Flatten for fully connected layers ---
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x