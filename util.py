# COMPLETE ENHANCED IMAGE INPAINTING SYSTEM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import shutil

# ==================== DEVICE SETUP ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== ENHANCED MODEL ARCHITECTURE ====================
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels*2, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x, gate = x.chunk(2, dim=1)
        return x * self.sigmoid(gate)

class ContextualAttention(nn.Module):
    def __init__(self, patch_size=3, stride=1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.unfold = nn.Unfold(patch_size, stride, padding=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x).view(B, C, self.patch_size, self.patch_size, -1)
        # Simplified attention - full implementation would include similarity computation
        return x

class EnhancedDeepFill(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            GatedConv2d(4, 96, 5, padding=2),
            nn.InstanceNorm2d(96),
            nn.ReLU(),
            GatedConv2d(96, 192, 3, stride=2, padding=1),
            nn.InstanceNorm2d(192),
            nn.ReLU(),
            GatedConv2d(192, 384, 3, stride=2, padding=1),
            nn.InstanceNorm2d(384),
            nn.ReLU()
        )
        
        # Attention
        self.attention = ContextualAttention()
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            GatedConv2d(384, 192, 3, padding=1),
            nn.InstanceNorm2d(192),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            GatedConv2d(192, 96, 3, padding=1),
            nn.InstanceNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        return (x + 1) / 2  # Convert [-1,1] to [0,1]

# ==================== DISCRIMINATOR ==================== 
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# ==================== LOSS FUNCTIONS ====================
def perceptual_loss(pred, target, vgg):
    pred_features = vgg(pred.expand(-1,3,-1,-1))
    target_features = vgg(target.expand(-1,3,-1,-1))
    return F.l1_loss(pred_features, target_features)

# ==================== TRAINING LOOP ====================
def train(model, discriminator, train_loader, val_loader, epochs=100):
    # Initialize
    model = model.to(device)
    discriminator = discriminator.to(device)
    vgg = vgg16(pretrained=True).features[:16].to(device).eval()
    
    # Optimizers
    optimizer_G = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Loss weights
    lambda_l1 = 10.0
    lambda_adv = 1.0
    lambda_perceptual = 0.1
    
    for epoch in range(epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Generator update
            optimizer_G.zero_grad()
            outputs = model(inputs)
            
            # Losses
            loss_l1 = F.l1_loss(outputs, targets) * lambda_l1
            loss_adv = -torch.mean(discriminator(outputs)) * lambda_adv
            loss_percep = perceptual_loss(outputs, targets, vgg) * lambda_perceptual
            loss_G = loss_l1 + loss_adv + loss_percep
            loss_G.backward()
            optimizer_G.step()
            
            # Discriminator update
            optimizer_D.zero_grad()
            real_loss = torch.mean(F.relu(1.0 - discriminator(targets)))
            fake_loss = torch.mean(F.relu(1.0 + discriminator(outputs.detach())))
            loss_D = (real_loss + fake_loss) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # Logging
            if i % 20 == 0:
                print(f"Epoch {epoch} Batch {i}: "
                      f"G_Loss={loss_G.item():.3f} "
                      f"D_Loss={loss_D.item():.3f}")
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_input, val_target = next(iter(val_loader))
                val_input, val_target = val_input.to(device), val_target.to(device)
                val_output = model(val_input)
                val_psnr = psnr(val_target.cpu().numpy(), val_output.cpu().numpy(), data_range=1.0)
                print(f"Validation PSNR: {val_psnr:.2f} dB")



def move_images():
    # Define paths
    original_folder = 'dataset'
    output_folder = 'output_images'

    # Static destination folders
    static_dir = 'static'

    # Create static folders if they don't exist
    os.makedirs(static_dir, exist_ok=True)

    # Move images from original_folder to static/original
    for filename in os.listdir(original_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and not filename.lower().endswith(('_mask.png', '_mask.jpg', '_mask.jpeg', '_mask.bmp', '_mask.tiff')):
            src_path = os.path.join(original_folder, filename)
            dst_path = os.path.join(static_dir, filename)
            shutil.copy2(src_path, dst_path)  # use copy2 to preserve metadata

    # Move images from output_folder to static/output
    for filename in os.listdir(output_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            src_path = os.path.join(output_folder, filename)
            dst_path = os.path.join(static_dir, filename)
            shutil.copy2(src_path, dst_path)

    print("Images copied to static folder.")