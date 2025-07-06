import torch
# from util import EnhancedDeepFill, device
import cv2
import numpy as np
from mask import process_user_image  

def load_model(model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location=device)
    model = EnhancedDeepFill().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

import cv2
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_mask(image_path, mask_size=(256, 256)):
    """Generate a mask for the input image"""
    # Create a single-channel mask with center region to inpaint
    mask = process_user_image(image_path)
    mask = np.array(mask)  # Convert PIL image to numpy array
    # mask = cv2.resize(mask, mask_size, interpolation=cv2.INTER_NEAREST)
    return mask

def process_image( image_path, output_size=(256, 256)):
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, output_size)
    
    # Generate mask
    mask = generate_mask(image_path, output_size)
    
    # # Convert to tensors with proper dimensions
    # img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)  # Shape: [H,W,3]
    # img_tensor = img_tensor.permute(2, 0, 1)                       # Shape: [3,H,W]
    
    # mask_tensor = torch.from_numpy(mask).squeeze(-1)                # Shape: [H,W]
    # mask_tensor = mask_tensor.unsqueeze(0)                          # Shape: [1,H,W]
    
    # # Combine RGB + mask
    # input_tensor = torch.cat([
    #     img_tensor,    # [3,H,W]
    #     mask_tensor    # [1,H,W]
    # ], dim=0).unsqueeze(0).to(device)  # Final shape: [1,4,H,W]
    
    # # Process through model
    # with torch.no_grad():
    #     output = model(input_tensor)
    
    # # Convert output to image
    # output = output.squeeze(0)        # [3,H,W]
    # output = output.permute(1, 2, 0)  # [H,W,3]
    # output = output.clamp(0, 1).cpu().numpy()
    # output = (output * 255).astype(np.uint8)
    
    return mask