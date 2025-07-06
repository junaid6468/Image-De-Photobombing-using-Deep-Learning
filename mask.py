import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from threading import Thread
import numpy as np

# Your original transform (must match training)
IMAGE_SIZE = 512
transform_image = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalizing to [-1, 1]
])

def process_user_image(image_path, model_path='deeplabv3_photobomb_removal_state_dict.pth'):
    print(f"Processing image: {image_path}")
    # 1. Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size  # (width, height)

    # Apply the SAME transform as training
    input_tensor = transform_image(original_image).unsqueeze(0).to(device)  # Add batch dim

    # 3. Run inference
    with torch.no_grad():
        output = model(input_tensor)['out']  # Get mask logits
        pred_mask = torch.sigmoid(output)    # Convert to [0, 1]
        binary_mask = (pred_mask > 0.5).float()  # Threshold at 0.5

    # 4. Post-process mask
    binary_mask = binary_mask.squeeze().cpu().numpy()  # (512, 512)

    # Resize mask to original image size (critical fix!)
    mask_img = Image.fromarray((binary_mask * 255).astype('uint8'))
    mask_img = mask_img.resize(original_size, Image.NEAREST)  # Preserve hard edges

    # 5. Apply mask to original image
    result = original_image.copy()
    result.putalpha(mask_img)  # Now sizes match!

    # # 6. Display results
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 3, 1); plt.imshow(original_image); plt.title("Original"); plt.axis("off")
    # def myPlot():
    #     plt.imshow(mask_img, cmap='gray')
    #     plt.show()
    # myPlot()
    # plt.subplot(1, 3, 3); plt.imshow(result); plt.title("Masked Result"); plt.axis("off")
    # plt.tight_layout()
    # plt.show()
    # save_path = image_path.replace('.jpg', '_mask.png')
    # print(f"Saving mask image to: {save_path}")
    # mask_img.save(save_path)  # Save mask image

    return mask_img
if __name__ == "__main__":
# Example usage
    process_user_image("bird1.jpg")