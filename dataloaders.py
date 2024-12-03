import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import ImageOps

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)
        padding = (max_dim - w) // 2, (max_dim - h) // 2
        return ImageOps.expand(image, border=padding, fill=0)  # Fill with black
    
class XRayDataset(Dataset):
    def __init__(self, front_images, side_images, scores, mode='train'):
        """
        Args:
            front_images (list): Paths to frontal X-ray images
            side_images (list): Paths to side X-ray images
            scores (list): Target scores
            mode (str): 'train', 'val', or 'test' - determines whether to use augmentations
        """
        self.front_images = front_images
        self.side_images = side_images
        self.scores = scores
        self.mode = mode
        
        # Base transforms that are always applied
        self.base_transform = transforms.Compose([
            SquarePad(),  # Add black padding to make the image square
            transforms.Resize((224, 224),interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentations only for training
        self.train_transform = transforms.Compose([
            # Geometric augmentations
            transforms.RandomAffine(
                degrees=5,  # Slight rotation
                translate=(0.05, 0.05),  # Small translations
                scale=(0.95, 1.05),  # Minor scaling
                fill=0  # Fill empty areas with black
            ),
            
            # Intensity augmentations
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,  # Brightness variation
                    contrast=0.2,    # Contrast variation
                )
            ], p=0.5),
            
            # Random horizontal flip for frontal images only
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Slight random rotation
            transforms.RandomRotation(
                degrees=3,
                fill=0
            ),
            
            # Random crop and resize
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05)
            )
        ])
        
    def __len__(self):
        return len(self.scores)
    
    def apply_random_gamma(self, img, gamma_range=(0.8, 1.2)):
        """Apply random gamma correction"""
        if random.random() > 0.5:
            gamma = random.uniform(gamma_range[0], gamma_range[1])
            img = transforms.functional.adjust_gamma(img, gamma)
        return img
    
    def add_random_noise(self, img, noise_factor=0.02):
        """Add random Gaussian noise"""
        if random.random() > 0.5:
            img_array = np.array(img)
            noise = np.random.normal(loc=0, scale=noise_factor, size=img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 1)
            img = Image.fromarray((noisy_img * 255).astype(np.uint8))
        return img
    
    def __getitem__(self, idx):
        # Load images
        front_img = Image.open(self.front_images[idx]).convert('RGB')
        side_img = Image.open(self.side_images[idx]).convert('RGB')
        
        # Apply transformations based on mode
        if self.mode == 'train':
            # Apply base geometric augmentations
            if random.random() > 0.5:
                front_img = self.train_transform(front_img)
                side_img = self.train_transform(side_img)
            # Convert to tensor first for intensity augmentations
            front_img = self.base_transform(front_img)
            side_img = self.base_transform(side_img)
            
            # Apply random intensity augmentations
            if random.random() > 0.5:
# #                 Random gamma correction
#                 gamma = random.uniform(0.8, 1.2)
#                 front_img = transforms.functional.adjust_gamma(front_img, gamma)
#                 side_img = transforms.functional.adjust_gamma(side_img, gamma)
                
                # Random brightness adjustment
                brightness_factor = random.uniform(0.9, 1.1)
                front_img = transforms.functional.adjust_brightness(front_img, brightness_factor)
                side_img = transforms.functional.adjust_brightness(side_img, brightness_factor)
                
                # Random contrast adjustment
                contrast_factor = random.uniform(0.9, 1.1)
                front_img = transforms.functional.adjust_contrast(front_img, contrast_factor)
                side_img = transforms.functional.adjust_contrast(side_img, contrast_factor)
        else:
            # Val/Test mode - only apply base transform
            front_img = self.base_transform(front_img)
            side_img = self.base_transform(side_img)
        
        # Convert score to tensor
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        
        return front_img, side_img, score

def visualize_augmentations(dataset, num_samples=5):
    """
    Visualize augmentations from the dataset.

    Args:
        dataset (Dataset): Instance of the XRayDataset.
        num_samples (int): Number of samples to visualize.
    """
    for i in range(num_samples):
        # Pick a random index
        idx = random.randint(0, len(dataset) - 1)

        # Get the data
        front_img, side_img, score = dataset[idx]
        
        # Convert tensors to PIL images for visualization
        front_img_pil = F.to_pil_image(front_img)
        side_img_pil = F.to_pil_image(side_img)
        
        # Plot the images
        plt.figure(figsize=(10, 5),dpi=150)
        plt.subplot(1, 2, 1)
        plt.imshow(front_img_pil)
        plt.title("Frontal X-Ray")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(side_img_pil)
        plt.title("Side X-Ray")
        plt.axis('off')

        plt.suptitle(f"Target Score: {score.item():.2f}", fontsize=14)
        plt.savefig(f'augment_{i}.jpg')
        
        
def prepare_data(data_dir='JPG', score_file='final_scores.npy', batch_size=16):
    """
    Prepare data loaders for training, validation, and testing
    
    Args:
        data_dir (str): Directory containing X-ray images
        score_file (str): Path to CSV file with patient IDs and scores
        batch_size (int): Batch size for data loaders
    """
    # Read scores file
    
    # Assuming directory structure: data_dir/patient_id/[front.png, side.png]
    front_images = []
    side_images = []
    scores = np.load(score_file)
    #print(scores)
    for img in sorted(os.listdir(data_dir)):
        im=img.split('.')[2]
        if  im== 'ap':
            front_images.append(os.path.join(data_dir,img))
        elif im=='lat1':
            side_images.append(os.path.join(data_dir,img))
#     print(len(front_images))
#     print(len(side_images))
    
    # Split data
    train_front, temp_front, train_side, temp_side, train_scores, temp_scores = train_test_split(
        front_images, side_images, scores, test_size=0.2, random_state=42
    )
    
    val_front, test_front, val_side, test_side, val_scores, test_scores = train_test_split(
        temp_front, temp_side, temp_scores, test_size=0.5, random_state=42
    )
    
    # Create datasets with appropriate modes
    train_dataset = XRayDataset(train_front, train_side, train_scores, mode='train')
    val_dataset = XRayDataset(val_front, val_side, val_scores, mode='val')
    test_dataset = XRayDataset(test_front, test_side, test_scores, mode='test')
#     print(len(train_dataset))
#     print(len(val_dataset))
#     print(len(test_dataset))
    #Visualize the dataset
#     visualize_augmentations(test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__=='__main__':
    train_loader, val_loader, test_loader= prepare_data()
    for _,_,score in test_loader:
        print(score)
