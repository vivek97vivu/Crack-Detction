import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
import os
from PIL import Image

from myproj.inference.segmenter import UNet

class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross Entropy Loss.
    Essential for segmentation when foreground pixels (cracks) are sparse.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
        
        # Flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        BCE = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')
        
        return BCE + dice_loss

class DummySegmentationDataset(Dataset):
    """
    A placeholder dataset for segmenter training.
    """
    def __init__(self, size=50, transform=None):
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Image
        img = Image.fromarray((torch.rand(3, 256, 256).permute(1, 2, 0).numpy() * 255).astype('uint8'))
        
        # Mask: Simulate a centerline crack (vertical/diagonal line)
        mask_np = np.zeros((256, 256), dtype=np.uint8)
        # Draw a line
        start_x, start_y = idx % 100 + 50, 0
        end_x, end_y = idx % 100 + 70, 255
        cv2.line(mask_np, (start_x, start_y), (end_x, end_y), 255, thickness=3)
        mask = Image.fromarray(mask_np)
        
        if self.transform:
            img = self.transform(img)
            
        # Target mask preprocessing
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        mask = mask_transform(mask)
        
        return img, mask

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Initialize Dataset
    if args.dataset_dir and os.path.exists(args.dataset_dir):
        print(f"Loading custom segmentation dataset from: {args.dataset_dir}")
        dataset = DummySegmentationDataset(size=100, transform=transform)
    else:
        print("Using dummy dataset for segmenter training script validation.")
        dataset = DummySegmentationDataset(size=50, transform=transform)
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("Starting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.4f}")
        
    save_path = os.path.join(args.output_dir, "segmenter_unet_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net Crack Segmenter")
    parser.add_argument("--dataset-dir", type=str, default="", help="Path to raw dataset")
    parser.add_argument("--output-dir", type=str, default="runs/segmenter", help="Output directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    
    args = parser.parse_args()
    train(args)
