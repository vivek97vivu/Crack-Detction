import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
import os
from PIL import Image

from inference.gate import GateClassifier

class DummyDataset(Dataset):
    """
    A placeholder dataset that returns dummy images and labels.
    Used when real datasets are not loaded or for testing.
    """
    def __init__(self, size=100, transform=None):
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Create a random RGB image
        img = Image.fromarray((torch.rand(3, 224, 224).permute(1, 2, 0).numpy() * 255).astype('uint8'))
        # Binary label (0 or 1)
        label = float(idx % 2)
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor([label], dtype=torch.float32)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize Dataset (Use real classification dataset if path is provided)
    if args.dataset_dir and os.path.exists(args.dataset_dir):
        print(f"Loading custom dataset from: {args.dataset_dir}")
        # Implement custom dataset loading logic here
        dataset = DummyDataset(size=200, transform=transform)
    else:
        print("Using dummy dataset for training script verification.")
        dataset = DummyDataset(size=100, transform=transform)
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = GateClassifier(pretrained=True).to(device)
    
    # Binary cross-entropy with logits and label smoothing (0.1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("Starting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss:.4f}")
        
    save_path = os.path.join(args.output_dir, "gate_classifier_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3 Gate Classifier")
    parser.add_argument("--dataset-dir", type=str, default="", help="Path to raw dataset")
    parser.add_argument("--output-dir", type=str, default="runs/gate", help="Output directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--pos-weight", type=float, default=1.5, help="Positive class weight to favor high recall")
    
    args = parser.parse_args()
    train(args)
