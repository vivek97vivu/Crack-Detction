import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class GateClassifier(nn.Module):
    """
    MobileNetV3-Small binary classifier to gate frames.
    Filters out frames that do not contain cracks.
    """
    def __init__(self, pretrained=False):
        super(GateClassifier, self).__init__()
        # Load backbone
        if pretrained:
            self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)
            
        # Replace classifier head for binary output (1 logit)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, 1)
        
    def forward(self, x):
        return self.backbone(x)

class GateInference:
    def __init__(self, checkpoint_path=None, threshold=0.4, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        self.model = GateClassifier(pretrained=(checkpoint_path is None))
        if checkpoint_path:
            print(f"Loading gate classifier checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # Support both raw state dict and dict wrapped in 'model' or 'state_dict'
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image):
        """
        Runs binary gate classification on an image.
        Returns:
            bool: True if crack_present is predicted above threshold, False otherwise.
            float: The predicted probability.
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Check shape: if (H, W, C)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL Image or NumPy array")
            
        # Preprocess
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).item()
            
        passed = prob >= self.threshold
        return passed, prob
