import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from src.utils.geometry import extract_geometry

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        u1 = self.up1(x4)
        if u1.shape != x3.shape:
            u1 = F.interpolate(u1, size=x3.shape[2:])
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        if u2.shape != x2.shape:
            u2 = F.interpolate(u2, size=x2.shape[2:])
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        if u3.shape != x1.shape:
            u3 = F.interpolate(u3, size=x1.shape[2:])
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv_up3(u3)
        
        return self.outc(u3)

class SegmenterInference:
    def __init__(self, checkpoint_path=None, device=None, fallback_to_heuristic=True):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=3, out_channels=1)
        self.fallback_to_heuristic = fallback_to_heuristic
        self.is_trained = False
        
        if checkpoint_path:
            print(f"Loading segmenter checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict)
            self.is_trained = True
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
    def predict(self, crop):
        """
        Runs segmentation on an image crop.
        Returns:
            np.ndarray: Binary mask of shape (H, W) with values 0 or 255.
        """
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros((256, 256), dtype=np.uint8)
            
        orig_h, orig_w = crop.shape[:2]
        
        if self.fallback_to_heuristic and not self.is_trained:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            return opened
            
        pil_crop = Image.fromarray(crop)
        x = self.transform(pil_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).squeeze(0).squeeze(0)
            mask_256 = (prob >= 0.5).cpu().numpy().astype(np.uint8) * 255
            
        mask_orig = cv2.resize(mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask_orig

    def process_crop(self, crop, pixel_per_mm, min_length_px, min_area_px, sample_interval, severity_mapper):
        """
        Segments a crop and processes geometry + severity.
        Returns:
            (mask, geom_list, worst_sev)
        """
        mask = self.predict(crop)
        geom_list, worst_sev = self.process_mask(
            mask, pixel_per_mm, min_length_px, min_area_px, sample_interval, severity_mapper
        )
        return mask, geom_list, worst_sev

    def process_mask(self, mask, pixel_per_mm, min_length_px, min_area_px, sample_interval, severity_mapper):
        """
        Processes geometry and severity classification from a binary mask.
        Returns:
            (geom_list, worst_sev)
        """
        geom_list = extract_geometry(
            mask, 
            pixel_per_mm=pixel_per_mm,
            min_length_px=min_length_px,
            min_area_px=min_area_px,
            sample_interval=sample_interval
        )
        
        worst_sev = None
        if geom_list:
            severity_results = [
                severity_mapper.classify(g.width_mean_mm, g.length_mm)
                for g in geom_list
            ]
            worst_sev = severity_mapper.worst_level(severity_results)
            
        return geom_list, worst_sev
