import rfdetr
import numpy as np

class DetectorInference:
    """
    Wrapper for RF-DETR object detector that handles model loading,
    inference, and mapping of the 4 trained classes down to 3 clean target classes.
    """
    def __init__(self, checkpoint_path, threshold=0.3):
        print(f"Loading RF-DETR model from checkpoint: {checkpoint_path}")
        self.model = rfdetr.from_checkpoint(checkpoint_path)
        self.threshold = threshold
        
        # Original class_names: ['crack', 'crack', 'rebar', 'spall']
        # We map them to clean 3 classes:
        # Index 0 & 1 -> Class 0 ('crack')
        # Index 2     -> Class 1 ('rebar')
        # Index 3     -> Class 2 ('spall')
        self.target_classes = ["crack", "rebar", "spall"]
        
    def predict(self, image):
        """
        Runs object detection on the image.
        Returns:
            list[dict]: A list of detections with mapped class IDs and labels.
        """
        # Run prediction via rfdetr
        # This returns a supervision.Detections object
        detections = self.model.predict(image, threshold=self.threshold)
        
        results = []
        if len(detections) == 0:
            return results
            
        xyxy = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id
        masks = detections.mask if (hasattr(detections, "mask") and detections.mask is not None) else None
        
        for i in range(len(xyxy)):
            orig_cid = class_ids[i]
            box = xyxy[i].tolist()
            conf = float(confidences[i])
            
            # Map original 4 classes to target 3 classes
            if orig_cid in (0, 1):
                mapped_cid = 0
            elif orig_cid == 2:
                mapped_cid = 1
            elif orig_cid == 3:
                mapped_cid = 2
            else:
                continue # ignore any invalid classes
                
            det_dict = {
                "box": box, # [x1, y1, x2, y2]
                "confidence": conf,
                "class_id": mapped_cid,
                "class_name": self.target_classes[mapped_cid]
            }
            if masks is not None:
                det_dict["mask"] = masks[i]
                
            results.append(det_dict)
            
        return results
