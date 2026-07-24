import numpy as np

class SimpleBBoxTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.tracked_dets = {} # track_id -> bbox_xyxy (list or np.ndarray)

    def update(self, detections):
        """
        Vectorized tracking matching using NumPy.
        Args:
            detections: list of Detection objects (with bbox_xyxy)
        Returns:
            list of Detection objects with assigned track_ids
        """
        if not detections:
            return []

        updated_detections = []
        
        # If we have no active tracks, assign new IDs to all
        if not self.tracked_dets:
            for det in detections:
                det.track_id = self.next_track_id
                self.next_track_id += 1
                self.tracked_dets[det.track_id] = det.bbox_xyxy
                updated_detections.append(det)
            return updated_detections

        # Match incoming detections with tracked detections using vectorized IOU
        matched_track_ids = set()
        new_tracked_dets = {}
        
        # Convert active tracks to numpy arrays
        track_ids = list(self.tracked_dets.keys())
        tracked_boxes = np.array([self.tracked_dets[tid] for tid in track_ids]) # shape (M, 4)

        for det in detections:
            bbox = np.array(det.bbox_xyxy)
            
            # Mask out already matched track IDs
            mask = np.array([tid not in matched_track_ids for tid in track_ids], dtype=bool)
            if not np.any(mask):
                # No unmatched tracks left: assign a new track ID
                det.track_id = self.next_track_id
                self.next_track_id += 1
                new_tracked_dets[det.track_id] = det.bbox_xyxy
                updated_detections.append(det)
                continue

            filtered_track_ids = [track_ids[i] for i in range(len(track_ids)) if mask[i]]
            filtered_boxes = tracked_boxes[mask]

            # Vectorized IoU calculation
            ious = self._compute_iou_batch(bbox, filtered_boxes)
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]
            
            if best_iou >= self.iou_threshold:
                best_track_id = filtered_track_ids[best_idx]
                det.track_id = best_track_id
                matched_track_ids.add(best_track_id)
                new_tracked_dets[best_track_id] = det.bbox_xyxy
            else:
                det.track_id = self.next_track_id
                self.next_track_id += 1
                new_tracked_dets[det.track_id] = det.bbox_xyxy
                
            updated_detections.append(det)
            
        self.tracked_dets = new_tracked_dets
        return updated_detections

    def _compute_iou_batch(self, boxA, boxesB):
        """
        Compute IoU between one box A and a batch of boxes B (vectorized).
        boxA: shape (4,)
        boxesB: shape (M, 4)
        """
        # Determine the intersection coordinates
        xAs = np.maximum(boxA[0], boxesB[:, 0])
        yAs = np.maximum(boxA[1], boxesB[:, 1])
        xBs = np.minimum(boxA[2], boxesB[:, 2])
        yBs = np.minimum(boxA[3], boxesB[:, 3])

        # Compute area of intersection
        interAreas = np.maximum(0.0, xBs - xAs) * np.maximum(0.0, yBs - yAs)
        
        # Compute areas of boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBAreas = (boxesB[:, 2] - boxesB[:, 0]) * (boxesB[:, 3] - boxesB[:, 1])

        # Compute IoU
        ious = interAreas / (boxAArea + boxBAreas - interAreas + 1e-6)
        return ious

