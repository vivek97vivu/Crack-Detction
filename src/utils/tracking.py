import numpy as np

class SimpleBBoxTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.tracked_dets = {} # track_id -> bbox_xyxy (list or np.ndarray)

    def update(self, detections):
        """
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

        # Match incoming detections with tracked detections using IOU
        matched_track_ids = set()
        new_tracked_dets = {}
        
        for det in detections:
            bbox = np.array(det.bbox_xyxy)
            best_iou = -1.0
            best_track_id = None
            
            for track_id, tracked_box in self.tracked_dets.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._compute_iou(bbox, np.array(tracked_box))
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None and best_iou >= self.iou_threshold:
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

    def _compute_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
