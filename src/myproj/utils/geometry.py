import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def opencv_skeletonize(binary_mask):
    """
    Fallback skeletonization using morphological thinning in OpenCV.
    """
    size = np.size(binary_mask)
    skel = np.zeros(binary_mask.shape, np.uint8)
    img = binary_mask.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

def get_skeleton(binary_mask):
    """
    Runs skeletonization on a binary mask.
    Input mask should be a numpy array with values 0 (background) and 1 or 255 (foreground).
    Returns binary mask with centerline as 255, background as 0.
    """
    # Normalize to 0 and 255
    binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    
    if HAS_SKIMAGE:
        # skimage expects binary bool array
        bool_mask = binary_mask > 0
        skel = skeletonize(bool_mask)
        return (skel.astype(np.uint8)) * 255
    else:
        return opencv_skeletonize(binary_mask)

def compute_eccentricity(contour):
    """
    Computes eccentricity of a contour using central moments.
    Eccentricity ranges from 0 (perfect circle) to 1 (line).
    """
    mu = cv2.moments(contour)
    if mu['m00'] == 0:
        return 0.0
    
    mu20 = mu['mu20'] / mu['m00']
    mu02 = mu['mu02'] / mu['m00']
    mu11 = mu['mu11'] / mu['m00']
    
    # Calculate eigenvalues of covariance matrix
    diff = mu20 - mu02
    common = np.sqrt(diff**2 + 4 * mu11**2)
    lambda1 = 0.5 * (mu20 + mu02 + common)
    lambda2 = 0.5 * (mu20 + mu02 - common)
    
    if lambda1 == 0:
        return 0.0
    
    # Avoid numerical issues
    ratio = lambda2 / lambda1
    if ratio < 0:
        ratio = 0.0
    elif ratio > 1:
        ratio = 1.0
        
    return np.sqrt(1.0 - ratio)

def extract_geometry(binary_mask, px_to_mm=0.1, min_eccentricity=0.6):
    """
    Extracts crack skeleton and measures width and length.
    
    Args:
        binary_mask (np.ndarray): Binary segmentation mask (0 and 255/1).
        px_to_mm (float): Scaling factor to convert pixels to mm.
        min_eccentricity (float): Threshold below which blobs are rejected.
        
    Returns:
        dict: Extracted geometry features or None if no valid crack found.
    """
    # Ensure binary format
    mask = (binary_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return None
        
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Filter by eccentricity to reject circular blobs (not cracks)
    valid_contours = []
    for c in contours:
        ecc = compute_eccentricity(c)
        if ecc >= min_eccentricity:
            valid_contours.append(c)
            
    if not valid_contours:
        return None
        
    # Re-draw filtered mask
    filtered_mask = np.zeros_like(mask)
    cv2.drawContours(filtered_mask, valid_contours, -1, 255, -1)
    
    if cv2.countNonZero(filtered_mask) == 0:
        return None
        
    # Get skeleton
    skel = get_skeleton(filtered_mask)
    skel_pixels = np.argwhere(skel > 0)
    if len(skel_pixels) == 0:
        return None
        
    # Compute distance transform on filtered mask
    dist_transform = cv2.distanceTransform(filtered_mask, cv2.DIST_L2, 5)
    
    # Get widths at centerline pixels
    # distance transform values are from pixel to nearest edge (radius)
    # width in pixels = radius * 2
    widths_px = dist_transform[skel > 0] * 2
    widths_mm = widths_px * px_to_mm
    
    max_width_mm = float(np.max(widths_mm))
    mean_width_mm = float(np.mean(widths_mm))
    std_width_mm = float(np.std(widths_mm))
    
    # Compute length
    # An estimate of length is count of skeleton pixels scaled by calibration factor.
    # To be more precise, diagonal connections represent sqrt(2) pixels.
    # Here we use pixel count * px_to_mm as a robust baseline length estimate.
    length_mm = float(len(skel_pixels) * px_to_mm)
    
    # Calculate bounding box aspect ratio
    x, y, w, h = cv2.boundingRect(np.vstack(valid_contours))
    aspect_ratio = float(max(w, h) / (min(w, h) if min(w, h) > 0 else 1))
    
    # Calculate centerline path coordinates
    # downsample path to max 50 points to prevent sending too much data
    path_coords = skel_pixels[:, ::-1].tolist() # convert to (x, y)
    if len(path_coords) > 50:
        indices = np.linspace(0, len(path_coords) - 1, 50, dtype=int)
        path_coords = [path_coords[i] for i in indices]
        
    return {
        "max_width_mm": max_width_mm,
        "mean_width_mm": mean_width_mm,
        "std_width_mm": std_width_mm,
        "length_mm": length_mm,
        "aspect_ratio": aspect_ratio,
        "path_coords": path_coords,
        "bounding_box": [x, y, w, h]
    }
