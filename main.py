from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import io
import mediapipe as mp
from typing import List, Dict, Any, Tuple, Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://localhost",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HotspotRequest(BaseModel):
    image_base64: str
    dom_data: list = []
    viewport_size: dict = {"width": 1280, "height": 800}  # Default viewport size

class HotspotResponse(BaseModel):
    message: str
    saliency_map_base64: str = None
    attention_heatmap_base64: str = None  # New heatmap visualization
    hotspot_regions: list = []
    face_regions: list = []
    eye_movement_path: list = []  # New: predicted eye movement path
    dom_importance: dict = {}  # DOM element importance scores
    hotspot_scores: dict = {}  # Include hotspot scores for debugging/visualization

# Initialize Mediapipe Face Detection model
mp_face_detection = mp.solutions.face_detection
face_detection_model = mp_face_detection.FaceDetection(
    model_selection=1,  # 1 for full-range detection
    min_detection_confidence=0.7  # Increased confidence threshold
)

# DOM element importance weights - based on interaction likelihood
DOM_ELEMENT_WEIGHTS = {
    'button': 3.0,      # High interaction elements
    'a': 2.5,
    'input': 2.8,
    'textarea': 2.5,
    'select': 2.5,
    'form': 2.0,
    'nav': 2.2,
    'img': 1.8,         # Content elements
    'video': 2.0,
    'h1': 2.0,
    'h2': 1.7,
    'h3': 1.5,
    'li': 1.3,
    'div': 1.0,         # Structure elements
    'p': 1.2,           # Text content
    'span': 1.0,
    'label': 1.5
}

# Default weight for unlisted elements
DEFAULT_DOM_WEIGHT = 1.0

# New: Figma layer type weights
FIGMA_LAYER_WEIGHTS = {
    'frame': 1.0,
    'group': 1.0,
    'rectangle': 1.5, # Often used for buttons or content backgrounds
    'text': 1.8,      # Text content is usually important
    'instance': 2.0,  # Component instances can be very important (e.g. buttons, icons)
    'component': 2.0, # Similar to instances
    'vector': 1.2,    # Icons or graphical elements
    'ellipse': 1.2,
    'line': 1.0,
    'boolean_operation': 1.0,
    # Add other Figma layer types as needed
}
DEFAULT_FIGMA_WEIGHT = 1.0

class FigmaElement(BaseModel):
    id: str
    name: str
    type: str  # Layer type e.g. FRAME, TEXT, RECTANGLE
    bounding_box: List[float] # [x1, y1, x2, y2] relative to the frame
    text_content: Optional[str] = None
    visible: bool = True
    opacity: Optional[float] = 1.0
    children_count: Optional[int] = 0 # Number of direct children, can indicate complexity
    # We can add more Figma-specific properties like fills, effects, component_properties later
    # New fields for detailed Figma properties
    fontSize: Optional[float] = None
    fontWeight: Optional[float] = None # Figma API typically provides this as a number (e.g., 400, 700)
    fontName: Optional[Dict[str, str]] = None # e.g., {"family": "Inter", "style": "Regular"}
    textCase: Optional[str] = None # e.g., "ORIGINAL", "UPPER", "LOWER", "TITLE"
    lineHeight: Optional[Dict[str, Any]] = None # e.g., {"value": 24, "unit": "PIXELS"}
    
    layoutMode: Optional[str] = None # "NONE", "HORIZONTAL", "VERTICAL"
    primaryAxisSizingMode: Optional[str] = None # "AUTO", "FIXED"
    counterAxisSizingMode: Optional[str] = None # "AUTO", "FIXED"
    itemSpacing: Optional[float] = None # Spacing between items in auto-layout

    fillColor: Optional[Dict[str, float]] = None # e.g., {"r": 0, "g": 0, "b": 0} (0-1 scale)
    fillOpacity: Optional[float] = None


class FigmaHotspotRequest(BaseModel):
    image_base64: str
    figma_layer_data: List[FigmaElement] = []
    viewport_size: dict = {"width": 1280, "height": 800} # Viewport of the exported frame

def apply_center_bias(saliency_map: np.ndarray, img_shape: Tuple, sigma: float = 0.33) -> np.ndarray:
    """
    Apply center bias to saliency map (humans tend to look at center of screen).
    
    Args:
        saliency_map: Original saliency map
        img_shape: Shape of the image
        sigma: Strength of center bias (proportion of image size)
        
    Returns:
        Center-biased saliency map
    """
    h, w = img_shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    center_y, center_x = h // 2, w // 2
    dist_from_center = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (sigma * min(h, w))**2))
    return saliency_map * dist_from_center

def get_element_importance(dom_element: Dict[str, Any]) -> float:
    """
    Calculate importance of a DOM element based on its type and attributes.
    
    Args:
        dom_element: Dictionary containing DOM element information
        
    Returns:
        Importance score of the element
    """
    tag_name = dom_element.get('tag_name', '').lower()
    
    # Base weight from element type
    weight = DOM_ELEMENT_WEIGHTS.get(tag_name, DEFAULT_DOM_WEIGHT)
    
    # Check attributes for semantic importance
    attributes = dom_element.get('attributes', {})
    
    # Check class names for importance indicators
    classes = attributes.get('class', '').lower()
    if any(key in classes for key in ['button', 'btn', 'nav', 'menu', 'primary', 'main', 'cta', 'action', 'header', 'footer']):
        weight *= 1.5
    
    # Check roles for accessibility importance
    role = attributes.get('role', '').lower()
    if role in ['button', 'link', 'navigation', 'menu', 'tab', 'search', 'banner']:
        weight *= 1.4
        
    # Check IDs for importance
    element_id = attributes.get('id', '').lower()
    if any(key in element_id for key in ['main', 'header', 'nav', 'menu', 'button', 'search', 'logo']):
        weight *= 1.3
    
    # Check text content - elements with shorter, action-oriented text often get more attention
    text_content = dom_element.get('text_content', '')
    if text_content and len(text_content) < 20:
        weight *= 1.2
        
    # Check for high contrast elements
    has_background = 'background' in attributes.get('style', '').lower() or 'background-color' in attributes.get('style', '').lower()
    if has_background:
        weight *= 1.2
        
    return weight

def calculate_dom_boost(dom_element: Dict[str, Any], img_shape: Tuple) -> float:
    """
    Calculate boost factor for DOM element based on its position and size.
    
    Args:
        dom_element: Dictionary containing DOM element information
        img_shape: Shape of the image
        
    Returns:
        Boost factor for the element
    """
    # Get element position relative to viewport
    bbox = dom_element.get('bounding_box', [0, 0, 10, 10])
    
    # Skip elements that are very small (likely not important)
    element_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    viewport_area = img_shape[0] * img_shape[1]
    
    # If element is too small (less than 0.5% of viewport)
    if element_area / viewport_area < 0.005:
        return 1.0  # No boost for tiny elements
    
    # Check if element is in F-pattern (Western reading pattern - high attention to top-left)
    # F-pattern: higher weight for elements in top third and left half
    is_in_top_third = bbox[1] < img_shape[0] / 3
    is_in_left_half = bbox[0] < img_shape[1] / 2
    
    f_pattern_boost = 1.0
    if is_in_top_third:
        f_pattern_boost += 0.3
    if is_in_left_half:
        f_pattern_boost += 0.2
    
    # Check if element is in viewport center (more important)
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Calculate distance from viewport center
    viewport_center_x = img_shape[1] / 2
    viewport_center_y = img_shape[0] / 2
    
    # Normalize distance
    max_distance = np.sqrt(viewport_center_x**2 + viewport_center_y**2)
    distance = np.sqrt((center_x - viewport_center_x)**2 + (center_y - viewport_center_y)**2)
    normalized_distance = distance / max_distance
    
    # Center elements get higher boost (inverse relationship with distance)
    center_boost = 1.0 + (1.0 - normalized_distance) * 0.5
    
    # Get base importance from element type and attributes
    element_importance = get_element_importance(dom_element)
    
    # Calculate final boost as combination of element importance, center position and F-pattern
    final_boost = element_importance * center_boost * f_pattern_boost
    
    return final_boost

def check_bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """
    Checks if two bounding boxes (in [left, top, right, bottom] format) overlap.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        True if boxes overlap, False otherwise
    """
    r1 = {'x': bbox1[0], 'y': bbox1[1], 'width': bbox1[2] - bbox1[0], 'height': bbox1[3] - bbox1[1]}
    r2 = {'x': bbox2[0], 'y': bbox2[1], 'width': bbox2[2] - bbox2[0], 'height': bbox2[3] - bbox2[1]}
    return not (r1['x'] + r1['width'] < r2['x'] or r1['x'] > r2['x'] + r2['width'] or 
                r1['y'] + r1['height'] < r2['y'] or r1['y'] > r2['y'] + r2['height'])

def non_max_suppression(boxes: List[List[int]], scores: List[float], overlap_thresh: float = 0.4) -> List[int]:
    """
    Apply non-maximum suppression to avoid duplicate hotspots.
    
    Args:
        boxes: List of bounding boxes
        scores: List of corresponding scores
        overlap_thresh: Overlap threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
        
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Convert to format expected by NMS: [x1, y1, x2, y2]
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    # Initialize list of picked indices
    pick = []
    
    # Grab coordinates of bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute area of bounding boxes and sort by confidence score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    
    # Keep looping while indices remain
    while len(idxs) > 0:
        # Grab last index (highest score)
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find indices of all boxes with overlap above threshold
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Calculate overlap area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        
        # Delete all indices that overlap significantly
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return pick

def generate_eye_movement_path(hotspot_regions: List[List[int]], hotspot_scores: Dict, 
                              img_shape: Tuple, num_points: int = 10) -> List[List[int]]:
    """
    Generate a predicted eye movement path based on hotspot regions.
    
    Args:
        hotspot_regions: List of hotspot bounding boxes
        hotspot_scores: Dictionary of scores for each region
        img_shape: Shape of the image
        num_points: Number of points in the eye movement path
        
    Returns:
        List of [x, y] coordinates representing predicted eye movement
    """
    if not hotspot_regions:
        return []
    
    # Extract center points of hotspots
    centers = []
    scores = []
    for region in hotspot_regions:
        center_x = (region[0] + region[2]) // 2
        center_y = (region[1] + region[3]) // 2
        centers.append([center_x, center_y])
        
        # Get score for this region
        region_tuple = tuple(region)
        score = hotspot_scores.get(region_tuple, 1.0)
        scores.append(score)
    
    # Add viewport center as starting point (people often start looking at center)
    h, w = img_shape[:2]
    centers.insert(0, [w//2, h//3])  # Start slightly above center (common pattern)
    scores.insert(0, max(scores) * 0.8)  # Give it a reasonably high score
    
    # Sort hotspots by score
    sorted_indices = np.argsort(scores)[::-1]  # Descending order
    sorted_centers = [centers[i] for i in sorted_indices]
    
    # Take top hotspots based on num_points
    top_centers = sorted_centers[:min(num_points, len(sorted_centers))]
    
    # F-pattern reading - add some bias to top-left region (Western reading pattern)
    # Add a point in top-left quadrant if not already present
    top_left_present = any(c[0] < w//2 and c[1] < h//2 for c in top_centers)
    if not top_left_present and len(top_centers) < num_points:
        top_centers.append([w//4, h//4])  # Add point in top-left quadrant
    
    # Start from center or top hotspot and create a path
    path = [top_centers[0]]
    remaining = top_centers[1:]
    
    # Greedy path construction - go to nearest unvisited hotspot each time
    while remaining and len(path) < num_points:
        current = path[-1]
        
        # Find closest remaining hotspot
        distances = [np.sqrt((p[0]-current[0])**2 + (p[1]-current[1])**2) for p in remaining]
        nearest_idx = np.argmin(distances)
        path.append(remaining[nearest_idx])
        remaining.pop(nearest_idx)
    
    return path

def create_attention_heatmap(img: np.ndarray, hotspot_regions: List[List[int]], 
                           hotspot_scores: Dict, path: List[List[int]]) -> np.ndarray:
    """
    Create a visual heatmap showing predicted attention/eye movement.
    
    Args:
        img: Original image
        hotspot_regions: List of hotspot bounding boxes
        hotspot_scores: Dictionary of scores for each region
        path: Eye movement path
        
    Returns:
        Attention heatmap as a numpy array
    """
    # Create empty heatmap
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    
    # Add hotspots to heatmap
    for region in hotspot_regions:
        x1, y1, x2, y2 = region
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Get score for this region
        region_tuple = tuple(region)
        intensity = hotspot_scores.get(region_tuple, 1.0)
        
        # Calculate radius based on region size
        radius = int(min(x2-x1, y2-y1) * 0.5)
        radius = max(radius, 30)  # Ensure minimum radius
        
        # Apply Gaussian around center (weighted by intensity)
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = np.exp(-(dist_from_center**2) / (2 * radius**2)) * intensity
        heatmap = np.maximum(heatmap, mask)
    
    # Add eye path to heatmap with stronger intensity
    if path:
        for i, point in enumerate(path):
            x, y = point
            # Make sure coordinates are within bounds
            x = min(max(x, 0), img.shape[1]-1)
            y = min(max(y, 0), img.shape[0]-1)
            
            # Calculate intensity (decreasing along path)
            path_intensity = 2.0 * (1.0 - i / len(path))
            
            # Apply smaller Gaussian at path point
            y_grid, x_grid = np.ogrid[:img.shape[0], :img.shape[1]]
            dist = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
            mask = np.exp(-(dist**2) / (2 * 20**2)) * path_intensity  # Smaller radius
            heatmap = np.maximum(heatmap, mask)
    
    # Normalize to 0-1
    heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original image
    alpha = 0.6
    blended = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    
    return blended

# New: Calculate importance for Figma elements
def get_figma_element_importance(figma_element: FigmaElement) -> float:
    """
    Calculate importance of a Figma element based on its type and properties.
    """
    layer_type = figma_element.type.lower()
    weight = FIGMA_LAYER_WEIGHTS.get(layer_type, DEFAULT_FIGMA_WEIGHT)

    # Boost based on name (e.g., if name contains 'button', 'cta', 'icon', 'image', 'title', 'header')
    name_lower = figma_element.name.lower()
    if any(key in name_lower for key in ['button', 'btn', 'cta', 'primary', 'action', 'submit', 'buy', 'shop']):
        weight *= 1.8
    elif any(key in name_lower for key in ['icon', 'logo', 'image', 'avatar', 'profile', 'thumbnail', 'illustration']):
        weight *= 1.5
    elif any(key in name_lower for key in ['title', 'header', 'headline', 'caption', 'banner', 'hero']):
        weight *= 1.6
    elif any(key in name_lower for key in ['menu', 'nav', 'navigation', 'link', 'tab']):
        weight *= 1.4
    elif any(key in name_lower for key in ['input', 'form', 'field', 'search']):
        weight *= 1.3


    # Boost for text layers with short, concise text (often labels or CTAs)
    if layer_type == 'text' and figma_element.text_content:
        if len(figma_element.text_content) > 0 and len(figma_element.text_content) < 50: # Short text
            weight *= 1.3
            # Further boost for all-caps short text
            if figma_element.textCase == 'UPPER' and len(figma_element.text_content) < 25:
                 weight *= 1.2
        elif len(figma_element.text_content) == 0: # Empty text layers are less important
             weight *= 0.7
        
        # Font size boost (relative to a baseline, e.g. 16px)
        if figma_element.fontSize:
            if figma_element.fontSize > 24: # Significantly larger text
                weight *= 1.25
            elif figma_element.fontSize > 18: # Larger text
                weight *= 1.1
            elif figma_element.fontSize < 12: # Very small text, less important
                weight *= 0.85
        
        # Font weight boost
        if figma_element.fontWeight:
            if figma_element.fontWeight >= 700: # Bold or heavier
                weight *= 1.2
            elif figma_element.fontWeight <= 300: # Light
                weight *= 0.9


    # Boost for visibility and opacity
    if not figma_element.visible:
        return 0.01 # Very low importance if not visible
    if figma_element.opacity is not None and figma_element.opacity < 0.5:
        weight *= (0.3 + figma_element.opacity) # Scale down if low opacity, but not to zero if visible
    elif figma_element.opacity is not None and figma_element.opacity < 0.1: # Almost transparent
        return 0.05


    # Boost for component instances and layers with children (complexity/grouping)
    if layer_type in ['instance', 'component']:
        weight *= 1.2
    if figma_element.children_count and figma_element.children_count > 0:
        # Modest boost, as complex groups don't always mean visual importance of the group itself
        weight *= (1 + min(figma_element.children_count * 0.02, 0.1)) 

    # Boost for elements with a prominent fill color (not white or very light gray, and high opacity)
    if figma_element.fillColor:
        r, g, b = figma_element.fillColor.get('r', 1), figma_element.fillColor.get('g', 1), figma_element.fillColor.get('b', 1)
        # Avoid boosting for white-ish backgrounds by checking if color is not too light
        # (Luminance calculation approximation: 0.299*R + 0.587*G + 0.114*B)
        # If color is "dark" enough (e.g. luminance < 0.8 on 0-1 scale)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        fill_opacity = figma_element.fillOpacity if figma_element.fillOpacity is not None else 1.0
        
        if luminance < 0.85 and fill_opacity > 0.7: # Not too light and reasonably opaque
            weight *= 1.15
        elif fill_opacity < 0.3: # Very transparent fill
             weight *= 0.9


    # Consider auto-layout properties - e.g. if an element is a key part of a structured layout
    if figma_element.layoutMode and figma_element.layoutMode != 'NONE':
        weight *= 1.05 # Slight boost for being part of an auto-layout structure

    return max(0.01, weight) # Ensure a minimum small positive weight if not invisible

# New: Calculate boost for Figma layers (positional, size)
def calculate_figma_layer_boost(figma_element: FigmaElement, img_shape: Tuple) -> float:
    """
    Calculate boost factor for a Figma layer based on its position and size within the frame.
    """
    bbox = figma_element.bounding_box
    
    # Ensure bounding box has valid dimensions (x1, y1, x2, y2)
    if not (len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]):
        return 1.0 # No boost for invalid bbox

    element_width = bbox[2] - bbox[0]
    element_height = bbox[3] - bbox[1]
    
    # Skip elements that are very small (likely not important visually)
    element_area = element_width * element_height
    viewport_area = img_shape[0] * img_shape[1]
    
    if element_area <= 0 or viewport_area <= 0: # Avoid division by zero
        return 1.0

    if (element_area / viewport_area) < 0.001: # Adjusted threshold for Figma (0.1% of frame area)
        return 0.8  # Reduced boost for very tiny elements, but not zero

    # F-pattern: higher weight for elements in top third and left half
    # Assumes img_shape is (height, width)
    frame_height, frame_width = img_shape[:2]
    is_in_top_third = bbox[1] < frame_height / 3
    is_in_left_half = bbox[0] < frame_width / 2
    
    f_pattern_boost = 1.0
    if is_in_top_third:
        f_pattern_boost += 0.3
    if is_in_left_half:
        f_pattern_boost += 0.2
    
    # Center bias: elements closer to the center get higher boost
    center_x = bbox[0] + element_width / 2
    center_y = bbox[1] + element_height / 2
    
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
    if max_distance == 0: max_distance = 1 # Avoid division by zero for tiny frames
    
    distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
    normalized_distance = distance / max_distance
    
    center_boost = 1.0 + (1.0 - normalized_distance) * 0.5 # Boost ranges from 1.0 to 1.5
    
    # Get base importance from element type and properties
    element_importance = get_figma_element_importance(figma_element)
    if element_importance == 0: return 0 # If invisible or deemed unimportant

    final_boost = element_importance * center_boost * f_pattern_boost
    
    return max(0.1, final_boost) # Ensure a minimum boost if element is somewhat important

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/hotspots", response_model=HotspotResponse)
async def detect_hotspots(request: HotspotRequest):
    """
    Endpoint to analyze webpage, predict eye movement patterns and detect visual hotspots.
    Returns saliency map, hotspot regions, predicted eye movement path and attention heatmap.
    """
    try:
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        dom_data = request.dom_data
        viewport_size = request.viewport_size

        # 1. INITIALIZE SPECTRAL RESIDUAL SALIENCY ALGORITHM
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        # 2. COMPUTE SALIENCY MAP
        (success, saliency_map) = saliency.computeSaliency(img)
        if not success:
            raise HTTPException(status_code=500, detail="Saliency computation failed")

        # 3. APPLY CENTER BIAS TO SALIENCY MAP (people tend to look at center)
        saliency_map = apply_center_bias(saliency_map, img.shape)

        # 4. APPLY ADAPTIVE THRESHOLDING
        thresh_value = saliency_map.mean() + 1.5 * np.std(saliency_map)
        _, thresh_map = cv2.threshold(saliency_map, thresh_value, 1, cv2.THRESH_BINARY)
        thresh_map_uint8 = (thresh_map * 255).astype(np.uint8)

        # 5. FIND AND FILTER CONTOURS (REMOVE TINY NOISE)
        contours, _ = cv2.findContours(thresh_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = img.shape[0] * img.shape[1] * 0.001  # Minimum 0.1% of image area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # 6. EXTRACT BOUNDING BOXES AND INITIAL SCORES FOR HOTSPOTS
        hotspot_regions_bbox = []
        hotspot_scores = {}
        dom_importance = {}  # Store DOM element importance

        for i, contour in enumerate(filtered_contours):
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [int(x), int(y), int(x + w), int(y + h)]
            hotspot_regions_bbox.append(bbox)

            # Initial score based on weighted saliency and size
            mask = np.zeros_like(saliency_map, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_saliency = cv2.mean(saliency_map, mask=mask)[0]

            # Size factor - medium-sized regions get boost (not too small, not too large)
            region_area = w * h
            image_area = img.shape[0] * img.shape[1]
            size_ratio = region_area / image_area
            
            # Size scoring function - penalize very small and very large regions
            size_score = 1.0
            if size_ratio < 0.01:  # Too small
                size_score = 0.7
            elif size_ratio > 0.3:  # Too large
                size_score = 0.8
            
            # Position score - center regions get boost
            center_x = x + w/2
            center_y = y + h/2
            img_center_x = img.shape[1] / 2
            img_center_y = img.shape[0] / 2
            dist_to_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_dist = np.sqrt((img.shape[1])**2 + (img.shape[0])**2) / 2
            position_score = 1.0 - (dist_to_center / max_dist) * 0.5  # 0.5-1.0 range
            
            # Combined initial score
            initial_score = (0.6 * mean_saliency + 0.2 * size_score + 0.2 * position_score)
            hotspot_scores[tuple(bbox)] = initial_score

        # 7. FACE DETECTION USING MEDIAPIPE
        face_detection_result = face_detection_model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detected_faces_bbox = []

        if face_detection_result.detections:
            for detection in face_detection_result.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = [
                    int(bboxC.xmin * iw), 
                    int(bboxC.ymin * ih), 
                    int((bboxC.xmin + bboxC.width) * iw), 
                    int((bboxC.ymin + bboxC.height) * ih)
                ]
                detected_faces_bbox.append(bbox)

                # Add face hotspot with high score if not already detected
                face_added = False
                face_bbox_tuple = tuple(bbox)
                
                for hotspot_bbox in hotspot_scores.keys():
                    if check_bbox_overlap(bbox, hotspot_bbox):
                        # Boost existing hotspot that contains face
                        hotspot_scores[hotspot_bbox] *= 2.5  # High boost for faces
                        face_added = True
                
                if not face_added:
                    # Add new hotspot for face
                    hotspot_scores[face_bbox_tuple] = 2.0  # High initial score for faces
                    hotspot_regions_bbox.append(bbox)

        # 8. INCORPORATE DOM ELEMENTS FOR HYBRID SCORING
        if dom_data:
            for dom_element in dom_data:
                element_bbox = dom_element.get('bounding_box', [0, 0, 0, 0])
                
                # Skip invalid bounding boxes
                if element_bbox[2] <= element_bbox[0] or element_bbox[3] <= element_bbox[1]:
                    continue
                    
                element_bbox_tuple = tuple(map(int, element_bbox))
                
                # Calculate boost factor based on element properties
                dom_boost = calculate_dom_boost(dom_element, img.shape)
                dom_importance[str(element_bbox)] = dom_boost
                
                # Check if DOM element overlaps with any existing hotspot
                dom_element_added = False
                for bbox_tuple in list(hotspot_scores.keys()):
                    if check_bbox_overlap(element_bbox, bbox_tuple):
                        original_score = hotspot_scores[bbox_tuple]
                        hotspot_scores[bbox_tuple] *= dom_boost
                        dom_element_added = True
                
                # If important DOM element doesn't match any saliency region, add it as a new hotspot
                if not dom_element_added and dom_boost > 1.5:
                    # Only add important DOM elements that didn't match existing regions
                    element_bbox_int = [int(v) for v in element_bbox]
                    hotspot_scores[tuple(element_bbox_int)] = dom_boost * saliency_map.mean() * 1.5
                    hotspot_regions_bbox.append(element_bbox_int)

        # 9. NON-MAXIMUM SUPPRESSION - MERGE OVERLAPPING REGIONS
        # Convert dict of scores to list in same order as bboxes
        score_list = [hotspot_scores[tuple(bbox)] for bbox in hotspot_regions_bbox]
        
        # Apply NMS if we have multiple regions
        if len(hotspot_regions_bbox) > 1:
            keep_indices = non_max_suppression(hotspot_regions_bbox, score_list, overlap_thresh=0.4)
            hotspot_regions_bbox = [hotspot_regions_bbox[i] for i in keep_indices]
            
            # Update scores dictionary to only contain kept boxes
            new_scores = {}
            for bbox in hotspot_regions_bbox:
                bbox_tuple = tuple(bbox)
                if bbox_tuple in hotspot_scores:
                    new_scores[bbox_tuple] = hotspot_scores[bbox_tuple]
            hotspot_scores = new_scores

        # 10. GENERATE EYE MOVEMENT PATH BASED ON HOTSPOTS
        eye_movement_path = generate_eye_movement_path(
            hotspot_regions_bbox, 
            hotspot_scores, 
            img.shape
        )

        # 11. CREATE ATTENTION HEATMAP (VISUALIZATION)
        attention_heatmap = create_attention_heatmap(
            img,
            hotspot_regions_bbox,
            hotspot_scores,
            eye_movement_path
        )

        # 12. PREPARE VISUALIZATION OUTPUTS
        # Saliency map for visualization
        saliency_map_normalized = (saliency_map * 255).astype(np.uint8)
        saliency_map_colormap = cv2.applyColorMap(saliency_map_normalized, cv2.COLORMAP_JET)
        is_success, saliency_buf = cv2.imencode(".png", saliency_map_colormap)
        saliency_base64_str = base64.b64encode(saliency_buf.tobytes()).decode('utf-8')
        
        # Attention heatmap for visualization
        is_success, heatmap_buf = cv2.imencode(".png", attention_heatmap)
        heatmap_base64_str = base64.b64encode(heatmap_buf.tobytes()).decode('utf-8')

        # 13. RANK HOTSPOT REGIONS BY HYBRID SCORE
        ranked_hotspot_regions = sorted(hotspot_scores.keys(), key=hotspot_scores.get, reverse=True)
        ranked_hotspot_regions_bbox = [list(bbox) for bbox in ranked_hotspot_regions]

        # Convert score dictionary for JSON serialization (tuples can't be keys in JSON)
        json_scores = {str(bbox): score for bbox, score in hotspot_scores.items()}

        # Log stats about detection
        print(f"Number of Hotspot Regions: {len(ranked_hotspot_regions_bbox)}")
        print(f"Number of Face Regions: {len(detected_faces_bbox)}")
        print(f"Eye Movement Path Points: {len(eye_movement_path)}")

        # Return enhanced response with eye movement prediction and attention heatmap
        return HotspotResponse(
            message="Enhanced visual attention analysis completed with eye movement prediction.",
            saliency_map_base64=saliency_base64_str,
            attention_heatmap_base64=heatmap_base64_str,
            hotspot_regions=ranked_hotspot_regions_bbox,
            face_regions=detected_faces_bbox,
            eye_movement_path=eye_movement_path,
            dom_importance=dom_importance,
            hotspot_scores=json_scores
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, 
                          detail=f"Error during attention analysis: {str(e)}")

@app.post("/figma_visual_analysis", response_model=HotspotResponse)
async def figma_detect_hotspots(request: FigmaHotspotRequest):
    """
    Endpoint to analyze Figma frame, predict eye movement patterns and detect visual hotspots,
    using Figma layer data instead of DOM data.
    """
    try:
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        figma_layer_data = request.figma_layer_data
        # viewport_size = request.viewport_size # Use img.shape for actual image dimensions

        # 1. INITIALIZE SPECTRAL RESIDUAL SALIENCY ALGORITHM
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        # 2. COMPUTE SALIENCY MAP
        (success, saliency_map_cv) = saliency.computeSaliency(img)
        if not success or saliency_map_cv is None:
            raise HTTPException(status_code=500, detail="Saliency computation failed")
        
        # Ensure saliency map is float
        saliency_map_cv = saliency_map_cv.astype(np.float32)


        # 3. APPLY CENTER BIAS TO SALIENCY MAP
        saliency_map_biased = apply_center_bias(saliency_map_cv, img.shape)

        # 4. APPLY ADAPTIVE THRESHOLDING
        # Ensure saliency_map_biased is not empty and has finite values
        if saliency_map_biased.size == 0 or not np.all(np.isfinite(saliency_map_biased)) :
            raise HTTPException(status_code=500, detail="Biased saliency map is invalid")

        mean_val = np.mean(saliency_map_biased[np.isfinite(saliency_map_biased)])
        std_val = np.std(saliency_map_biased[np.isfinite(saliency_map_biased)])
        thresh_value = mean_val + 1.5 * std_val
        
        # Handle cases where std_val might be zero or very small
        if std_val < 1e-6:
            thresh_value = mean_val * 1.1 # Fallback threshold
        if not np.isfinite(thresh_value): # if mean/std were non-finite (e.g. all-zero map)
             thresh_value = 0.5 # Default if stats are weird

        _, thresh_map = cv2.threshold(saliency_map_biased, thresh_value, 1.0, cv2.THRESH_BINARY) # Output as float 0.0-1.0
        thresh_map_uint8 = (thresh_map * 255).astype(np.uint8)


        # 5. FIND AND FILTER CONTOURS
        contours, _ = cv2.findContours(thresh_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = img.shape[0] * img.shape[1] * 0.001  # Minimum 0.1% of image area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # 6. EXTRACT BOUNDING BOXES AND INITIAL SCORES FOR HOTSPOTS FROM SALIENCY
        hotspot_regions_bbox = []
        hotspot_scores = {} # Using tuple(bbox) as key

        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [int(x), int(y), int(x + w), int(y + h)]
            
            mask = np.zeros_like(saliency_map_biased, dtype=np.uint8) # Use biased map for scoring
            cv2.drawContours(mask, [contour], -1, 1, -1) # Binary mask
            
            # Calculate mean saliency only on the biased map
            mean_saliency = cv2.mean(saliency_map_biased, mask=mask)[0]

            region_area = w * h
            image_area = img.shape[0] * img.shape[1]
            size_ratio = region_area / image_area if image_area > 0 else 0
            
            size_score = 1.0
            if size_ratio < 0.005: size_score = 0.7 # Penalize very small (less than 0.5%)
            elif size_ratio > 0.35: size_score = 0.8 # Penalize very large

            center_x_pos = x + w/2
            center_y_pos = y + h/2
            img_center_x = img.shape[1] / 2
            img_center_y = img.shape[0] / 2
            max_dist = np.sqrt((img.shape[1]/2)**2 + (img.shape[0]/2)**2) if img.shape[0]*img.shape[1] > 0 else 1
            dist_to_center = np.sqrt((center_x_pos - img_center_x)**2 + (center_y_pos - img_center_y)**2)
            position_score = 1.0 - (dist_to_center / max_dist) * 0.5 if max_dist > 0 else 0.75
            
            initial_score = (0.6 * mean_saliency + 0.2 * size_score + 0.2 * position_score)
            hotspot_regions_bbox.append(bbox)
            hotspot_scores[tuple(bbox)] = max(0.01, initial_score) # Ensure positive score


        # 7. FACE DETECTION USING MEDIAPIPE
        face_detection_result = face_detection_model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        detected_faces_bbox = []

        if face_detection_result.detections:
            for detection in face_detection_result.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                face_bbox = [
                    int(bboxC.xmin * iw), 
                    int(bboxC.ymin * ih), 
                    int((bboxC.xmin + bboxC.width) * iw), 
                    int((bboxC.ymin + bboxC.height) * ih)
                ]
                # Ensure face_bbox coordinates are within image bounds and valid
                face_bbox[0] = max(0, face_bbox[0])
                face_bbox[1] = max(0, face_bbox[1])
                face_bbox[2] = min(iw -1, face_bbox[2])
                face_bbox[3] = min(ih -1, face_bbox[3])
                if face_bbox[2] <= face_bbox[0] or face_bbox[3] <= face_bbox[1]: continue

                detected_faces_bbox.append(face_bbox)
                face_bbox_tuple = tuple(face_bbox)
                
                face_added_to_hotspots = False
                for hs_bbox_tuple in list(hotspot_scores.keys()): # Iterate over a copy of keys
                    if check_bbox_overlap(face_bbox, list(hs_bbox_tuple)):
                        hotspot_scores[hs_bbox_tuple] = hotspot_scores.get(hs_bbox_tuple, 0.1) * 2.5 # Boost existing
                        face_added_to_hotspots = True
                
                if not face_added_to_hotspots:
                    hotspot_scores[face_bbox_tuple] = 2.0 # High initial score for new face hotspot
                    if face_bbox not in hotspot_regions_bbox: # Avoid duplicates if somehow already present
                        hotspot_regions_bbox.append(face_bbox)
        
        # 8. INCORPORATE FIGMA LAYER DATA FOR HYBRID SCORING
        figma_layer_importance_scores = {} # For storing figma layer based scores, similar to dom_importance

        if figma_layer_data:
            for f_element in figma_layer_data:
                if not f_element.visible: continue

                element_bbox = [int(c) for c in f_element.bounding_box] # Ensure int coords
                # Validate Figma element bounding box against image dimensions
                element_bbox[0] = max(0, element_bbox[0])
                element_bbox[1] = max(0, element_bbox[1])
                element_bbox[2] = min(img.shape[1] - 1, element_bbox[2])
                element_bbox[3] = min(img.shape[0] - 1, element_bbox[3])

                if element_bbox[2] <= element_bbox[0] or element_bbox[3] <= element_bbox[1]:
                    continue # Skip invalid or zero-area boxes

                figma_boost = calculate_figma_layer_boost(f_element, img.shape)
                figma_layer_importance_scores[f_element.id] = figma_boost # Use Figma element ID as key

                if figma_boost <= 0.1: continue # Skip if not deemed important by Figma properties

                element_bbox_tuple = tuple(element_bbox)
                element_added_or_boosted = False
                for hs_bbox_tuple in list(hotspot_scores.keys()):
                    if check_bbox_overlap(element_bbox, list(hs_bbox_tuple)):
                        hotspot_scores[hs_bbox_tuple] = hotspot_scores.get(hs_bbox_tuple, 0.1) * figma_boost
                        element_added_or_boosted = True
                
                if not element_added_or_boosted and figma_boost > 1.5: # Threshold for adding as new hotspot
                     # Base score on figma_boost and mean saliency of image
                    mean_saliency_val = np.mean(saliency_map_biased[np.isfinite(saliency_map_biased)])
                    if not np.isfinite(mean_saliency_val): mean_saliency_val = 0.1

                    new_score = figma_boost * mean_saliency_val * 0.5 # Modest score contribution
                    hotspot_scores[element_bbox_tuple] = max(0.1, new_score)
                    if element_bbox not in hotspot_regions_bbox:
                         hotspot_regions_bbox.append(element_bbox)
        
        # 9. NON-MAXIMUM SUPPRESSION
        if not hotspot_regions_bbox: # No hotspots found at all
             # Return empty or minimal response
            return HotspotResponse(
                message="No distinct visual hotspots or important Figma elements detected.",
                saliency_map_base64="", # Or a blank image base64
                attention_heatmap_base64="",
                hotspot_regions=[],
                face_regions=detected_faces_bbox, # Keep faces if any
                eye_movement_path=[],
                dom_importance={}, # Empty for Figma context, or use figma_layer_importance_scores
                hotspot_scores={}
            )

        # Ensure hotspot_scores keys match current hotspot_regions_bbox before NMS
        current_hs_tuples = {tuple(b) for b in hotspot_regions_bbox}
        valid_scores = {k: v for k, v in hotspot_scores.items() if k in current_hs_tuples}
        
        # Rebuild score_list based on valid_scores and hotspot_regions_bbox order
        score_list = [valid_scores.get(tuple(bbox), 0.01) for bbox in hotspot_regions_bbox] # Default to low score if missing

        final_hotspot_regions_bbox = hotspot_regions_bbox
        final_hotspot_scores = valid_scores

        if len(hotspot_regions_bbox) > 1:
            keep_indices = non_max_suppression(np.array(hotspot_regions_bbox, dtype=object), np.array(score_list), overlap_thresh=0.4)
            final_hotspot_regions_bbox = [hotspot_regions_bbox[i] for i in keep_indices]
            
            new_scores_after_nms = {}
            for bbox_idx in keep_indices:
                bbox_tuple = tuple(hotspot_regions_bbox[bbox_idx])
                if bbox_tuple in valid_scores: # ensure key exists
                    new_scores_after_nms[bbox_tuple] = valid_scores[bbox_tuple]
            final_hotspot_scores = new_scores_after_nms
        elif not hotspot_regions_bbox: # If all were suppressed or started empty
            final_hotspot_regions_bbox = []
            final_hotspot_scores = {}


        # 10. GENERATE EYE MOVEMENT PATH
        eye_movement_path = generate_eye_movement_path(
            final_hotspot_regions_bbox, 
            final_hotspot_scores, 
            img.shape
        )

        # 11. CREATE ATTENTION HEATMAP
        attention_heatmap = create_attention_heatmap(
            img,
            final_hotspot_regions_bbox,
            final_hotspot_scores,
            eye_movement_path
        )

        # 12. PREPARE VISUALIZATION OUTPUTS (Saliency & Heatmap)
        # Saliency map for visualization (using the biased one)
        saliency_map_display = (saliency_map_biased / np.max(saliency_map_biased) * 255).astype(np.uint8) if np.max(saliency_map_biased) > 0 else np.zeros_like(saliency_map_biased, dtype=np.uint8)
        saliency_map_colormap = cv2.applyColorMap(saliency_map_display, cv2.COLORMAP_JET)
        is_success, saliency_buf = cv2.imencode(".png", saliency_map_colormap)
        saliency_base64_str = base64.b64encode(saliency_buf.tobytes()).decode('utf-8') if is_success else ""
        
        is_success, heatmap_buf = cv2.imencode(".png", attention_heatmap)
        heatmap_base64_str = base64.b64encode(heatmap_buf.tobytes()).decode('utf-8') if is_success else ""

        # 13. RANK HOTSPOT REGIONS
        # Use final_hotspot_scores which keys are tuples
        ranked_hotspot_tuples = sorted(final_hotspot_scores.keys(), key=final_hotspot_scores.get, reverse=True)
        ranked_hotspot_regions_list = [list(bbox) for bbox in ranked_hotspot_tuples]

        json_scores = {str(bbox): score for bbox, score in final_hotspot_scores.items()}
        # Use figma_layer_importance_scores for dom_importance field or rename it in HotspotResponse
        json_figma_importance = {key: val for key, val in figma_layer_importance_scores.items()}


        print(f"Figma Analysis - Hotspots: {len(ranked_hotspot_regions_list)}, Faces: {len(detected_faces_bbox)}, Eye Path Points: {len(eye_movement_path)}")

        return HotspotResponse(
            message="Figma frame visual analysis completed.",
            saliency_map_base64=saliency_base64_str,
            attention_heatmap_base64=heatmap_base64_str,
            hotspot_regions=ranked_hotspot_regions_list,
            face_regions=detected_faces_bbox,
            eye_movement_path=eye_movement_path,
            dom_importance=json_figma_importance, # Sending Figma layer importance here
            hotspot_scores=json_scores
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        import traceback
        print(f"Error processing Figma frame: {e} {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error during Figma attention analysis: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)