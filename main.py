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
                              img_shape: Tuple, num_points: int = 12) -> List[List[int]]:
    """
    Generate a predicted eye movement path based on hotspot regions.
    Uses cognitive models of visual attention and reading patterns.
    
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
    
    h, w = img_shape[:2]
    
    # Calculate hotspot centers and collect their scores
    centers = []
    scores = []
    for region in hotspot_regions:
        # Calculate center of region
        center_x = int((region[0] + region[2]) / 2)
        center_y = int((region[1] + region[3]) / 2)
        centers.append([center_x, center_y])
        
        # Get score for this region
        region_tuple = tuple(region)
        score = hotspot_scores.get(region_tuple, 1.0)
        scores.append(score)
    
    # Initialize with entry point - users typically start looking at top-left or center
    # Mix both to create a realistic starting point slightly above center
    entry_point = [w // 3, h // 3]  # Start at about 1/3 from top-left, common entry point
    path = [entry_point]
    
    # Add points with strategic logic based on cognitive models
    remaining_hotspots = list(zip(centers, scores))
    
    # Sort hotspots by importance (score)
    remaining_hotspots.sort(key=lambda x: x[1], reverse=True)
    
    # Extract top N hotspots based on score (where N is num_points - 1 to account for entry point)
    top_hotspots = remaining_hotspots[:min(num_points - 1, len(remaining_hotspots))]
    
    # First add the most important hotspot after entry point (highest score)
    if top_hotspots:
        path.append(top_hotspots[0][0])
        top_hotspots = top_hotspots[1:]
    
    # Now build a path considering both proximity and importance
    # This is more realistic than just nearest-neighbor or pure importance
    while top_hotspots and len(path) < num_points:
        current = path[-1]
        
        # Calculate scores that balance distance and importance
        # People tend to look at important things even if they're further away
        composite_scores = []
        for i, (center, importance) in enumerate(top_hotspots):
            # Calculate distance
            distance = np.sqrt((center[0] - current[0])**2 + (center[1] - current[1])**2)
            
            # Normalize distance to 0-1 range (1 is closest)
            max_distance = np.sqrt(w**2 + h**2)
            normalized_distance = 1 - (distance / max_distance)
            
            # Weight importance more than distance (70% importance, 30% proximity)
            composite_score = 0.7 * importance + 0.3 * normalized_distance
            composite_scores.append((i, composite_score))
        
        # Choose the hotspot with highest composite score
        best_index = max(composite_scores, key=lambda x: x[1])[0]
        path.append(top_hotspots[best_index][0])
        top_hotspots.pop(best_index)
    
    # Add realistic micro-saccades (small eye movements between fixations)
    enhanced_path = []
    for i in range(len(path)):
        if i > 0:
            # Add a micro-saccade midpoint with slight random deviation
            # This simulates how eyes don't move in perfectly straight lines
            start = path[i-1]
            end = path[i]
            
            # Calculate midpoint with jitter
            jitter_x = np.random.randint(-20, 20)
            jitter_y = np.random.randint(-15, 15)
            
            # Add perpendicular deviation to create arc-like movement
            dx, dy = end[0] - start[0], end[1] - start[1]
            dist = max(1, np.sqrt(dx**2 + dy**2))
            
            # Perpendicular vector with magnitude proportional to distance
            perp_x, perp_y = -dy/dist * dist*0.1, dx/dist * dist*0.1
            
            mid_x = int((start[0] + end[0])/2 + perp_x + jitter_x)
            mid_y = int((start[1] + end[1])/2 + perp_y + jitter_y)
            
            # Keep within image bounds
            mid_x = max(0, min(mid_x, w-1))
            mid_y = max(0, min(mid_y, h-1))
            
            enhanced_path.append(start)
            enhanced_path.append([mid_x, mid_y])
        
        # Add the actual fixation point
        enhanced_path.append(path[i])
    
    # Add a final point that simulates return to the main content
    # People often return to the main content area after exploring
    if enhanced_path and enhanced_path[-1][1] > h/2:  # If we ended in the bottom half
        # Add a final point heading back toward the top
        final_x = enhanced_path[-1][0]
        final_y = max(h//4, enhanced_path[-1][1] - h//3)
        enhanced_path.append([final_x, final_y])
    
    return enhanced_path

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
    
    # Calculate max score for normalization
    max_score = max(hotspot_scores.values()) if hotspot_scores else 1.0
    
    # Add hotspots to heatmap with improved Gaussian distribution
    for region in hotspot_regions:
        x1, y1, x2, y2 = region
        
        # Skip invalid regions
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Calculate region dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Find region center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Get score for this region, normalized between 0.2-1.0
        region_tuple = tuple(region)
        score = hotspot_scores.get(region_tuple, 0.2)
        normalized_score = 0.2 + 0.8 * (score / max_score)
        
        # Calculate adaptive radius based on region size and score
        # Larger elements need larger radius, but radius should scale sublinearly
        base_radius = min(width, height) * 0.4
        # Higher scores get larger radius (more attention spread)
        radius = max(30, base_radius * (0.8 + 0.4 * normalized_score))
        
        # Create a 2D Gaussian kernel around the center point
        y_grid, x_grid = np.ogrid[:img.shape[0], :img.shape[1]]
        
        # Adaptive sigma for Gaussian based on region shape - wider for wider regions
        sigma_x = radius * (width / max(1, height)) * 0.8
        sigma_y = radius * (height / max(1, width)) * 0.8
        
        # Ensure minimum sigma
        sigma_x = max(20, sigma_x)
        sigma_y = max(20, sigma_y)
        
        # Anisotropic Gaussian (different spread in x and y directions based on region shape)
        gaussian = np.exp(-(
            ((x_grid - center_x) ** 2) / (2 * sigma_x ** 2) + 
            ((y_grid - center_y) ** 2) / (2 * sigma_y ** 2)
        )) * normalized_score
        
        # Add to existing heatmap using maximum blend
        heatmap = np.maximum(heatmap, gaussian)
    
    # Add eye path to heatmap with realistic scanpath modeling
    if path:
        # Generate more natural eye movement pattern with fixations and saccades
        for i, point in enumerate(path):
            if i >= len(path) - 1:
                break
                
            x1, y1 = point
            x2, y2 = path[i + 1]
            
            # Ensure coordinates are within bounds
            x1 = min(max(x1, 0), img.shape[1]-1)
            y1 = min(max(y1, 0), img.shape[0]-1)
            x2 = min(max(x2, 0), img.shape[1]-1)
            y2 = min(max(y2, 0), img.shape[0]-1)
            
            # Fixation intensity - decreases along path (first points get more attention)
            # We use a nonlinear decay function to model attention decay
            fixation_intensity = 1.8 * np.exp(-i / max(1, len(path) * 0.33))
            
            # Create a more prominent fixation point at start of path segment
            y_grid, x_grid = np.ogrid[:img.shape[0], :img.shape[1]]
            # Fixations are more concentrated (smaller sigma)
            fixation_sigma = 15 + (5 * i / max(1, len(path)))  # Fixations get slightly wider later in path
            dist_sqr = ((x_grid - x1)**2 + (y_grid - y1)**2)
            fixation = np.exp(-dist_sqr / (2 * fixation_sigma**2)) * fixation_intensity
            heatmap = np.maximum(heatmap, fixation)
            
            # Generate saccade (eye movement path) between fixation points with gradually decreasing intensity
            # Only create saccade for significant movements
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist > 10:  # Only draw saccade if points are far enough apart
                num_steps = max(3, int(dist / 15))  # More steps for longer distances
                for step in range(1, num_steps):
                    # Interpolate position along saccade path
                    alpha = step / num_steps
                    # Add slight curve to saccade path (more realistic)
                    mid_deviation = np.sin(alpha * np.pi) * 10
                    dx = (x2 - x1) * alpha
                    dy = (y2 - y1) * alpha
                    # Perpendicular offset for curved path
                    if abs(x2 - x1) > abs(y2 - y1):
                        # More horizontal movement
                        px, py = x1 + dx, y1 + dy + mid_deviation
                    else:
                        # More vertical movement
                        px, py = x1 + dx + mid_deviation, y1 + dy
                        
                    # Constrain to image bounds
                    px = min(max(int(px), 0), img.shape[1]-1)
                    py = min(max(int(py), 0), img.shape[0]-1)
                    
                    # Saccades have lower intensity than fixations and fade along path
                    saccade_intensity = fixation_intensity * 0.3 * (1 - alpha)
                    saccade_sigma = 10  # Narrower spread for saccade path
                    
                    # Add saccade point to heatmap
                    dist = ((x_grid - px)**2 + (y_grid - py)**2)
                    saccade_point = np.exp(-dist / (2 * saccade_sigma**2)) * saccade_intensity
                    heatmap = np.maximum(heatmap, saccade_point)
    
    # Normalize to 0-1
    heatmap_max = np.max(heatmap)
    if heatmap_max > 0:
        heatmap = heatmap / heatmap_max 
    
    # Apply subtle Gaussian blur to smooth the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # Create customized colormap for better visualization
    # Convert from grayscale (0-1) to colormap
    heatmap_8bit = (heatmap * 255).astype(np.uint8)
    
    # Use a custom color mapping for more realistic attention heatmap
    # Start with INFERNO colormap but modify alpha/opacity by value
    colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_INFERNO)
    
    # Create alpha channel based on intensity - low values become transparent
    alpha_channel = np.clip(heatmap * 2.5, 0, 1)  # Boost low values for visibility, but keep transparent
    
    # Convert to BGRA (BGR with alpha channel)
    bgra = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    bgra[..., 0:3] = colored_heatmap
    bgra[..., 3] = (alpha_channel * 255).astype(np.uint8)
    
    # Blend with original image
    # Using alpha compositing
    alpha_fg = alpha_channel[..., np.newaxis]
    alpha_bg = 1.0
    
    # Final blend
    foreground = colored_heatmap * alpha_fg
    background = img * alpha_bg * (1 - alpha_fg)
    blended = (foreground + background).astype(np.uint8)
    
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
    # New: Detect common UI patterns
    elif any(key in name_lower for key in ['card', 'panel', 'section']):
        weight *= 1.2
    elif any(key in name_lower for key in ['footer', 'copyright', 'terms']):
        weight *= 0.7  # Less important
    elif any(key in name_lower for key in ['divider', 'separator', 'spacer']):
        weight *= 0.5  # Even less important

    # Boost for text layers with short, concise text (often labels or CTAs)
    if layer_type == 'text' and figma_element.text_content:
        if len(figma_element.text_content) > 0 and len(figma_element.text_content) < 50: # Short text
            weight *= 1.3
            # Further boost for all-caps short text
            if figma_element.textCase == 'UPPER' and len(figma_element.text_content) < 25:
                 weight *= 1.2
            # New: Detect common call-to-action text
            text_lower = figma_element.text_content.lower()
            if any(cta in text_lower for cta in ['sign up', 'login', 'register', 'buy now', 'get started', 'try free', 'download', 'subscribe']):
                weight *= 1.4
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

        # New: Boost for high contrast colors (very dark or saturated colors)
        if luminance < 0.2 or (max(r, g, b) - min(r, g, b) > 0.5):  # Dark or highly saturated
            weight *= 1.25

    # Consider auto-layout properties - e.g. if an element is a key part of a structured layout
    if figma_element.layoutMode and figma_element.layoutMode != 'NONE':
        weight *= 1.05 # Slight boost for being part of an auto-layout structure
        # New: More boost for primary axis items in auto-layout
        if figma_element.primaryAxisSizingMode == 'FIXED':
            weight *= 1.1  # Fixed size items in auto-layout often more important
    
    # New: Evaluate item importance by position within parent frame
    if any(pos in name_lower for pos in ['top', 'header']):
        weight *= 1.15
    elif any(pos in name_lower for pos in ['first', 'primary']):
        weight *= 1.2
    
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

    # Calculate area ratio - use logarithmic scale to better handle wide range of element sizes
    area_ratio = element_area / viewport_area
    if area_ratio < 0.001: # Very tiny elements (0.1% of frame area or less)
        return 0.7  # Reduced boost for very tiny elements, but not zero
    elif area_ratio > 0.5:  # Very large elements (50% of frame or more)
        return 1.1  # Large elements get slight boost but not excessive

    # Modified F-pattern: Western reading patterns use more sophisticated model
    # - Top third horizontal band (high importance)
    # - Left vertical band (high importance)
    # - Middle third horizontal band (medium importance)
    # - Z-pattern diagonal relationships
    frame_height, frame_width = img_shape[:2]
    
    # Element center point
    center_x = bbox[0] + element_width / 2
    center_y = bbox[1] + element_height / 2
    
    # Normalized coordinates (0-1 range)
    norm_x = center_x / frame_width
    norm_y = center_y / frame_height
    
    # Base reading pattern boost - Z-pattern with higher weights for top and left
    reading_pattern_boost = 1.0
    
    # Top band boost (decreases as we move down)
    top_boost = max(0.0, 1.0 - (norm_y * 2.0)) * 0.4  # Ranges from 0.4 to 0 top-to-bottom
    
    # Left side boost (decreases as we move right)
    left_boost = max(0.0, 1.0 - (norm_x * 1.5)) * 0.3  # Ranges from 0.3 to 0 left-to-right
    
    # Golden ratio points boost (naturally attractive focal points)
    golden_ratio = 0.618
    gr_points = [
        (golden_ratio, golden_ratio),  # Golden ratio point
        (1-golden_ratio, golden_ratio),  # Second focal point
        (golden_ratio, 1-golden_ratio),  # Third focal point
        (1-golden_ratio, 1-golden_ratio)  # Fourth focal point
    ]
    
    # Calculate distance to nearest golden ratio point
    min_gr_dist = min(
        ((norm_x - gr_x)**2 + (norm_y - gr_y)**2)**0.5
        for gr_x, gr_y in gr_points
    )
    
    # Boost for proximity to golden ratio points
    golden_boost = max(0.0, 0.2 - min_gr_dist) * 1.0  # Up to 0.2 boost for exact match
    
    # Combined reading pattern boost
    reading_pattern_boost += top_boost + left_boost + golden_boost
    
    # Center bias: elements closer to the center still get a boost
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    max_distance = (frame_width**2 + frame_height**2)**0.5 / 2
    if max_distance == 0: max_distance = 1  # Avoid division by zero
    
    distance = ((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)**0.5
    normalized_distance = distance / max_distance
    
    # Exponential falloff for center distance (gentler falloff than linear)
    center_boost = 1.0 + (1.0 - min(1.0, normalized_distance**0.8)) * 0.4  # Ranges from 1.0 to 1.4
    
    # Get base importance from element type and properties
    element_importance = get_figma_element_importance(figma_element)
    if element_importance == 0: return 0  # If invisible or deemed unimportant
    
    # Size weight - neither too small nor too large is optimal for attention
    size_weight = 1.0
    if area_ratio > 0.001 and area_ratio < 0.05:  # Between 0.1% and 5% of frame
        size_weight = 1.2  # Boost for "just right" sized elements
    
    # Aspect ratio analysis - elements with unusual aspect ratios draw more attention
    if element_width > 0 and element_height > 0:
        aspect_ratio = element_width / element_height
        aspect_normalcy = min(aspect_ratio, 1/aspect_ratio) if aspect_ratio != 0 else 0.1
        
        # Unusual aspect ratios get a slight boost (very wide or very tall)
        if aspect_normalcy < 0.3:  # Aspect ratio more extreme than 1:3
            size_weight *= 1.1
    
    # Final boost calculation
    final_boost = element_importance * center_boost * reading_pattern_boost * size_weight
    
    return max(0.1, min(3.0, final_boost))  # Clamp between 0.1 and 3.0 to avoid extreme values

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

        # Preprocess image for better saliency detection
        # Convert to Lab color space for perceptual processing
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        
        # Enhanced preprocessing for better visual saliency detection
        # Perceptual sharpening to accentuate edges and features
        img_l = img_lab[:,:,0]
        sharpened_l = cv2.addWeighted(img_l, 1.5, cv2.GaussianBlur(img_l, (0, 0), 3), -0.5, 0)
        img_lab[:,:,0] = np.clip(sharpened_l, 0, 255).astype(np.uint8)
        
        # Convert back to BGR for saliency detection
        img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)

        # 2. COMPUTE SALIENCY MAP
        (success, saliency_map_cv) = saliency.computeSaliency(img_enhanced)
        if not success or saliency_map_cv is None:
            raise HTTPException(status_code=500, detail="Saliency computation failed")
        
        # Ensure saliency map is float
        saliency_map_cv = saliency_map_cv.astype(np.float32)
        
        # 2.5 APPLY ADDITIONAL PERCEPTUAL FACTORS TO SALIENCY MAP
        
        # Color psychology factors - highlight red and yellow (action/attention colors)
        # Convert to HSV for easier color detection
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create color prominence maps
        # Red detection (wraps around hue space)
        red_mask1 = cv2.inRange(img_hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(img_hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.add(red_mask1, red_mask2)
        
        # Yellow detection
        yellow_mask = cv2.inRange(img_hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
        
        # Blue detection (trust/calm - slightly less attention-grabbing than red/yellow)
        blue_mask = cv2.inRange(img_hsv, np.array([100, 100, 100]), np.array([140, 255, 255]))
        
        # Normalize and weight color masks
        red_influence = cv2.GaussianBlur(red_mask.astype(np.float32) / 255.0, (21, 21), 0) * 0.3
        yellow_influence = cv2.GaussianBlur(yellow_mask.astype(np.float32) / 255.0, (21, 21), 0) * 0.25
        blue_influence = cv2.GaussianBlur(blue_mask.astype(np.float32) / 255.0, (21, 21), 0) * 0.1
        
        # Edge detection for contrast (high contrast areas attract attention)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_influence = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (21, 21), 0) * 0.2
        
        # Combine color psychology and edge detection with original saliency
        saliency_map_perceptual = np.clip(
            saliency_map_cv + red_influence + yellow_influence + blue_influence + edge_influence, 
            0, 1
        )

        # 3. APPLY CENTER BIAS TO SALIENCY MAP
        saliency_map_biased = apply_center_bias(saliency_map_perceptual, img.shape)

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
            
            # New: Calculate visual contrast score within region
            if region_area > 0:
                region_roi = img[y:y+h, x:x+w]
                if region_roi.size > 0:
                    region_gray = cv2.cvtColor(region_roi, cv2.COLOR_BGR2GRAY)
                    if region_gray.size > 0:
                        region_std = np.std(region_gray)
                        contrast_score = min(1.0, region_std / 50.0)  # Normalize, cap at 1.0
                    else:
                        contrast_score = 0.5  # Default if region is empty
                else:
                    contrast_score = 0.5  # Default if region is empty
            else:
                contrast_score = 0.5  # Default for zero area
            
            # Weighted factors for combined score
            initial_score = (
                0.45 * mean_saliency +   # Saliency is most important
                0.20 * size_score +      # Size is relevant but not dominant
                0.15 * position_score +  # Position has moderate influence
                0.20 * contrast_score    # Contrast is important for visual pop
            )
            
            hotspot_regions_bbox.append(bbox)
            hotspot_scores[tuple(bbox)] = max(0.01, initial_score) # Ensure positive score

        # Continue with the rest of the function unchanged...
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
        # Saliency map for visualization (using the perceptual enhanced one)
        saliency_map_display = (saliency_map_perceptual / np.max(saliency_map_perceptual) * 255).astype(np.uint8) if np.max(saliency_map_perceptual) > 0 else np.zeros_like(saliency_map_perceptual, dtype=np.uint8)
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