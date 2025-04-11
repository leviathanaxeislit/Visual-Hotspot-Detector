from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import io
import mediapipe as mp  # Import Mediapipe

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

class HotspotResponse(BaseModel):
    message: str
    saliency_map_base64: str = None
    hotspot_regions: list = []
    face_regions: list = [] # Add face_regions to response (optional)
    hotspot_scores: dict = {} # Include hotspot scores for debugging/visualization


# Initialize Mediapipe Face Detection model (initialize globally for efficiency)
mp_face_detection = mp.solutions.face_detection
face_detection_model = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 or 1 (0 for short-range, 1 for full-range) - 1 is generally better
    min_detection_confidence=0.5 # Confidence threshold for detections
)


@app.post("/hotspots", response_model=HotspotResponse)
async def detect_hotspots(request: HotspotRequest):
    """
    Endpoint to receive image and DOM data, compute Spectral Residual Saliency,
    incorporate DOM heuristics and BlazeFace detection, extract hotspot regions,
    and return bounding boxes using HYBRID SCORING.
    """
    try:
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        dom_data = request.dom_data # Access DOM data from request

        # --- 1. Initialize Spectral Residual Saliency Algorithm ---
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        # --- 2. Compute Saliency Map ---
        (success, saliency_map) = saliency.computeSaliency(img)

        if not success:
            raise HTTPException(status_code=500, detail="Saliency computation failed")

        # --- 3. Threshold Saliency Map to Get Initial Hotspot Mask ---
        thresh_value = saliency_map.mean() * 1.5 # Example threshold
        _, thresh_map = cv2.threshold(saliency_map, thresh_value, 1, cv2.THRESH_BINARY)
        thresh_map_uint8 = (thresh_map * 255).astype(np.uint8)

        # --- 4. Find Contours (Initial Hotspot Regions) ---
        contours, _ = cv2.findContours(thresh_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 5. Extract Bounding Boxes and Initial Scores for Saliency Hotspots ---
        hotspot_regions_bbox = []
        hotspot_scores = {} # Dictionary to store scores for each region (bbox: score)
        for i, contour in enumerate(contours): # Enumerate contours for indexing
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [int(x), int(y), int(x + w), int(x + h)]
            hotspot_regions_bbox.append(bbox)
            # Initial score based on average saliency in the region (you can refine this)
            mask = np.zeros_like(saliency_map, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1) # Draw filled contour
            mean_saliency = cv2.mean(saliency_map, mask=mask)[0] # Get mean saliency within contour
            hotspot_scores[tuple(bbox)] = mean_saliency # Store score, use tuple for bbox as key

        # --- 6. Face Detection using BlazeFace (Mediapipe) ---
        face_detection_result = face_detection_model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Mediapipe expects RGB input

        detected_faces_bbox = [] # List to store bounding boxes of detected faces
        if face_detection_result.detections:
            for detection in face_detection_result.detections:
                bboxC = detection.location_data.relative_bounding_box # Relative bounding box
                ih, iw, ic = img.shape # Image height, width, channels
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)
                detected_faces_bbox.append(bbox) # Store face bounding box [x1, y1, x2, y2]

        print(f"Number of faces detected by BlazeFace: {len(detected_faces_bbox)}") # Log face detection count


        # --- 7. Incorporate DOM Heuristics and Face Detection for HYBRID SCORING ---
        dom_boost_factor = 1.5 # Reduced DOM boost factor slightly
        face_boost_factor = 2.5 # Example boost factor for face regions

        if dom_data:
            for dom_element in dom_data:
                tag_name = dom_element['tag_name']
                element_bbox = dom_element['bounding_box'] # [left, top, right, bottom] from JS
                element_bbox_tuple = tuple(map(int, element_bbox))

                # Check if DOM element's bounding box overlaps with any saliency-based hotspot region
                for bbox_tuple in list(hotspot_scores.keys()):
                    if check_bbox_overlap(element_bbox, bbox_tuple):
                        if tag_name in ['input', 'button', 'a', 'nav', 'form', 'textarea', 'select']:
                            original_score = hotspot_scores[bbox_tuple]
                            hotspot_scores[bbox_tuple] *= dom_boost_factor # Boost score
                            boosted_score = hotspot_scores[bbox_tuple]
                            print(f"Boosted hotspot score for region {bbox_tuple} from {original_score:.4f} to {boosted_score:.4f} due to DOM element: {tag_name}")

        if detected_faces_bbox: # Check if faces were detected
            for face_bbox in detected_faces_bbox:
                face_bbox_tuple = tuple(map(int, face_bbox))
                for bbox_tuple in list(hotspot_scores.keys()):
                    if check_bbox_overlap(face_bbox, bbox_tuple):
                        original_score = hotspot_scores[bbox_tuple]
                        hotspot_scores[bbox_tuple] *= face_boost_factor # Boost score for face regions
                        boosted_score = hotspot_scores[bbox_tuple]
                        print(f"Boosted hotspot score for region {bbox_tuple} from {original_score:.4f} to {boosted_score:.4f} due to FACE detection.")


        # --- 8. Rank Hotspot Regions by HYBRID SCORE (Saliency + DOM + Face Boost) ---
        ranked_hotspot_regions = sorted(hotspot_scores.keys(), key=hotspot_scores.get, reverse=True) # Rank ALL hotspots by their combined scores
        ranked_hotspot_regions_bbox = [list(bbox) for bbox in ranked_hotspot_regions]

        # --- 9. Convert Saliency Map to Grayscale Image (uint8) for visualization (optional) ---
        saliency_map_normalized = (saliency_map * 255).astype(np.uint8)
        saliency_map_grayscale = cv2.cvtColor(saliency_map_normalized, cv2.COLOR_GRAY2BGR)
        is_success, im_buf_arr = cv2.imencode(".png", saliency_map_grayscale)
        byte_im = im_buf_arr.tobytes()
        saliency_map_base64_str = base64.b64encode(byte_im).decode('utf-8')


        # --- For now, print a success message and number of hotspot regions ---
        print("Spectral Residual Saliency Map computed, Hotspot Regions extracted, DOM Heuristics and Face Detection applied (HYBRID SCORING)!")
        print(f"Number of Hotspot Regions (after Hybrid Scoring): {len(ranked_hotspot_regions_bbox)}")

        # --- Return Response with base64 encoded saliency map and RANKED hotspot regions ---
        return HotspotResponse(
            message="Spectral Residual Saliency computed, Hotspot Regions extracted, DOM Heuristics and Face Detection applied (HYBRID SCORING).",
            saliency_map_base64=saliency_map_base64_str, # Keep saliency map for visualization if needed
            hotspot_regions=ranked_hotspot_regions_bbox, # Return RANKED list of hotspot bounding boxes
            face_regions=detected_faces_bbox, # Return face bounding boxes (optional)
            hotspot_scores=hotspot_scores # Return hotspot scores for debugging/visualization
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error during saliency computation or hotspot extraction/DOM/Face processing: {e}")


def check_bbox_overlap(bbox1, bbox2):
    """
    Checks if two bounding boxes (in [left, top, right, bottom] format) overlap.
    """
    r1 = {'x': bbox1[0], 'y': bbox1[1], 'width': bbox1[2] - bbox1[0], 'height': bbox1[3] - bbox1[1]}
    r2 = {'x': bbox2[0], 'y': bbox2[1], 'width': bbox2[2] - bbox2[0], 'height': bbox2[3] - bbox2[1]}
    return not (r1['x'] + r1['width'] < r2['x'] or r1['x'] > r2['x'] + r2['width'] or r1['y'] + r1['height'] < r2['y'] or r1['y'] > r2['y'] + r2['height'])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)