from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import io

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

@app.post("/hotspots", response_model=HotspotResponse)
async def detect_hotspots(request: HotspotRequest):
    """
    Endpoint to receive image and DOM data, compute Spectral Residual Saliency,
    incorporate DOM heuristics, extract hotspot regions, and return bounding boxes.
    """
    try:
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        dom_data = request.dom_data # Access DOM data from request
        print("\n--- Received DOM Data from Plugin ---")
        if dom_data:
            for element_data in dom_data:
                print(f"  Tag: {element_data['tag_name']}, BBox: {element_data['bounding_box']}, Text: {element_data.get('text_content', 'No Text')[:50]}...") # Print first 50 chars of text
        else:
            print("  No DOM data received.")
        print("--- End of DOM Data ---\n")

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
            bbox = [int(x), int(y), int(x + w), int(y + h)]
            hotspot_regions_bbox.append(bbox)
            # Initial score based on average saliency in the region (you can refine this)
            mask = np.zeros_like(saliency_map, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1) # Draw filled contour
            mean_saliency = cv2.mean(saliency_map, mask=mask)[0] # Get mean saliency within contour
            hotspot_scores[tuple(bbox)] = mean_saliency # Store score, use tuple for bbox as key

        # --- 6. Incorporate DOM Heuristics to Adjust Hotspot Scores ---
        dom_boost_factor = 2.0 # Example boost factor for DOM-important elements
        if dom_data: # Check if dom_data is not empty
            for dom_element in dom_data:
                tag_name = dom_element['tag_name']
                element_bbox = dom_element['bounding_box'] # [left, top, right, bottom] from JS
                element_bbox_tuple = tuple(map(int, element_bbox)) # Convert to tuple of ints for comparison

                # Check if DOM element's bounding box overlaps with any saliency-based hotspot region
                for bbox_tuple in list(hotspot_scores.keys()): # Iterate over a copy of keys to allow modification
                    if check_bbox_overlap(element_bbox, bbox_tuple): # Function to check overlap (see below)
                        # Boost score of overlapping hotspot region based on DOM element type
                        if tag_name in ['input', 'button', 'a', 'nav', 'form', 'textarea', 'select']: # Example important tags
                            hotspot_scores[bbox_tuple] *= dom_boost_factor # Boost score
                            print(f"Boosted hotspot score for region {bbox_tuple} due to DOM element: {tag_name}")


        # --- 7. Sort Hotspot Regions by Score (Descending) ---
        ranked_hotspot_regions = sorted(hotspot_scores.keys(), key=hotspot_scores.get, reverse=True)
        ranked_hotspot_regions_bbox = [list(bbox) for bbox in ranked_hotspot_regions] # Convert back to list of lists

        # --- 8. Convert Saliency Map to Grayscale Image (uint8) for visualization (optional) ---
        saliency_map_normalized = (saliency_map * 255).astype(np.uint8)
        saliency_map_grayscale = cv2.cvtColor(saliency_map_normalized, cv2.COLOR_GRAY2BGR)
        is_success, im_buf_arr = cv2.imencode(".png", saliency_map_grayscale)
        byte_im = im_buf_arr.tobytes()
        saliency_map_base64_str = base64.b64encode(byte_im).decode('utf-8')


        # --- For now, print a success message and number of hotspot regions ---
        print("Spectral Residual Saliency Map computed, Hotspot Regions extracted, and DOM Heuristics applied!")
        print(f"Number of Hotspot Regions (after DOM boost): {len(ranked_hotspot_regions_bbox)}")

        # --- Return Response with base64 encoded saliency map and RANKED hotspot regions ---
        return HotspotResponse(
            message="Spectral Residual Saliency computed, Hotspot Regions extracted, and DOM Heuristics applied.",
            saliency_map_base64=saliency_map_base64_str, # Keep saliency map for visualization if needed
            hotspot_regions=ranked_hotspot_regions_bbox # Return RANKED list of hotspot bounding boxes
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error during saliency computation or hotspot extraction/DOM processing: {e}")


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