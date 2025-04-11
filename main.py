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

class HotspotResponse(BaseModel):
    message: str
    saliency_map_base64: str = None
    hotspot_regions: list = [] # Add hotspot regions to response

@app.post("/hotspots", response_model=HotspotResponse)
async def detect_hotspots(request: HotspotRequest):
    """
    Endpoint to receive a base64 encoded image, compute Spectral Residual Saliency,
    extract hotspot regions, and return bounding boxes.
    """
    try:
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # --- 1. Initialize Spectral Residual Saliency Algorithm ---
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

        # --- 2. Compute Saliency Map ---
        (success, saliency_map) = saliency.computeSaliency(img)

        if not success:
            raise HTTPException(status_code=500, detail="Saliency computation failed")

        # --- 3. Threshold Saliency Map to Get Hotspot Mask ---
        thresh_value = saliency_map.mean() * 1.5 # Example: Threshold based on mean saliency
        _, thresh_map = cv2.threshold(saliency_map, thresh_value, 1, cv2.THRESH_BINARY)
        thresh_map_uint8 = (thresh_map * 255).astype(np.uint8) # For contour finding

        # --- 4. Find Contours (Connected Hotspot Regions) ---
        contours, _ = cv2.findContours(thresh_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 5. Extract Bounding Boxes for Hotspot Regions ---
        hotspot_regions_bbox = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            hotspot_regions_bbox.append([int(x), int(y), int(x + w), int(y + h)]) # Convert to int and list

        # --- 6. Convert Saliency Map to Grayscale Image (uint8) for visualization (optional) ---
        saliency_map_normalized = (saliency_map * 255).astype(np.uint8)
        saliency_map_grayscale = cv2.cvtColor(saliency_map_normalized, cv2.COLOR_GRAY2BGR)
        is_success, im_buf_arr = cv2.imencode(".png", saliency_map_grayscale)
        byte_im = im_buf_arr.tobytes()
        saliency_map_base64_str = base64.b64encode(byte_im).decode('utf-8')


        # --- For now, print a success message and number of hotspot regions ---
        print("Spectral Residual Saliency Map computed and Hotspot Regions extracted!")
        print(f"Number of Hotspot Regions: {len(hotspot_regions_bbox)}")

        # --- Return Response with base64 encoded saliency map and hotspot regions ---
        return HotspotResponse(
            message="Spectral Residual Saliency computed and Hotspot Regions extracted.",
            saliency_map_base64=saliency_map_base64_str, # Keep saliency map for visualization if needed
            hotspot_regions=hotspot_regions_bbox # Return list of hotspot bounding boxes
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error during saliency computation or hotspot extraction: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 