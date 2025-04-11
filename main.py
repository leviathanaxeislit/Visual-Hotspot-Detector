from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64

app = FastAPI()

class HotspotRequest(BaseModel):
    image_base64: str

class HotspotResponse(BaseModel):
    message: str
    # We will expand this response model later, maybe include saliency map later

@app.post("/hotspots", response_model=HotspotResponse)
async def detect_hotspots(request: HotspotRequest):
    """
    Endpoint to receive a base64 encoded image and process it for hotspot detection using
    Spectral Residual Saliency.
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

        # --- 3. Basic Processing/Normalization of Saliency Map (Optional, for visualization later) ---
        # Saliency map is usually in range [0, 255] or [0, 1].  Normalize to [0, 255] for now if needed.
        # if saliency_map.max() > 1: # Assuming it might be in range [0, 255]
        #     saliency_map = (saliency_map * 255).astype(np.uint8)
        saliency_map_uint8 = (saliency_map * 255).astype(np.uint8) # Convert to uint8 for image display if needed

        # --- For now, just print a success message and the shape of the saliency map ---
        print("Spectral Residual Saliency Map computed successfully!")
        print(f"Saliency Map Shape: {saliency_map.shape}, Data Type: {saliency_map.dtype}")

        # --- Placeholder response ---
        return HotspotResponse(message="Spectral Residual Saliency computed. Hotspot region extraction to be implemented.")

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error during saliency computation: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)