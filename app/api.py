from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import uvicorn
import io
import base64
from PIL import Image
import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuroscan.inference import NeuroScanPipeline

app = FastAPI(title="NeuroScan API", description="Advanced MRI Pathology Classification API", version="2.0.0")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline lazily to handle missing models gracefully
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    print("Initializing NeuroScan pipeline...")
    pipeline = NeuroScanPipeline()

def encode_image_base64(img_array):
    if img_array is None:
        return None
    # Convert RGB/BGR to BGR for imencode if needed, but since we usually send RGB to frontend,
    # let's assume it's RGB and convert to BGR for OpenCV encoding.
    # Actually, imencode expects BGR. Our pipeline returns RGB for overlay.
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/api/v1/predict")
async def predict_mri(file: UploadFile = File(...)):
    if pipeline.seg_model is None or pipeline.cls_model is None:
        raise HTTPException(status_code=503, detail="Models are not fully loaded. Train the models first.")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
        # Convert to BGR for the pipeline
        open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = pipeline.process_image(open_cv_image)
        
        # Prepare response (encode images to base64)
        response_data = {
            "pred_class": results["pred_class"],
            "confidence": results["confidence"],
            "probabilities": results["probabilities"],
            "images": {
                "overlay": encode_image_base64(results["overlay"]),
                "cropped_roi": encode_image_base64(results["cropped_roi"]),
                "gradcam_overlay": encode_image_base64(results["gradcam_overlay"])
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
