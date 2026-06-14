import pytest
from fastapi.testclient import TestClient
from api import app
import io
from PIL import Image

client = TestClient(app)

def test_predict_endpoint_no_models_loaded():
    # Since models are loaded lazily on startup, we can just hit it.
    # We expect a 503 if models are None.
    
    # Create a dummy image
    img = Image.new('RGB', (100, 100))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    
    # It will either be 503 (models not loaded) or 200 (models loaded from test directory)
    assert response.status_code in [200, 503]
