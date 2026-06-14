import pytest
import numpy as np
import tensorflow as tf
from neuroscan.cls_model import build_classifier_net
from neuroscan.explainability import generate_gradcam_heatmap, superimpose_heatmap

def test_generate_gradcam_heatmap():
    model = build_classifier_net(input_shape=(224, 224, 3), num_classes=4)
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Generate heatmap
    heatmap = generate_gradcam_heatmap(model, dummy_input)
    
    assert heatmap is not None
    assert len(heatmap.shape) == 2 # 2D array
    assert heatmap.shape[0] == 7 and heatmap.shape[1] == 7 # EfficientNetB0/V2B0 last conv output size for 224x224 input
    
    # Check bounds
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

def test_superimpose_heatmap():
    dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    dummy_heatmap = np.random.rand(7, 7).astype(np.float32)
    
    overlay = superimpose_heatmap(dummy_img, dummy_heatmap)
    
    assert overlay is not None
    assert overlay.shape == (128, 128, 3)
    assert overlay.dtype == np.uint8
