import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from neuroscan.inference import NeuroScanPipeline

@patch('neuroscan.inference.tf.keras.models.load_model')
def test_pipeline_initialization(mock_load):
    # Mocking successful model load
    pipeline = NeuroScanPipeline(seg_model_path="dummy", cls_model_path="dummy")
    assert pipeline.seg_model is not None
    assert pipeline.cls_model is not None

@patch('neuroscan.inference.tf.keras.models.load_model')
def test_pipeline_process_image(mock_load):
    # Setup mocks
    mock_seg = MagicMock()
    # Mask predict returns array of shape (1, 224, 224, 1)
    mock_seg.predict.return_value = np.random.rand(1, 224, 224, 1)
    
    mock_cls = MagicMock()
    # Cls predict returns array of shape (1, 4)
    mock_cls.predict.return_value = np.array([[0.1, 0.7, 0.1, 0.1]])
    
    # Configure load_model to return our mocks
    mock_load.side_effect = [mock_seg, mock_cls]
    
    pipeline = NeuroScanPipeline(seg_model_path="dummy", cls_model_path="dummy")
    
    dummy_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    results = pipeline.process_image(dummy_image)
    
    assert "mask" in results
    assert "overlay" in results
    assert "cropped_roi" in results
    assert "pred_class" in results
    assert "confidence" in results
    assert "probabilities" in results
    
    assert results["pred_class"] == pipeline.classes[1]
    assert results["confidence"] == 0.7
