import pytest
import tensorflow as tf
from keras import layers
from neuroscan.seg_model import build_segmentation_net

def test_seg_model_build():
    model = build_segmentation_net(input_shape=(224, 224, 1), filter_list=[16, 32, 64, 128])
    assert model is not None
    assert model.input_shape == (None, 224, 224, 1)
    assert model.output_shape == (None, 224, 224, 1)

def test_seg_model_layers():
    model = build_segmentation_net(input_shape=(224, 224, 1), filter_list=[16, 32])
    
    has_sep_conv = any(isinstance(l, layers.SeparableConv2D) for l in model.layers)
    has_batch_norm = any(isinstance(l, layers.BatchNormalization) for l in model.layers)
    has_upsampling = any(isinstance(l, layers.UpSampling2D) for l in model.layers)
    
    assert has_sep_conv, "Model should use SeparableConv2D"
    assert has_batch_norm, "Model should use BatchNormalization"
    assert has_upsampling, "Model should use UpSampling2D"

def test_seg_model_forward_pass():
    model = build_segmentation_net(input_shape=(64, 64, 1), filter_list=[8, 16]) # smaller for quick test
    dummy_input = tf.random.normal((1, 64, 64, 1))
    output = model(dummy_input)
    assert output.shape == (1, 64, 64, 1)
    
    # Check sigmoid output range
    out_min = tf.reduce_min(output).numpy()
    out_max = tf.reduce_max(output).numpy()
    assert out_min >= 0.0 and out_max <= 1.0
