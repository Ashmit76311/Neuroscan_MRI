import pytest
import tensorflow as tf
from keras import layers
from neuroscan.cls_model import build_classifier_net, unfreeze_classifier_base

def test_cls_model_build():
    model = build_classifier_net(input_shape=(224, 224, 3), num_classes=4)
    assert model is not None
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 4)
    
    # Verify base is frozen
    base_model = model.layers[1] # EfficientNet is layer 1
    assert not base_model.trainable

def test_cls_model_layers():
    model = build_classifier_net(input_shape=(224, 224, 3), num_classes=4)
    has_dropout = any(isinstance(l, layers.Dropout) for l in model.layers)
    has_dense = any(isinstance(l, layers.Dense) for l in model.layers)
    assert has_dropout
    assert has_dense

def test_unfreeze_base():
    model = build_classifier_net(input_shape=(224, 224, 3), num_classes=4)
    model = unfreeze_classifier_base(model, num_layers_to_unfreeze=2)
    
    base_model = model.layers[1]
    assert base_model.trainable
    
    # Check top layers of base model are trainable
    assert base_model.layers[-1].trainable
    assert base_model.layers[-2].trainable
    assert not base_model.layers[0].trainable

def test_cls_model_forward_pass():
    model = build_classifier_net(input_shape=(224, 224, 3), num_classes=4)
    dummy_input = tf.random.normal((2, 224, 224, 3))
    output = model(dummy_input)
    assert output.shape == (2, 4)
    
    # Check softmax output
    row_sums = tf.reduce_sum(output, axis=1).numpy()
    for row_sum in row_sums:
        assert abs(row_sum - 1.0) < 1e-5
