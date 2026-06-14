import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetV2B0

def build_classifier_net(input_shape=(224, 224, 3), num_classes=4):
    """
    Builds an EfficientNetV2B0-based classification model.
    
    Args:
        input_shape (tuple): Input image shape (must be 3 channels for EfficientNet).
        num_classes (int): Number of output classes.
        
    Returns:
        keras.Model: The classification model.
    """
    # Load base model with pre-trained ImageNet weights
    base_model = EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build custom head
    inputs = layers.Input(shape=input_shape)
    
    # EfficientNet expects [-1, 1] normalized inputs or [0, 255] depending on preprocessing.
    # We will assume inputs are [0, 255] and let EfficientNet's internal rescaling handle it,
    # or handle scaling in our pipeline. Since EfficientNetB0 in Keras includes its own
    # normalization layer, it expects inputs in [0, 255].
    x = base_model(inputs, training=False)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs, outputs, name="neuroscan_efficientnet_cls")
    return model

def unfreeze_classifier_base(model, num_layers_to_unfreeze=20):
    """
    Unfreezes the top layers of the EfficientNet base for fine-tuning.
    
    Args:
        model (keras.Model): The built classifier model.
        num_layers_to_unfreeze (int): Number of layers from the end of the base model to unfreeze.
    """
    base_model = model.layers[1] # EfficientNetB0 is usually at index 1
    base_model.trainable = True
    
    # Freeze all layers except the top 'num_layers_to_unfreeze'
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
        
    return model
