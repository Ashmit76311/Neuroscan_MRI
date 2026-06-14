import tensorflow as tf
from keras import layers, models

def _conv_block(inputs, filters, kernel_size=(3, 3), padding="same"):
    """
    Creates a block with SeparableConv2D -> BatchNorm -> ReLU.
    """
    x = layers.SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_block(x, g, inter_channel):
    """
    Attention gate for U-Net.
    x: Skip connection feature map
    g: Gating signal from the next layer down
    """
    theta_x = layers.Conv2D(inter_channel, kernel_size=(1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
    phi_g = layers.Conv2D(inter_channel, kernel_size=(1, 1), padding="same", use_bias=False)(g)
    
    concat_xg = layers.Add()([theta_x, phi_g])
    act_xg = layers.Activation("relu")(concat_xg)
    
    psi = layers.Conv2D(1, kernel_size=(1, 1), padding="same", use_bias=False)(act_xg)
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    
    # Upsample the attention weights to match the skip connection size
    # Since g is half the size of x, we just upsample by 2
    upsample_psi = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(sigmoid_xg)
    
    return layers.Multiply()([x, upsample_psi])

def build_segmentation_net(input_shape=(224, 224, 1), filter_list=[16, 32, 64, 128, 256]):
    """
    Builds a U-Net architecture using SeparableConv2D and UpSampling2D.
    
    Args:
        input_shape (tuple): Shape of the input image.
        filter_list (list): Number of filters for each encoder block. The last element is the bottleneck.
        
    Returns:
        keras.Model: The compiled segmentation model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    skip_connections = []
    x = inputs
    
    for filters in filter_list[:-1]:
        x = _conv_block(x, filters)
        skip_connections.append(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
    # Bottleneck
    bottleneck_filters = filter_list[-1]
    x = _conv_block(x, bottleneck_filters)
    
    # Decoder
    decoder_filters = filter_list[:-1][::-1]
    skip_connections = skip_connections[::-1]
    
    for filters, skip in zip(decoder_filters, skip_connections):
        g = x # The gating signal from the previous layer, before upsampling
        
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Apply attention gate
        skip_attended = attention_block(x=skip, g=g, inter_channel=filters // 2)
        
        x = layers.Concatenate()([x, skip_attended])
        x = _conv_block(x, filters)
        
    # Output layer
    outputs = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs, name="neuroscan_unet_sepconv")
    return model
