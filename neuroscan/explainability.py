import numpy as np
import cv2
import tensorflow as tf


def _find_last_conv_layer(model):
    """
    Scans a model (non-recursively) for the name of the last Conv2D,
    DepthwiseConv2D or SeparableConv2D layer.
    """
    last_conv_name = None
    for layer in model.layers:
        if isinstance(layer, (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.SeparableConv2D,
        )):
            last_conv_name = layer.name
    return last_conv_name


def generate_gradcam_heatmap(model, img_array, target_class_idx=None,
                              last_conv_layer_name=None):
    """
    Generates a Grad-CAM heatmap compatible with Keras 3 nested models
    (e.g. a custom head wrapping EfficientNetV2B0).

    Strategy
    --------
    For a model that contains a nested sub-model (the backbone):
    1. Build a *flat* feature extractor:  backbone.input → last_conv.output
       This is safe because both tensors live in the backbone's own graph.
    2. Inside a single GradientTape context:
       a. Run the feature extractor to get `conv_out`.
       b. Call ``tape.watch(conv_out)`` to treat it as a leaf variable.
       c. Run the remaining head layers *on top of* ``conv_out`` to get
          predictions.
    3. Compute ``d(class_score)/d(conv_out)`` — the classic Grad-CAM gradient.

    For flat models (no nested sub-model), falls back to building the
    standard two-output Functional model directly.

    Args:
        model             : The classification Keras model.
        img_array         : Input batch, shape (1, H, W, C), dtype float32.
        target_class_idx  : Class index for Grad-CAM. Uses top prediction if None.
        last_conv_layer_name: Override auto-detection of the last conv layer.

    Returns:
        np.ndarray: Normalised 2-D heatmap, values in [0, 1].
    """
    img_tensor = tf.cast(img_array, tf.float32)

    # ── Detect whether the model wraps a sub-model (e.g. EfficientNet) ───────
    backbone = None
    head_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
        elif not isinstance(layer, tf.keras.layers.InputLayer):
            head_layers.append(layer)

    # ── Case A: nested backbone (the common NeuroScan architecture) ──────────
    if backbone is not None:
        # Find target layer inside the backbone
        conv_name = last_conv_layer_name or _find_last_conv_layer(backbone)
        if conv_name is None:
            raise ValueError("No Conv2D layer found inside the backbone sub-model.")

        target_layer = backbone.get_layer(conv_name)

        # Build a flat extractor entirely within the backbone's own graph
        feature_extractor = tf.keras.Model(
            inputs=backbone.input,
            outputs=target_layer.output,
            name="gradcam_feature_extractor",
        )

        with tf.GradientTape() as tape:
            # Forward pass through the backbone up to the chosen conv layer
            conv_out = feature_extractor(img_tensor, training=False)

            # Watch conv_out so the tape tracks operations applied to it
            tape.watch(conv_out)

            # Forward pass through the head layers (on top of conv_out)
            x = conv_out
            for layer in head_layers:
                if isinstance(layer, tf.keras.layers.Dense) and layer == head_layers[-1]:
                    # Manually compute dense layer without activation (softmax) to get raw logits
                    # Softmax causes gradient saturation when confidence is ~100%, leading to empty heatmaps.
                    w, b = layer.get_weights()
                    x = tf.matmul(x, w) + b
                else:
                    x = layer(x, training=False)
            preds = x  # shape: (1, num_classes)

            if target_class_idx is None:
                target_class_idx = int(tf.argmax(preds[0]))
            class_score = preds[:, target_class_idx]

        # d(class_score) / d(conv_out)
        grads = tape.gradient(class_score, conv_out)

    # ── Case B: flat model (no nested sub-model) ─────────────────────────────
    else:
        conv_name = last_conv_layer_name or _find_last_conv_layer(model)
        if conv_name is None:
            raise ValueError("No Conv2D layer found in the model.")

        target_layer = model.get_layer(conv_name)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_tensor, training=False)
            tape.watch(conv_out)
            if target_class_idx is None:
                target_class_idx = int(tf.argmax(preds[0]))
            class_score = preds[:, target_class_idx]

        grads = tape.gradient(class_score, conv_out)

    # ── Compute the heatmap from gradients ────────────────────────────────────
    if grads is None:
        raise ValueError(
            "Gradient computation returned None. Ensure the model is not compiled "
            "with `run_eagerly=False` or that inputs pass through the target layer."
        )

    # Global-average-pool the gradients over the spatial dimensions:
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # Weighted combination of feature maps
    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                          # (H, W)

    # ReLU + normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def superimpose_heatmap(img_rgb, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap onto an RGB image.

    Args:
        img_rgb  : Original image, shape (H, W, 3), dtype uint8, values [0, 255].
        heatmap  : 2-D Grad-CAM heatmap, values in [0, 1].
        alpha    : Opacity of the heatmap layer.
        colormap : OpenCV colormap (default: COLORMAP_JET).

    Returns:
        np.ndarray: Blended image, shape (H, W, 3), dtype uint8.
    """
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
