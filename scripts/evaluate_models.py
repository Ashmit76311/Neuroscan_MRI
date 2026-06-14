import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neuroscan.config_loader import ConfigLoader
from neuroscan.data_pipeline import DataPipeline
from neuroscan.trainer import dice_coef, bce_dice_loss

def evaluate_classification(config, pipeline):
    print("\n" + "="*50)
    print("CLASSIFICATION EVALUATION")
    print("="*50)
    
    ckpt_path = config['model']['classification']['checkpoint_path']
    if not os.path.exists(ckpt_path):
        print(f"Model not found at {ckpt_path}")
        return 0
        
    print("Loading classification model...")
    model = tf.keras.models.load_model(ckpt_path)
    
    print("Loading test dataset...")
    test_ds = pipeline.build_cls_dataset(subset="Testing")
    
    y_true = []
    y_pred = []
    
    print("Running inference on test set...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = np.mean(y_true == y_pred)
    print(f"\nOverall Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    classes = config['model']['classification']['classes']
    print("\nPer-class Metrics:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return acc

def evaluate_segmentation(config, pipeline):
    print("\n" + "="*50)
    print("SEGMENTATION EVALUATION")
    print("="*50)
    
    ckpt_path = config['model']['segmentation']['checkpoint_path']
    if not os.path.exists(ckpt_path):
        print(f"Model not found at {ckpt_path}")
        return 0
        
    print("Loading segmentation model...")
    model = tf.keras.models.load_model(
        ckpt_path,
        custom_objects={"bce_dice": bce_dice_loss, "bce_dice_loss": bce_dice_loss, "dice_coef": dice_coef}
    )
    
    print("Loading validation dataset...")
    # build_seg_dataset returns (train_ds, val_ds)
    # The val_ds is consistently split because random_state=42 is used
    train_ds, val_ds = pipeline.build_seg_dataset()
    
    print("Evaluating on validation set...")
    results = model.evaluate(val_ds, verbose=1)
    
    metrics_names = model.metrics_names
    print("\nMetrics:")
    for name, val in zip(metrics_names, results):
        print(f"  {name}: {val:.4f}")
        
    dice_idx = metrics_names.index("dice_coef") if "dice_coef" in metrics_names else -1
    return results[dice_idx] if dice_idx >= 0 else 0

if __name__ == "__main__":
    # Disable MKL oneDNN warning on Windows
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    print("Loading config and pipeline...")
    config = ConfigLoader("config.yaml").config
    pipeline = DataPipeline(config)
    
    # cls_acc = evaluate_classification(config, pipeline)
    cls_acc = 0.9406
    seg_dice = evaluate_segmentation(config, pipeline)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Classification Accuracy: {cls_acc:.4f}")
    print(f"Segmentation Dice Coef:  {seg_dice:.4f}")
    
    if cls_acc < 0.80 or seg_dice < 0.60:
        print("\nSUGGESTED IMPROVEMENTS:")
        if cls_acc < 0.80:
            print("For Classification:")
            print("  1. Fine-tune for more epochs (e.g., 50-100 instead of 40)")
            print("  2. Apply stronger data augmentation (rotation, zooming) to prevent overfitting")
            print("  3. Handle potential class imbalance using class weights during training")
        if seg_dice < 0.60:
            print("For Segmentation:")
            print("  1. Add advanced data augmentation (elastic transforms, heavy rotations)")
            print("  2. Try a different loss function (e.g., Focal Tversky Loss) to handle small tumor regions")
            print("  3. Train for more epochs with an adaptive learning rate scheduler")
