import os
import argparse
import tensorflow as tf

# ── Disable MKL/oneDNN optimisations that cause 'could not create a primitive'
# crashes on some Windows + TF 2.x configurations.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from neuroscan.config_loader import ConfigLoader
from neuroscan.data_pipeline import DataPipeline
from neuroscan.seg_model import build_segmentation_net
from neuroscan.trainer import SegmentationTrainer

def main():
    parser = argparse.ArgumentParser(description="Train the NeuroScan U-Net Segmentation Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    cfg_loader = ConfigLoader(args.config)
    config = cfg_loader.config

    print("Building data pipeline...")
    pipeline = DataPipeline(config)
    train_ds, val_ds = pipeline.build_seg_dataset()

    ckpt_path = config['model']['segmentation']['checkpoint_path']

    # ── Resume from saved checkpoint if one exists ───────────────────────────
    if os.path.exists(ckpt_path):
        print(f"Found existing checkpoint at {ckpt_path}. Resuming training...")
        from neuroscan.trainer import bce_dice_loss, dice_coef
        model = tf.keras.models.load_model(
            ckpt_path,
            custom_objects={"bce_dice_loss": bce_dice_loss, "dice_coef": dice_coef}
        )
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found. Building fresh segmentation model...")
        model = build_segmentation_net(
            input_shape=tuple(config['model']['segmentation']['input_shape']),
            filter_list=config['model']['segmentation']['filters']
        )
        model.summary()

    print("Initializing trainer...")
    trainer = SegmentationTrainer(config, model, train_ds, val_ds)

    print("Starting training...")
    trainer.train()

    print(f"Training complete. Model saved to {ckpt_path}")

if __name__ == "__main__":
    main()
