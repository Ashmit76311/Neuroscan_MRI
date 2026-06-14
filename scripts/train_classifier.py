import argparse
from neuroscan.config_loader import ConfigLoader
from neuroscan.data_pipeline import DataPipeline
from neuroscan.cls_model import build_classifier_net, unfreeze_classifier_base
from neuroscan.trainer import ClassificationTrainer

def main():
    parser = argparse.ArgumentParser(description="Train the NeuroScan EfficientNet Classification Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--finetune-only", action="store_true", help="Skip base training and only fine-tune")
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    cfg_loader = ConfigLoader(args.config)
    config = cfg_loader.config

    print("Building data pipeline...")
    pipeline = DataPipeline(config)
    train_ds = pipeline.build_cls_dataset(subset="Training")
    val_ds = pipeline.build_cls_dataset(subset="Testing") # Assuming 'Testing' is used for validation during training here

    print("Building classification model...")
    model = build_classifier_net(
        input_shape=tuple(config['model']['classification']['input_shape']),
        num_classes=len(config['model']['classification']['classes'])
    )
    model.summary()

    trainer = ClassificationTrainer(config, model, train_ds, val_ds)

    if not args.finetune_only:
        print("Starting Phase 1: Training top layers (base frozen)...")
        trainer.train(phase="base")
        print("Phase 1 complete.")

    print("Unfreezing base model for fine-tuning...")
    model = unfreeze_classifier_base(model, num_layers_to_unfreeze=20)
    model.summary()

    print("Starting Phase 2: Fine-tuning...")
    trainer.train(phase="finetune")

    print(f"Training complete. Model saved to {config['model']['classification']['checkpoint_path']}")

if __name__ == "__main__":
    main()
