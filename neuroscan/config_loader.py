import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class ConfigLoader:
    """Loads configuration from config.yaml and applies environment variable overrides."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_yaml()
        self._apply_env_overrides()

    def _load_yaml(self) -> dict:
        path = Path(self.config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _apply_env_overrides(self):
        """Overrides specific config values if matching env vars are set."""
        # Data paths
        if seg_dir := os.getenv("NEUROSCAN_SEG_DATA_DIR"):
            self._config['data']['segmentation_dir'] = seg_dir
        if cls_dir := os.getenv("NEUROSCAN_CLS_DATA_DIR"):
            self._config['data']['classification_dir'] = cls_dir
            
        # Checkpoints
        if seg_ckpt := os.getenv("NEUROSCAN_SEG_CKPT"):
            self._config['model']['segmentation']['checkpoint_path'] = seg_ckpt
        if cls_ckpt := os.getenv("NEUROSCAN_CLS_CKPT"):
            self._config['model']['classification']['checkpoint_path'] = cls_ckpt
            
        # Hyperparameters
        if batch_size := os.getenv("NEUROSCAN_BATCH_SIZE"):
            bs = int(batch_size)
            self._config['training']['segmentation']['batch_size'] = bs
            self._config['training']['classification']['batch_size'] = bs

    @property
    def config(self) -> dict:
        """Returns the full configuration dictionary."""
        return self._config

# Global instance for easy import
cfg_loader = ConfigLoader()
config = cfg_loader.config
