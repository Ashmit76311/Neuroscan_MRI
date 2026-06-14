import pytest
from neuroscan.config_loader import ConfigLoader
from neuroscan.data_pipeline import DataPipeline

@pytest.fixture
def dummy_config():
    cfg = ConfigLoader().config
    cfg['training']['segmentation']['batch_size'] = 2
    cfg['training']['classification']['batch_size'] = 2
    return cfg

def test_dummy_seg_dataset(dummy_config):
    pipeline = DataPipeline(dummy_config)
    train_ds, val_ds = pipeline._create_dummy_seg_dataset()
    
    for x, y in train_ds.take(1):
        assert x.shape == (2, 224, 224, 1)
        assert y.shape == (2, 224, 224, 1)

def test_dummy_cls_dataset(dummy_config):
    pipeline = DataPipeline(dummy_config)
    ds = pipeline._create_dummy_cls_dataset()
    
    for x, y in ds.take(1):
        assert x.shape == (2, 224, 224, 3)
        assert y.shape == (2, 4)
