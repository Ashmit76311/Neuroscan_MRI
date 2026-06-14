import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1.0 - dice_coef(y_true, y_pred)
    return bce + dice

class TrainerBase:
    def __init__(self, config):
        self.config = config
        
    def _get_callbacks(self, checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        return [
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

class SegmentationTrainer(TrainerBase):
    def __init__(self, config, model, train_ds, val_ds):
        super().__init__(config)
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        
    def train(self):
        conf = self.config['training']['segmentation']
        ckpt_path = self.config['model']['segmentation']['checkpoint_path']
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=conf['learning_rate']),
            loss=bce_dice_loss,
            metrics=["accuracy", dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)]
        )
        
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=conf['epochs'],
            callbacks=self._get_callbacks(ckpt_path)
        )
        return history

class ClassificationTrainer(TrainerBase):
    def __init__(self, config, model, train_ds, val_ds=None):
        super().__init__(config)
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        
    def train(self, phase="base"):
        """
        phase: 'base' or 'finetune'
        """
        conf = self.config['training']['classification']
        ckpt_path = self.config['model']['classification']['checkpoint_path']
        
        lr = conf['learning_rate'] if phase == "base" else conf['learning_rate'] / 10.0
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=conf['epochs'] if phase == "base" else conf['epochs'] // 2,
            callbacks=self._get_callbacks(ckpt_path)
        )
        return history
