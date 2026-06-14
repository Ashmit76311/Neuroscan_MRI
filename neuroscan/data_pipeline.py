import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.seg_input_shape = self.config['model']['segmentation']['input_shape']
        self.cls_input_shape = self.config['model']['classification']['input_shape']
        self.classes = self.config['model']['classification']['classes']

    def _load_img_mask_pair(self, img_path, mask_path):
        """Loads and processes an image and its corresponding mask."""
        # Read image
        img = tf.io.read_file(img_path)
        img = tf.io.decode_image(img, channels=1, expand_animations=False) # Grayscale for U-Net
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (self.seg_input_shape[0], self.seg_input_shape[1]))
        
        # Read mask
        mask = tf.io.read_file(mask_path)
        mask = tf.io.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        mask = tf.image.resize(mask, (self.seg_input_shape[0], self.seg_input_shape[1]), method='nearest')
        mask = tf.cast(mask > 0.5, tf.float32) # Ensure binary
        
        return img, mask

    def build_seg_dataset(self, subset="Training"):
        """
        Builds tf.data.Dataset for segmentation.
        Assumes structure: seg_dir/subset/class_name/{images, masks}/*
        or similar, logic may need adjustment based on exact dataset layout.
        For simplicity, assuming flat structure or single folder if not specified.
        """
        # Let's write a generalized loader based on the typical structure
        # original structure was: Segmentation/{Class}/ images and masks or similar
        seg_dir = self.config['data']['segmentation_dir']
        
        image_paths = []
        mask_paths = []
        
        # Assuming images have '_mask' in their name or are in separate folders
        # For this skeleton, we assume globbing all images and trying to find matching masks
        # Since we don't have the dataset locally to introspect, we'll provide a placeholder
        # logic that can be adapted.
        
        # In a real scenario, this logic maps the raw paths to pairs.
        all_imgs = glob.glob(os.path.join(seg_dir, "**", "*.png"), recursive=True) + \
                   glob.glob(os.path.join(seg_dir, "**", "*.jpg"), recursive=True) + \
                   glob.glob(os.path.join(seg_dir, "**", "*.tif"), recursive=True)
                   
        # Naive matching: if file is not a mask, try to find a mask version
        for img_p in all_imgs:
            if "mask" not in img_p.lower():
                # Attempt to find mask
                base, ext = os.path.splitext(img_p)
                mask_p = f"{base}_mask{ext}"
                if os.path.exists(mask_p):
                    image_paths.append(img_p)
                    mask_paths.append(mask_p)
                    
        # If no pairs found, just create dummy data for testing
        if not image_paths:
            print("WARNING: No segmentation pairs found. Using dummy dataset for testing.")
            return self._create_dummy_seg_dataset()

        # Split into train/val
        img_train, img_val, mask_train, mask_val = train_test_split(
            image_paths, mask_paths, 
            test_size=self.config['training']['segmentation']['validation_split'],
            random_state=42
        )

        train_ds = tf.data.Dataset.from_tensor_slices((img_train, mask_train))
        train_ds = train_ds.map(self._load_img_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(1000).batch(self.config['training']['segmentation']['batch_size']).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((img_val, mask_val))
        val_ds = val_ds.map(self._load_img_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.config['training']['segmentation']['batch_size']).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    def _load_cls_image(self, img_path, label):
        """Loads and processes an image for classification."""
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3) # RGB for EfficientNet
        img = tf.image.resize(img, (self.cls_input_shape[0], self.cls_input_shape[1]))
        
        # Note: EfficientNetB0 expects inputs in [0, 255] and handles normalization internally
        # However, to be safe with standard tf.data practices, we often just pass it as float32
        # We will keep it as 0-255 float32
        img = tf.cast(img, tf.float32)
        
        return img, label

    def _augment_cls(self, img, label):
        """Applies data augmentation for classification."""
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        return img, label

    def build_cls_dataset(self, subset="Training"):
        """Builds tf.data.Dataset for classification."""
        cls_dir = os.path.join(self.config['data']['classification_dir'], subset)
        
        image_paths = []
        labels = []
        
        for idx, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(cls_dir, cls_name)
            if os.path.exists(cls_folder):
                imgs = glob.glob(os.path.join(cls_folder, "*.png")) + \
                       glob.glob(os.path.join(cls_folder, "*.jpg"))
                image_paths.extend(imgs)
                labels.extend([idx] * len(imgs))
                
        if not image_paths:
            print(f"WARNING: No classification images found in {cls_dir}. Using dummy dataset.")
            return self._create_dummy_cls_dataset()
            
        # Convert labels to categorical
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(self.classes))
        
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(self._load_cls_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        if subset == "Training" and self.config['training']['classification']['data_augmentation']:
            ds = ds.map(self._augment_cls, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.shuffle(1000)
            
        ds = ds.batch(self.config['training']['classification']['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        return ds
        
    def _create_dummy_seg_dataset(self):
        bs = self.config['training']['segmentation']['batch_size']
        h, w, c = self.seg_input_shape
        x = np.random.rand(bs * 2, h, w, c).astype(np.float32)
        y = np.random.randint(0, 2, size=(bs * 2, h, w, 1)).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(bs)
        return ds, ds
        
    def _create_dummy_cls_dataset(self):
        bs = self.config['training']['classification']['batch_size']
        h, w, c = self.cls_input_shape
        x = np.random.rand(bs * 2, h, w, c).astype(np.float32) * 255.0
        y = tf.keras.utils.to_categorical(np.random.randint(0, len(self.classes), size=(bs * 2,)), num_classes=len(self.classes))
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(bs)
        return ds
