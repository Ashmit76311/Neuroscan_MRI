import os, glob, cv2, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
import tensorflow.keras.backend as K

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {gpus}')
if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)

# === AUTO-DISCOVER PATHS (works with any Kaggle mount structure) ===
def find_dir(pattern):
    results = glob.glob(pattern, recursive=True)
    return results[0] if results else None

SEG_DIR   = find_dir('/kaggle/input/**/kaggle_3m')
CLS_TRAIN = find_dir('/kaggle/input/**/brain-tumor-mri-dataset/Training') or \
            find_dir('/kaggle/input/**/Training')
CLS_TEST  = find_dir('/kaggle/input/**/brain-tumor-mri-dataset/Testing') or \
            find_dir('/kaggle/input/**/Testing')

print(f'SEG_DIR:   {SEG_DIR}')
print(f'CLS_TRAIN: {CLS_TRAIN}')
print(f'CLS_TEST:  {CLS_TEST}')
if not SEG_DIR:   raise RuntimeError('LGG kaggle_3m not found — add the LGG Segmentation dataset!')
if not CLS_TRAIN: raise RuntimeError('Brain Tumor MRI Training folder not found!')

OUT_DIR = '/kaggle/working'
CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']
IMG_SZ  = (224, 224)
BATCH   = 16
LR      = 5e-4
print('Config loaded.')

# Segmentation Data
def load_seg_pair(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SZ)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, IMG_SZ, method='nearest')
    mask = tf.cast(mask > 0.5, tf.float32)
    return img, mask

all_tifs = glob.glob(os.path.join(SEG_DIR, '**', '*.tif'), recursive=True)
img_paths, mask_paths = [], []
for p in all_tifs:
    if 'mask' not in p.lower():
        base, ext = os.path.splitext(p)
        msk = f'{base}_mask{ext}'
        if os.path.exists(msk):
            img_paths.append(p)
            mask_paths.append(msk)
print(f'Found {len(img_paths)} pairs')

os.makedirs('/kaggle/working/seg_data', exist_ok=True)
png_imgs, png_masks = [], []
for ip, mp in zip(img_paths, mask_paths):
    fn = os.path.basename(ip).replace('.tif','.png')
    fn_m = os.path.basename(mp).replace('.tif','.png')
    pp = f'/kaggle/working/seg_data/{fn}'
    pm = f'/kaggle/working/seg_data/{fn_m}'
    if not os.path.exists(pp): cv2.imwrite(pp, cv2.imread(ip, cv2.IMREAD_UNCHANGED))
    if not os.path.exists(pm): cv2.imwrite(pm, cv2.imread(mp, cv2.IMREAD_UNCHANGED))
    png_imgs.append(pp); png_masks.append(pm)

ti, vi, tm, vm = train_test_split(png_imgs, png_masks, test_size=0.15, random_state=42)
print(f'Train: {len(ti)} Val: {len(vi)}')

def make_seg_ds(imgs, masks, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    ds = ds.map(load_seg_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if augment: ds = ds.shuffle(500)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

seg_train_ds = make_seg_ds(ti, tm, True)
seg_val_ds = make_seg_ds(vi, vm)
print('Seg datasets ready.')

# U-Net Model
def conv_block(x, f):
    x = layers.SeparableConv2D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attn_gate(x, g, inter):
    tx = layers.Conv2D(inter, 1, strides=2, padding='same', use_bias=False)(x)
    pg = layers.Conv2D(inter, 1, padding='same', use_bias=False)(g)
    out = layers.Activation('relu')(layers.Add()([tx, pg]))
    psi = layers.Activation('sigmoid')(layers.Conv2D(1, 1, padding='same', use_bias=False)(out))
    psi = layers.UpSampling2D(2, interpolation='bilinear')(psi)
    return layers.Multiply()([x, psi])

def build_unet(shape=(224,224,1), filters=[16,32,64,128,256]):
    inp = layers.Input(shape=shape)
    skips, x = [], inp
    for f in filters[:-1]:
        x = conv_block(x, f); skips.append(x); x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, filters[-1])
    for f, sk in zip(reversed(filters[:-1]), reversed(skips)):
        g = x
        x = layers.UpSampling2D(2, interpolation='bilinear')(x)
        x = layers.SeparableConv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.Activation('relu')(x)
        sk = attn_gate(sk, g, f//2)
        x = conv_block(layers.Concatenate()([x, sk]), f)
    out = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    return models.Model(inp, out, name='neuroscan_unet_sepconv')

seg_model = build_unet()
print('U-Net built:', seg_model.count_params(), 'params')

# Train Segmenter
def dice_coef(y_true, y_pred, smooth=1.):
    y_t = K.flatten(y_true); y_p = K.flatten(y_pred)
    return (2.*K.sum(y_t*y_p)+smooth)/(K.sum(y_t)+K.sum(y_p)+smooth)

def bce_dice(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true,y_pred) + 1. - dice_coef(y_true,y_pred)

SEG_CKPT = '/kaggle/working/neuroscan_seg.keras'
seg_model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=bce_dice,
    metrics=['accuracy', dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)])
seg_cbs = [
    tf.keras.callbacks.ModelCheckpoint(SEG_CKPT, save_best_only=True, monitor='val_loss', verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]
print('Training U-Net...')
seg_model.fit(seg_train_ds, validation_data=seg_val_ds, epochs=50, callbacks=seg_cbs)
print('Segmentation done! Saved to', SEG_CKPT)

# Classification Data + Model + Training
def load_cls(p, label):
    img = tf.io.read_file(p)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SZ)
    return tf.cast(img, tf.float32), label

def aug_cls(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    return tf.clip_by_value(img, 0, 255), label

def make_cls_ds(root, augment=False):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        imgs = glob.glob(os.path.join(root, c, '*.jpg')) + glob.glob(os.path.join(root, c, '*.png'))
        paths += imgs; labels += [i]*len(imgs)
    print(f'{root}: {len(paths)} images')
    cats = tf.keras.utils.to_categorical(labels, len(CLASSES))
    ds = tf.data.Dataset.from_tensor_slices((paths, cats)).map(load_cls, num_parallel_calls=tf.data.AUTOTUNE)
    if augment: ds = ds.map(aug_cls, num_parallel_calls=tf.data.AUTOTUNE).shuffle(2000)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

cls_train_ds = make_cls_ds(CLS_TRAIN, True)
cls_val_ds = make_cls_ds(CLS_TEST)

base = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224,224,3))
base.trainable = False
inp = layers.Input((224,224,3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(4, activation='softmax')(x)
cls_model = models.Model(inp, out, name='neuroscan_efficientnet_cls')

CLS_CKPT = '/kaggle/working/neuroscan_cls.keras'
cls_cbs = [
    tf.keras.callbacks.ModelCheckpoint(CLS_CKPT, save_best_only=True, monitor='val_loss', verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

cls_model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
print('Training Classifier Phase 1...')
cls_model.fit(cls_train_ds, validation_data=cls_val_ds, epochs=40, callbacks=cls_cbs)

# Fine-tune
base.trainable = True
for layer in base.layers[:-20]: layer.trainable = False
cls_model.compile(optimizer=tf.keras.optimizers.Adam(LR/10), loss='categorical_crossentropy', metrics=['accuracy'])
print('Fine-tuning Phase 2...')
cls_model.fit(cls_train_ds, validation_data=cls_val_ds, epochs=20, callbacks=cls_cbs)
print('Done! Saved to', CLS_CKPT)

print('\n=== TRAINING COMPLETE ===')
for f in ['/kaggle/working/neuroscan_seg.keras', '/kaggle/working/neuroscan_cls.keras']:
    if os.path.exists(f):
        print(f'  {os.path.basename(f)}: {os.path.getsize(f)/1e6:.1f} MB')
print('Download from the Output tab on the right!')
