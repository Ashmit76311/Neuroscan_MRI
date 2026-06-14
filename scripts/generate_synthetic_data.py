import os
import cv2
import numpy as np
from pathlib import Path

def create_dirs():
    base_seg = Path("data/raw/segmentation")
    base_cls = Path("data/raw/classification")
    
    (base_seg / "images").mkdir(parents=True, exist_ok=True)
    (base_seg / "masks").mkdir(parents=True, exist_ok=True)
    
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    for c in classes:
        (base_cls / c).mkdir(parents=True, exist_ok=True)

def generate_segmentation_data(num_samples=200):
    images_dir = Path("data/raw/segmentation/images")
    masks_dir = Path("data/raw/segmentation/masks")
    
    for i in range(num_samples):
        # Create empty images
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        mask = np.zeros((224, 224), dtype=np.uint8)
        
        # Draw "brain" (large ellipse)
        center_x = np.random.randint(100, 124)
        center_y = np.random.randint(100, 124)
        axes = (np.random.randint(70, 90), np.random.randint(80, 100))
        cv2.ellipse(img, (center_x, center_y), axes, 0, 0, 360, (100, 100, 100), -1)
        
        # Add some noise to brain
        noise = np.random.normal(0, 10, (224, 224, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Zero out background again
        bg_mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.ellipse(bg_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
        img = cv2.bitwise_and(img, img, mask=bg_mask)
        
        # Draw "tumor" (smaller ellipse inside)
        t_center_x = center_x + np.random.randint(-30, 30)
        t_center_y = center_y + np.random.randint(-30, 30)
        t_axes = (np.random.randint(10, 25), np.random.randint(10, 25))
        cv2.ellipse(img, (t_center_x, t_center_y), t_axes, 0, 0, 360, (200, 200, 200), -1)
        
        # Draw tumor on mask
        cv2.ellipse(mask, (t_center_x, t_center_y), t_axes, 0, 0, 360, 255, -1)
        
        # Ensure mask is exactly strictly 255 where tumor is, 0 otherwise
        
        cv2.imwrite(str(images_dir / f"syn_{i:04d}.png"), img)
        cv2.imwrite(str(masks_dir / f"syn_{i:04d}.png"), mask)

def generate_classification_data(samples_per_class=100):
    base_cls = Path("data/raw/classification")
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    
    for c in classes:
        for i in range(samples_per_class):
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            center_x = np.random.randint(100, 124)
            center_y = np.random.randint(100, 124)
            axes = (np.random.randint(70, 90), np.random.randint(80, 100))
            cv2.ellipse(img, (center_x, center_y), axes, 0, 0, 360, (100, 100, 100), -1)
            
            # Add some texture to brain
            noise = np.random.normal(0, 10, (224, 224, 3)).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            bg_mask = np.zeros((224, 224), dtype=np.uint8)
            cv2.ellipse(bg_mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)
            img = cv2.bitwise_and(img, img, mask=bg_mask)
            
            if c != "notumor":
                # Add tumor based on class
                if c == "glioma":
                    # Diffuse, large
                    t_center = (center_x + np.random.randint(-20, 20), center_y + np.random.randint(-20, 20))
                    t_axes = (np.random.randint(20, 35), np.random.randint(20, 35))
                    color = (180, 180, 180)
                elif c == "meningioma":
                    # Edge
                    t_center = (center_x + np.random.choice([-40, 40]), center_y + np.random.choice([-40, 40]))
                    t_axes = (np.random.randint(15, 25), np.random.randint(15, 25))
                    color = (220, 220, 220)
                elif c == "pituitary":
                    # Bottom center
                    t_center = (center_x, center_y + 40 + np.random.randint(-10, 10))
                    t_axes = (np.random.randint(10, 15), np.random.randint(10, 15))
                    color = (240, 240, 240)
                    
                cv2.ellipse(img, t_center, t_axes, 0, 0, 360, color, -1)
            
            cv2.imwrite(str(base_cls / c / f"syn_{i:04d}.png"), img)

if __name__ == "__main__":
    print("Creating directories...")
    create_dirs()
    print("Generating segmentation data...")
    generate_segmentation_data()
    print("Generating classification data...")
    generate_classification_data()
    print("Done generating synthetic data!")
