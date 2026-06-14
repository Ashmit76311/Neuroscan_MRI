import os
import glob
import cv2

def convert_tifs_to_png(directory):
    tif_files = glob.glob(os.path.join(directory, "**", "*.tif"), recursive=True)
    print(f"Found {len(tif_files)} .tif files. Converting to .png...")
    
    for i, tif_path in enumerate(tif_files):
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            png_path = os.path.splitext(tif_path)[0] + ".png"
            cv2.imwrite(png_path, img)
            os.remove(tif_path) # Remove the original to save space
            
        if (i + 1) % 500 == 0:
            print(f"Converted {i + 1}/{len(tif_files)} files...")
            
    print("Conversion complete.")

if __name__ == "__main__":
    convert_tifs_to_png(r"d:\Projects\Brain Tumor\Brain_Tumor_Segmentation_and_Classification_using_U-NET_and_CNN\data\raw\segmentation")
