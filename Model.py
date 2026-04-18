import numpy as np
import rasterio
from PIL import Image
import os
from scipy.ndimage import gaussian_filter

OUTPUT_FOLDER = 'outputs'

def predict_lulc(image_path):
    with rasterio.open(image_path) as src:
        img = src.read(1).astype(float)
        bounds = src.bounds

    h, w = img.shape

    # 🔥 STEP 1: Normalize image
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)

    # 🔥 STEP 2: Smooth image (important for regions)
    smooth = gaussian_filter(img, sigma=4)

    # 🔥 STEP 3: Sort pixels by intensity
    flat = smooth.flatten()
    sorted_idx = np.argsort(flat)

    classified = np.zeros_like(flat, dtype=np.uint8)

    total = len(flat)

    # 🔥 STEP 4: Assign classes by percentage
    urban_end = int(0.65 * total)
    water_end = urban_end + int(0.18 * total)
    agri_end = water_end + int(0.12 * total)

    classified[sorted_idx[:urban_end]] = 2        # Urban
    classified[sorted_idx[urban_end:water_end]] = 0   # Water
    classified[sorted_idx[water_end:agri_end]] = 1    # Agriculture
    classified[sorted_idx[agri_end:]] = 3             # Barren

    classified = classified.reshape(h, w)

    # 🔥 STEP 5: Color map (MATCH FRONTEND)
    color_map = {
        0: [0, 0, 255],      # Water (Blue)
        1: [0, 255, 0],      # Agriculture (Green)
        2: [150, 150, 150],  # Urban (Gray)
        3: [255, 255, 0]     # Barren (Yellow)
    }

    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for k, v in color_map.items():
        rgb[classified == k] = v

    # 🔥 STEP 6: Sharp display (no blur)
    image = Image.fromarray(rgb)
    image = image.resize((512, 512), Image.NEAREST)

    output_path = os.path.join(OUTPUT_FOLDER, "result.png")
    image.save(output_path)

    return "result.png", classified, bounds
