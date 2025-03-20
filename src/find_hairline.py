import numpy as np

def find_hairline(image_np):
    """Find the top-most non-white pixel (hairline)"""
    height, width, _ = image_np.shape
    for y in range(height):
        row = image_np[y, :, :]
        if np.any(row < 240):  # Any non-white pixel
            return y
    return 0  # Default to the top if no non-white pixels are found