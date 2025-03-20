from rembg import remove
from PIL import Image

def remove_background(image_cv):
    # Remove background
    foreground = remove(image_cv)

    # Create a white background
    white_bg = Image.new("RGBA", foreground.size, (255, 255, 255, 255))

    # Composite the foreground over the white background
    final_image = Image.alpha_composite(white_bg, foreground).convert("RGB")

    return final_image