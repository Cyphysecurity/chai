from PIL import Image
import numpy as np
import os
import json
def crop_image(image: Image.Image, crop_box: tuple) -> Image.Image:
    """
    Crop the given image to the specified box.

    Parameters:
    image (PIL.Image.Image): The input image to be cropped.
    crop_box (tuple): A tuple (left, upper, right, lower) defining the crop box.

    Returns:
    PIL.Image.Image: The cropped image.
    """
    return image.crop(crop_box)

def main():
    # Example usage
    folder = "assets/training/"
    with(open(folder + "cars_json.json", "r")) as f:
        data = json.load(f)
    for entry in data:
        
        img_path = os.path.join(folder, entry["image"])
        bx = entry["box"]
        crop_box = (bx["xmin"], bx["ymin"], bx["xmax"], bx["ymax"])
        img = Image.open(img_path)
        cropped_img = crop_image(img, crop_box)
        cropped_img.show()  # Display the cropped image
        cropped_img.save(folder + "baseline/" + entry["image"])  # Save the cropped image

if __name__ == "__main__":
    main()