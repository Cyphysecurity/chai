from PIL import Image
import numpy as np
import os
import json


def main():
    # Example usage
    folder = "assets/training/"
    folder_bs = "assets/training/baseline/"
    with(open(folder + "cars_json.json", "r")) as f:
        data = json.load(f)
    for entry in data:
        
        img_path = os.path.join(folder, entry["image"])
        img_path_att = os.path.join(folder_bs, "cr_"+entry["image"])
        bx = entry["box"]
        
        img = Image.open(img_path)
        img_att = Image.open(img_path_att)

        img.paste(img_att, (bx["xmin"], bx["ymin"], bx["xmax"], bx["ymax"]))
        img.save(folder + "baseline/" + entry["image"])  # Save the cropped image
        

if __name__ == "__main__":
    main()