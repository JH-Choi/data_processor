import os
import cv2
from tqdm import tqdm

path = "rgbs"
path_4 = "rgbs_4"

image_names = os.listdir(path)

os.makedirs("rgbs_4", exist_ok=True)
for name in tqdm(image_names):
    image_path = os.path.join(path, name)
    if name.split(".")[-1] == "jpg" or name.split(".")[-1] == "png":
        image = cv2.imread(image_path)
        image_4 = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
        image_path_4 = os.path.join(path_4, name)
        cv2.imwrite(image_path_4, image_4)
