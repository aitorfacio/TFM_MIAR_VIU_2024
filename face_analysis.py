from deepface import DeepFace
import json
import cv2
from imutils.paths import list_images
from tqdm import tqdm

img_folder = r"C:\Users\Aitor\datasets\race_unbalance\African_112x112"
detected_race = {}

for img_path in tqdm(list_images(img_folder)):
    try:
        obj = DeepFace.analyze(img_path, actions=["race"], detector_backend='retinaface')
        detected_race[img_path] = obj[0]["dominant_race"]
    except ValueError:
        detected_race[img_path] = obj[0]["dominant_race"]

print(detected_race)
