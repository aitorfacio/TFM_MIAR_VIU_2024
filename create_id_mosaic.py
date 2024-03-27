import json
import cv2
from imutils import build_montages
from argparse import ArgumentParser
from pathlib import Path

ap = ArgumentParser()
ap.add_argument("-f", "--file", type=Path, required=True)
ap.add_argument("-i", "--identity", type=int, required=True)
ap.add_argument("-o", "--output", type=Path)
args = ap.parse_args()
labelID = args.identity

with open(args.file, 'r') as file:
    data = json.load(file)

    files = [Path(x) for x in data[str(args.identity)]]
    faces = []
    for f in files:
        if "112" not in f.parts[-2]:
            parent = f.parent
            parent = parent.with_name(parent.name + "_112x112")
            f = parent / f.name
        image = cv2.imread(str(f))
        face = cv2.resize(image, (96,96))
        faces.append(face)

    montage = build_montages(faces, (96, 96), (5, 5))[0]

    title = "Face ID #{}".format(labelID)
    title = "Unknown Faces" if labelID == -1 else title
    cv2.imshow(title, montage)
    cv2.waitKey(0)
    #cv2.imwrite(f"{str(args.output)}/mosaic_{args.identity}.jpg", montage)
    cv2.imwrite(str(args.output), montage)
