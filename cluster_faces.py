# import the necessary packages
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import argparse
import pickle
import cv2
import json
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of parallel jobs to run (-1 will use all CPUs)")
ap.add_argument("-m", "--metrics", type=str, default="euclidean", help="# of parallel jobs to run (-1 will use all CPUs)")
ap.add_argument("--eps", type=float, default=0.5, help="# of parallel jobs to run (-1 will use all CPUs)")
ap.add_argument("-s", "--samples", type=int, default=2, help="# of parallel jobs to run (-1 will use all CPUs)")
args = vars(ap.parse_args())
# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]
# cluster the embeddings
print("[INFO] clustering...")
clt = DBSCAN(metric=args["metrics"], n_jobs=args["jobs"], eps=args["eps"], min_samples=args["samples"])
clt.fit(encodings)
# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))
# loop over the unique face integers
identities = {}
for labelID in labelIDs:
    # find all indexes into the `data` array that belong to the
    # current label ID, then randomly sample a maximum of 25 indexes
    # from the set
    print("[INFO] faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    #idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)
    # initialize the list of faces to include in the montage
    faces = []
    # loop over the sampled indexes
    for i in idxs:
        faces.append(data[i]["imagePath"])
    identities[str(labelID)] = faces

with open("cluster_identities.json", "w") as f:
    json.dump(identities, f, indent=4)
        ## load the input image and extract the face ROI
        #image = cv2.imread(data[i]["imagePath"])
        #(top, right, bottom, left) = data[i]["loc"]
        #face = image[top:bottom, left:right]
        ## force resize the face ROI to 96x96 and then add it to the
        ## faces montage list
        #face = cv2.resize(face, (96, 96))
        #faces.append(face)
        ## create a montage using 96x96 "tiles" with 5 rows and 5 columns
        #montage = build_montages(faces, (96, 96), (5, 5))[0]


        # show the output montage
    #title = "Face ID #{}".format(labelID)
    #title = "Unknown Faces" if labelID == -1 else title
    #cv2.imshow(title, montage)
    #cv2.waitKey(0)