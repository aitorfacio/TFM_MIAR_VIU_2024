from deepface import DeepFace
from imutils.paths import list_images
from pathlib import Path
import pandas as pd
from sklearn.cluster import DBSCAN
import face_recognition
from tqdm import tqdm


def do_it(path):
    images = list(list_images(path))
    images = images[:100]
    embeddings = []
    for img in tqdm(images):
        #embedding_objs = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=False)
        #for embedding_obj in embedding_objs:
        #    embedding = embedding_obj["embedding"]
        #    embeddings.append([img, embedding])
        known_image = face_recognition.load_image_file(img)
        embedding = face_recognition.face_encodings(known_image)
        try:
            embeddings.append([img, embedding[0]])
        except IndexError:
            print(embedding)

    return embeddings


if __name__ == '__main__':
    path = r"C:\Users\Aitor\datasets\race_unbalance\dev\African"
    instances = do_it(path)
    df = pd.DataFrame(instances, columns=["file_name","embedding"])
    clustering = DBSCAN(eps=100, min_samples=2).fit(df["embedding"].values.tolist())
    print(clustering.labels_)
