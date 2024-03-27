import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from imutils.paths import list_images
from pathlib import Path
import shutil

# Define a dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Initialize the model
model = models.resnet50(pretrained=True)
model.eval()  # Set model to evaluation mode

# Modify model to use as a feature extractor
model = torch.nn.Sequential(*(list(model.children())[:-1]))

if torch.cuda.is_available():
    model.cuda()  # Move model to GPU if available

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_embeddings(data_loader, model):
    with torch.no_grad():
        embeddings = []
        for batch in data_loader:
            if torch.cuda.is_available():
                batch = batch.cuda()
            features = model(batch).cpu().numpy()
            features = features.squeeze()
            embeddings.append(features)
        embeddings = np.vstack(embeddings)
    return embeddings

if __name__ == '__main__':
    path =r"C:\Users\Aitor\datasets\race_unbalance\dev\African"
    image_paths = list(list_images(path))
    # Assuming `image_paths` is a list of paths to your images
    dataset = ImageDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    embeddings = extract_embeddings(data_loader, model)
    print(embeddings)

    from sklearn.cluster import DBSCAN
    import numpy as np

    # Assuming `embeddings` is the numpy array of extracted embeddings
    # Example: embeddings.shape might be (100000, 2048) for 100,000 images with 2048-dimensional embeddings

    # Perform DBSCAN clustering
    # Note: Adjust eps and min_samples based on your dataset characteristics and the desired granularity of clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
    if embeddings.ndim > 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    clusters = dbscan.fit_predict(embeddings)

    # `clusters` now contains the cluster labels for each image, with -1 indicating noise (outliers)

    # Assuming `clusters` is the array of cluster labels from DBSCAN
    # And `image_paths` is your list of image paths in the same order as the embeddings

    clustered_images = {}  # Dictionary to hold cluster_id: [image_paths]

    for i, cluster_id in enumerate(clusters):
        if cluster_id not in clustered_images:
            clustered_images[cluster_id] = []
        clustered_images[cluster_id].append(image_paths[i])

    for cluster_id, imgs in clustered_images.items():
        cluster_path = Path(path) / f"{cluster_id}"
        cluster_path.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            shutil.copy(img_path, cluster_path / Path(img_path).name)

    # `clustered_images` now maps each cluster ID to the list of image paths belonging to that cluster
    # Cluster ID of -1 represents noise or outliers
