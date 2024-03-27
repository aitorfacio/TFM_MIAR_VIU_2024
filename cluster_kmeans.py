import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import json

ap = ArgumentParser()
ap.add_argument('-f', '--file', type=Path)
ap.add_argument('-n', '--n_clusters', type=int)
ap.add_argument('-o','--output', type=Path)
args = ap.parse_args()



# Step 1: Load your encodings
with open(args.file, 'rb') as f:
    data = pickle.load(f)

# Step 2: Prepare the data
encodings = [d['encoding'] for d in data]  # Adjust if 'encoding' is not directly accessible
encodings = np.array(encodings)

# Verify the shape of your encodings array
print("Shape of encodings array:", encodings.shape)

# Step 3: Apply PCA
# Choose the number of components such that a desired variance is retained
pca = PCA(n_components=0.95)  # retains 95% of variance
encodings_reduced = pca.fit_transform(encodings)

print("Shape of reduced encodings array:", encodings_reduced.shape)

# Step 4: Cluster with K-Means
# You need to choose a suitable number of clusters (n_clusters)
n_clusters = args.n_clusters  # Example, adjust based on your needs
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(encodings_reduced)

# Get cluster labels for each encoding
cluster_labels = kmeans.labels_

# Optionally, you can map these cluster labels back to your images
for i, label in enumerate(cluster_labels):
    data[i]['cluster_label'] = label
# You can now use `data` with 'cluster_label' added to each item for further analysis or visualization
import json

# Assuming `data` contains your images' information and `cluster_labels` from the KMeans clustering
#clusters = {str(x): [y for y in data if y['cluster_label'] == y] for x in cluster_labels}
clusters = {}
for d in data:
    label = str(d['cluster_label'])
    image_path = d['imagePath']
    if label not in clusters:
        clusters[label] = [image_path]
    else:
        clusters[label].append(image_path)

# Specify your desired JSON file path
json_file_path = args.output

# Save the clusters dictionary as a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(clusters, json_file, indent=4)

print(f"Clusters saved to {json_file_path}")
