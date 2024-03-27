import argparse
from pathlib import Path
from imutils import paths
import shutil
import numpy as np
from tqdm import tqdm


def split_and_copy_dataset(source_folder, dest_folder1, dest_folder2, subset_size):
    """
    Splits a dataset from a source folder into two disjoint subsets of equal size and copies them into destination folders.

    Parameters:
    - source_folder (Path): Path to the source folder containing the dataset.
    - dest_folder1 (Path): Path to the destination folder for the first subset.
    - dest_folder2 (Path): Path to the destination folder for the second subset.
    - subset_size (int): The number of images in each subset.
    """
    # List all image files in the source folder
    all_files = list(paths.list_images(source_folder))
    dataset_size = len(all_files)

    if 2 * subset_size > dataset_size:
        raise ValueError("The sum of both subsets exceeds the dataset size.")

    # Shuffle the list of files
    np.random.shuffle(all_files)

    # Split the files into two subsets
    subset1_files = all_files[:subset_size]
    subset2_files = all_files[subset_size:2 * subset_size]

    # Ensure destination folders exist
    dest_folder1.mkdir(parents=True, exist_ok=True)
    dest_folder2.mkdir(parents=True, exist_ok=True)

    # Copy files to the respective destination folders
    for i, file_path in enumerate(tqdm(subset1_files)):
        shutil.copy(file_path, dest_folder1 / f"{i}_{Path(file_path).name}")
    for i, file_path in enumerate(tqdm(subset2_files)):
        shutil.copy(file_path, dest_folder2 / f"{i}_{Path(file_path).name}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Split a dataset into two disjoint subsets and copy them into new folders.")
    parser.add_argument("-s", "--source", required=True, help="Path to the source folder containing the dataset.")
    parser.add_argument("-d1", "--dest1", required=True, help="Path to the destination folder for the first subset.")
    parser.add_argument("-d2", "--dest2", required=True, help="Path to the destination folder for the second subset.")
    parser.add_argument("-n", "--size", type=int, required=True, help="The number of images in each subset.")

    args = parser.parse_args()

# Convert paths to Path objects
    source_folder = Path(args.source)
    dest_folder1 = Path(args.dest1)
    dest_folder2 = Path(args.dest2)
    subset_size = args.size

    # Run the function
    split_and_copy_dataset(source_folder, dest_folder1, dest_folder2, subset_size)