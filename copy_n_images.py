import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from imutils.paths import list_images

def select_and_copy_images(source_folder, dest_folder, num_images):
    # Convert source_folder and dest_folder to Path objects
    source_folder_path = Path(source_folder)
    dest_folder_path = Path(dest_folder)
    dest_folder_path.mkdir(parents=True, exist_ok=True)

    # List all files in the source folder
    #files = [file for file in source_folder_path.iterdir() if file.is_file()]
    files = list(list_images(source_folder_path))
    print(len(files))

    # Select num_images randomly from the files list
    selected_files = random.sample(files, min(num_images, len(files)))

    # Copy selected files to the destination folder
    for i, file in enumerate(tqdm(selected_files)):
        dest_path = dest_folder_path / f'{i}_{Path(file).name}'
        shutil.copy(file, dest_path)

    print(f"{len(selected_files)} images copied from {source_folder} to {dest_folder}.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select and copy images from source folder to destination folder.")
    parser.add_argument("source_folder", type=str, help="Path to the source folder containing images.")
    parser.add_argument("dest_folder", type=str, help="Path to the destination folder to copy the images.")
    parser.add_argument("num_images", type=int, help="Number of images to select and copy.")
    args = parser.parse_args()

    # Call the function to select and copy images
    select_and_copy_images(args.source_folder, args.dest_folder, args.num_images)
