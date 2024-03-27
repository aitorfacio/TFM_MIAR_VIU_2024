import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
from imutils import paths
import shutil



def organize_dataset_by_identity(source, destination, copy=False):
    limit = 0
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for img in tqdm(paths.list_images(source)):
        #limit += 1
        #if limit >= 10:
        #    break
        img = Path(img)
        ethnicity = Path(img.parents[1]).name
        identity = Path(img.parents[0]).name
        destination_dir = destination / ethnicity
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{identity}_{img.name}"
        if not copy:
            destination_path.symlink_to(img)
        else:
            shutil.copyfile(img, destination_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize dataset by identity")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("destination_dir", type=str, help="Path to the destination directory")
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    # Organize dataset by identity
    organize_dataset_by_identity(args.dataset_dir, args.destination_dir, args.copy)
