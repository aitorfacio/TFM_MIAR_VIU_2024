import random
import shutil
import argparse
from pathlib import Path


def create_random_subset(source_folder, target_folder, num_images):
    source_path = Path(source_folder)
    target_path = Path(target_folder)

    # List all files in the source img
    image_files = [f for f in source_path.iterdir() if f.is_symlink()]

    # Choose a random subset of symbolic links
    random_links = random.sample(image_files, min(num_images, len(image_files)))

    # Create target img if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Create symbolic links in the target img
    for link in random_links:
        target = link.resolve()  # Resolve to get the target of the symbolic link
        link_name = target_path / link.name
        link_name.symlink_to(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a random subset of symbolic links to images.')
    parser.add_argument('source_folder', type=str,
                        help='Path to the source img containing symbolic links to image files')
    parser.add_argument('target_folder', type=str,
                        help='Path to the target img where symbolic links will be created')
    parser.add_argument('num_images', type=int, help='Number of symbolic links to create in the subset')
    args = parser.parse_args()

    create_random_subset(args.source_folder, args.target_folder, args.num_images)
