import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse


def copy_random_identities(source_folder, destination_folder, fraction=0.5):
    source_folder = Path(source_folder)
    destination_folder = Path(destination_folder)

    # Iterate over ethnicities
    for ethnicity_folder in tqdm(source_folder.glob('*'), desc='Ethnicities'):
        if not ethnicity_folder.is_dir():
            continue

        ethnicity_destination_folder = destination_folder / ethnicity_folder.name
        ethnicity_destination_folder.mkdir(parents=True, exist_ok=True)

        # Iterate over identities
        for identity_folder in tqdm(ethnicity_folder.glob('*'), desc='Identities', leave=False):
            if not identity_folder.is_dir():
                continue

            # Randomly decide whether to copy this identity
            if random.random() < fraction:
                # Copy all images in this identity folder
                identity_destination_folder = ethnicity_destination_folder / identity_folder.name
                identity_destination_folder.mkdir(exist_ok=True)

                for image_file in identity_folder.glob('*'):
                    shutil.copy(str(image_file), str(identity_destination_folder))


def main():
    parser = argparse.ArgumentParser(description='Copy random identities from source folder to destination folder.')
    parser.add_argument('source_folder', help='Path to the source folder')
    parser.add_argument('destination_folder', help='Path to the destination folder')
    parser.add_argument('--fraction', type=float, default=0.5, help='Fraction of identities to copy (default: 0.5)')
    args = parser.parse_args()

    copy_random_identities(args.source_folder, args.destination_folder, args.fraction)


if __name__ == "__main__":
    main()
