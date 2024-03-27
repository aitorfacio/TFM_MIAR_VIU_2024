import json
from pathlib import Path, PureWindowsPath
import argparse
import os
from tqdm import tqdm

def windows_to_linux_path(the_path):
    windows_path = PureWindowsPath(the_path)
    drive = windows_path.drive.lower().replace(':','')
    rest_path = '/'.join(windows_path.parts[1:])
    linux_path = Path(f'/mnt/{drive}') / rest_path
    return linux_path

# Set up argument parsing
parser = argparse.ArgumentParser(description="Create folders for each identity and link their respective images.")
parser.add_argument("json_file", type=Path, help="Path to the JSON file with identity and images mapping.")
parser.add_argument("root_directory", type=Path, help="Root directory where identity folders will be created.")
parser.add_argument("--count", action="store_true")

args = parser.parse_args()

# Ensure the JSON file exists
if not args.json_file.exists():
    raise FileNotFoundError(f"The JSON file {args.json_file} does not exist.")

# Read the JSON file
with args.json_file.open('r') as json_file:
    identity_mapping = json.load(json_file)

count_files = 0
count_identities = 0
# Iterate through the identity mapping
for identity_id, image_paths in tqdm(identity_mapping.items()):
    count_identities += 1
    # Create a directory for the identity if it doesn't exist
    identity_directory = args.root_directory / identity_id
    identity_directory.mkdir(parents=True, exist_ok=True)

    # Create symbolic links for each image in the identity directory
    for image_path in image_paths:
        original_image_path = Path(image_path).resolve()
        #original_image_path = windows_to_linux_path(original_image_path)
        link_name = identity_directory / original_image_path.name

        if not args.count:
            # Check if the link or file already exists to avoid errors
            if not link_name.exists():
                os.symlink(original_image_path, link_name)
        else:
            count_files += 1
if args.count:
    print(f"There are {count_files} files to be linked in {count_identities}")

print("Folders and links have been created successfully.")
