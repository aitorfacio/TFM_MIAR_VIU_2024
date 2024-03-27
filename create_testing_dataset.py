import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def create_testing_dataset(sample_dataset_path, complete_dataset_path, testing_dataset_path, num_testing_identities):
    sample_dataset_path = Path(sample_dataset_path)
    complete_dataset_path = Path(complete_dataset_path)
    testing_dataset_path = Path(testing_dataset_path)

    # Step 1: Identify unique identities in the sample dataset
    sample_identities = [folder.name for folder in sample_dataset_path.iterdir() if folder.is_dir()]

    # Step 2: List all identities in the complete dataset
    complete_identities = [folder.name for folder in complete_dataset_path.iterdir() if folder.is_dir()]

    # Step 3: Remove identities present in the sample dataset
    remaining_identities = [identity for identity in complete_identities if identity not in sample_identities]

    # Step 4: Select random identities for testing dataset
    random_testing_identities = random.sample(remaining_identities, num_testing_identities)

    # Create testing dataset directory if it doesn't exist
    testing_dataset_path.mkdir(parents=True, exist_ok=True)

    # Copy selected identities to testing dataset directory
    for identity in tqdm(random_testing_identities):
        src_path = complete_dataset_path / identity
        dst_path = testing_dataset_path / identity
        shutil.copytree(src_path, dst_path)

    print("Testing dataset created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a testing dataset with random identities not present in the sample dataset.")
    parser.add_argument("--sample_dataset_path", required=True, type=str, help="Path to the sample dataset directory")
    parser.add_argument("--complete_dataset_path", required=True, type=str, help="Path to the complete dataset directory")
    parser.add_argument("--testing_dataset_path", required=True, type=str, help="Path to the testing dataset directory")
    parser.add_argument("--num_testing_identities", type=int, default=10, help="Number of random identities for the testing dataset")
    args = parser.parse_args()

    create_testing_dataset(args.sample_dataset_path, args.complete_dataset_path, args.testing_dataset_path, args.num_testing_identities)
