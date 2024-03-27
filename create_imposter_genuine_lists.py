from pathlib import Path
import random
import argparse
from tqdm import tqdm
from imutils.paths import list_images
from itertools import combinations

def generate_genuine_pairs(dataset_path, num_pairs):
    base = Path(dataset_path)
    identities = [f for f in base.iterdir() if f.is_dir()]
    random.shuffle(identities)
    pairs_per_identity = max(1, num_pairs // len(identities))
    pairs = []
    remaining_pairs = num_pairs
    for id in tqdm(identities):
        identity_pairs = list(combinations(list(list_images(id)), 2))
        random.shuffle(identity_pairs)
        if pairs_per_identity < len(identity_pairs):
            identity_pairs = identity_pairs[:pairs_per_identity]
        remaining_pairs -= len(identity_pairs)
        pairs.extend(identity_pairs)
        if remaining_pairs <= 0:
            break
    return pairs

def generate_imposter_pairs(dataset_path, num_pairs):
    base = Path(dataset_path)
    identities = [f for f in base.iterdir() if f.is_dir()]
    random.shuffle(identities)
    pairs_per_identity = max(1, num_pairs // len(identities))
    pairs = []
    remaining_pairs = num_pairs
    for i, identity_path in enumerate(tqdm(identities)):
        identity_pairs = []
        other_identities = list(random.choices(identities[:i] + identities[i:], k=pairs_per_identity))
        other_images = [list(list_images(path)) for path in other_identities]
        other_images = [element for sublist in other_images for element in sublist]
        random.shuffle(other_images)
        other_images = other_images[:pairs_per_identity]
        own_images = list(list_images(identity_path))
        random.shuffle(own_images)
        if len(own_images) < pairs_per_identity:
            diff = pairs_per_identity - len(own_images)
            own_images += own_images[:diff]
        else:
            own_images = own_images[:pairs_per_identity]
        identity_pairs = list(zip(own_images, other_images))
        pairs.extend(identity_pairs)
        remaining_pairs -= len(identity_pairs)
        if remaining_pairs <= 0:
            break
    return pairs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate genuine and imposter pairs from a dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--num_pairs", type=int, default=10, help="Number of pairs to generate")
    parser.add_argument("--mode", choices=['genuine, imposter, both'], default='both')
    parser.add_argument("--output", type=Path, default=".", help="Folder to save the list to.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix for the list.")
    args = parser.parse_args()

    imposter = args.mode in ['imposter', 'both']
    genuine = args.mode in ['genuine', 'both']
    args.output.mkdir(parents=True, exist_ok=True)

    if genuine:
        genuine_pairs = generate_genuine_pairs(args.dataset_path, args.num_pairs)
        with open(args.output / f"{args.prefix}_genuine_pairs.txt", "w") as file:
            for g in genuine_pairs:
                file.write(f"{g[0]} {g[1]} 1\n")
    if imposter:
        imposter_pairs = generate_imposter_pairs(args.dataset_path, args.num_pairs)
        with open(args.output / f"{args.prefix}_imposter_pairs.txt", "w") as file:
            for g in imposter_pairs:
                file.write(f"{g[0]} {g[1]} 0\n")
