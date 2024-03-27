from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from imutils.paths import list_images


def resize_images(source_dir, target_dir, width, height):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    size = (width, height)

    # Collect all jpg images in the source directory
    images = [Path(x) for x in list(list_images(source_dir))]

    # Initialize tqdm progress bar
    for img_path in tqdm(images, desc="Resizing images"):
        relative_path = img_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path)
        img = img.resize(size, Image.Resampling.LANCZOS)
        img.save(target_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Resize images and maintain directory structure.")
    parser.add_argument("source_dir", type=Path, help="Path to the source directory containing the images.")
    parser.add_argument("target_dir", type=Path,
                        help="Path to the target directory where resized images will be stored.")
    parser.add_argument("--width", type=int, default=112, help="Width to which the images will be resized.")
    parser.add_argument("--height", type=int, default=112, help="Height to which the images will be resized.")

    args = parser.parse_args()

    resize_images(args.source_dir, args.target_dir, args.width, args.height)
