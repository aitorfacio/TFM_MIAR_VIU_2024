import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from imutils.paths import list_images
import pickle
from pathlib import Path


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        root_A = Path(root_A)
        root_B = Path(root_B)
        print(f"reading {root_A}")
        if root_A.is_dir():
            self.files_A = sorted(list_images(root_A))
            with open(f"pickle_{root_A.name}.pkl", 'wb') as file:
                pickle.dump(self.files_A, file)
        elif root_A.is_file():
            with open(root_A, 'rb') as file:
                self.files_A = pickle.load(file)

        print(f"reading {root_B}")
        if root_B.is_dir():
            self.files_B = sorted(list_images(root_B))
            with open(f"pickle_{root_B.name}.pkl", 'wb') as file:
                pickle.dump(self.files_B, file)
        elif root_B.is_file():
            with open(root_B, 'rb') as file:
                self.files_B = pickle.load(file)
        self.max_images = 10000
        self.files_A = self.files_A[:self.max_images]
        self.files_B = self.files_B[:self.max_images]
        #self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        #self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
