import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from models import GeneratorResNet
from torch.autograd import Variable
from shutil import copy2
from tqdm import tqdm

def denorm(tensor):
    """Denormalizes a tensor from [-1,1] to [0,1]."""
    return tensor.mul(0.5).add(0.5)

def transform_image(image_path, model_AB, model_BA, direction, output_path):
    image = Image.open(str(image_path))
    transform = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #transform = transforms.Compose([
    #    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    #    transforms.RandomCrop((opt.img_height, opt.img_width)),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #])
    image_tensor = Variable(transform(image).unsqueeze(0).cuda())

    if direction in ['AtoB', 'both']:
        transformed_image = model_AB(image_tensor)
        save_image(image_path, output_path, transformed_image, "AtoB")
    if direction in ['BtoA', 'both']:
        transformed_image = model_BA(image_tensor)
        save_image(image_path, output_path, transformed_image, "BtoA")

    #dest_img_path = output_path / "original.jpg"
    #copy2(image_path, dest_img_path)


def save_image(image_path, output_path, transformed_image, suffix=""):
    output_path /= image_path.stem + suffix + image_path.suffix
    transformed_image = denorm(transformed_image)
    output_image = transforms.ToPILImage()(transformed_image.squeeze(0).cpu())
    output_image.save(output_path)


def transform_images_in_folder(input_folder, output_folder, model_AB, model_BA, direction):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for image_file in input_folder.glob('*'):
        output_path = output_folder
        transform_image(image_file, model_AB, model_BA, direction, output_path)


def transform_identities(identity_folders, output_folders, model_AB, model_BA, direction):
    for input_folder, output_folder in tqdm(zip(identity_folders, output_folders)):
        transform_images_in_folder(input_folder, output_folder, model_AB, model_BA, direction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=Path, help="Path to input image")
    parser.add_argument("--input_folder", type=Path, help="Path to input folder")
    parser.add_argument("--identity_folders", nargs='+', help="Paths to identity folders")
    parser.add_argument("--identities_file", type=Path)
    parser.add_argument("--output_folder", type=Path, help="Path to output folder")
    parser.add_argument("--G_AB", type=Path, required=True, help="Path to G_AB model checkpoint")
    parser.add_argument("--G_BA", type=Path, required=True, help="Path to G_BA model checkpoint")
    parser.add_argument("--direction", type=str, choices=['AtoB', 'BtoA', 'both'], default='both',
                        help="Transformation direction")
    parser.add_argument("--img_height", type=int, default=256, help="Size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="Size of image width")
    parser.add_argument("--channels", type=int, default=3, help="Channels")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    opt = parser.parse_args()

    opt.output_folder.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    input_shape = (opt.channels, opt.img_height, opt.img_width)
    # Load CycleGAN models
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
    G_AB.load_state_dict(torch.load(opt.G_AB))
    G_BA.load_state_dict(torch.load(opt.G_BA))
    G_AB.eval()
    G_BA.eval()

    if opt.identities_file:
        with open(opt.identities_file, 'r') as id_file:
            opt.identity_folders = [Path(x.strip()) for x in id_file.readlines()]

    if opt.input_image:
        transform_image(opt.input_image, G_AB, G_BA, opt.direction, opt.output_folder)
    elif opt.input_folder:
        transform_images_in_folder(opt.input_folder, opt.output_folder, G_AB, G_BA, opt.direction)
    elif opt.identity_folders:
        transform_identities(opt.identity_folders,[ opt.output_folder / x.name for x in opt.identity_folders], G_AB, G_BA, opt.direction)
    else:
        print("Please provide input image, input folder, or identity folders.")
