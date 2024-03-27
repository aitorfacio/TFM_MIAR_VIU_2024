import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import random
from string import ascii_letters, digits

def create_next_subfolder(path):
    created_path = Path(path)
    created_path.mkdir(exist_ok=True, parents=True)
    chk_existing_folders = [int(p.name) for p in created_path.iterdir()]
    chk_next_folder_num = max(chk_existing_folders) + 1 if chk_existing_folders else 1
    chk_current_session_dir = created_path / str(chk_next_folder_num)
    chk_current_session_dir.mkdir(parents=True, exist_ok=True)
    created_path = chk_current_session_dir
    return created_path


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--mode", type=str, choices=['train', 'generate'], default='train', help="Mode of operation: train or generate")
parser.add_argument("--generator_path", type=str, default="generator.pth", help="Path to the generator model file")
parser.add_argument("--target_path", type=str, default=".", help="Parent path to save the results to." )
parser.add_argument("--output", type=str, help="Exact path to save the results to." )
parser.add_argument("--number_of_generated_images", type=int, default=10)
parser.add_argument("--save_every", type=int, default=10, help="Save checkpoints every n epochs.")
parser.add_argument("--dataset", type=Path, required=False, help="Path for the dataset.")
parser.add_argument("--images_per_identity", type=int, default=1, help="Path for the dataset.")
parser.add_argument("--chunk_size", type=int, default=10000, help="Number of generated images per run")


opt = parser.parse_args()
if opt.mode == "train" and not opt.dataset:
    parser.error("--dataset argument is required for mode train.")
print(opt)

prefix_dir = Path(opt.target_path)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

if opt.output:
        current_session_dir = Path(opt.output)
        current_session_dir.mkdir(parents=True, exist_ok=True)
else:
    images_target_dir = prefix_dir / Path("images") / f"{opt.mode}"
    images_target_dir.mkdir(parents=True, exist_ok=True)
    current_session_dir = create_next_subfolder(images_target_dir)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
#os.makedirs("../../data/mnist", exist_ok=True)

def add_noise(images, mean=0.0, std=0.1):
    noise = torch.randn_like(images) * std + mean
    return images + noise


# ----------
#  Training
# ----------
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


if opt.mode == 'train':
    dataloader = torch.utils.data.DataLoader(
        # datasets.MNIST(
        #    "../../data/mnist",
        #    train=True,
        #    download=True,
        #    transform=transforms.Compose(
        #        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        #    ),
        # ),
        datasets.ImageFolder(
            # r"C:\Users\Aitor\datasets\race_unbalance\minisample_112x112",
            # r"C:\Users\Aitor\datasets\race_unbalance\African_64x64_single_dir",
            opt.dataset,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            )
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    best_g_loss = float('inf')
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    if not opt.output:
        checkpoint_path = create_next_subfolder(prefix_dir / 'checkpoints')
    else:
        checkpoint_path = Path(opt.output) / 'checkpoints'
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path/ "hyperparamenters.txt", 'w') as f:
        f.write(str(vars(opt)))
    # Inside your training loop
    losses_file_path = checkpoint_path / "training_losses.txt"
    with open(losses_file_path, "w") as file:
        file.write("Epoch,Batch,D_Loss,G_Loss\n")  # Header for the CSV file

    G_losses = []
    D_losses = []
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)
            # Inside the training loop
            #real_imgs = add_noise(real_imgs)
            #gen_imgs = add_noise(gen_imgs.detach())  # gen_imgs created by the generator

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            with open(losses_file_path, "a") as file:
                file.write(f"{epoch},{i},{d_loss.item()},{g_loss.item()}\n")

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25],current_session_dir/f"{batches_done}.png", nrow=5, normalize=True)
            if g_loss.item() < best_g_loss:
                best_g_loss = g_loss.item()
                best_generator_path = checkpoint_path / 'generator_best.pth'
                torch.save(generator.state_dict(), str(best_generator_path))
                best_discriminator_path = checkpoint_path / 'discriminator_best.pth'
                torch.save(discriminator.state_dict(), str(best_discriminator_path))
        if epoch % opt.save_every == 0:
            print(f"Saving checkpoint for epoch {epoch}")
            generator_checkpoint_path = checkpoint_path / f'generator_epoch_{epoch}.pth'
            discriminator_checkpoint_path = checkpoint_path / f'discriminator_epoch_{epoch}.pth'
            torch.save(generator.state_dict(), str(generator_checkpoint_path))
            torch.save(discriminator.state_dict(), str(discriminator_checkpoint_path))

    generator_checkpoint_path = checkpoint_path / f'generator_final.pth'
    discriminator_checkpoint_path = checkpoint_path / f'discriminator_final.pth'
    torch.save(generator.state_dict(), str(generator_checkpoint_path))
    torch.save(discriminator.state_dict(), str(discriminator_checkpoint_path))

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(str(checkpoint_path / 'training_losses_final.png'))
    plt.close()

elif opt.mode == 'generate':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.load_state_dict(torch.load(opt.generator_path, map_location=device))
    generator.eval()
    images_left_to_generate = opt.number_of_generated_images
    dirs = []
    while images_left_to_generate > 0:
        number_of_images_to_generate = min(opt.chunk_size, images_left_to_generate)
        number_of_identities = number_of_images_to_generate // opt.images_per_identity + 1
        print(f"Images left to generate: {images_left_to_generate}")
        for i in range(number_of_identities):
            base_z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
            variations = Variable(Tensor(np.random.normal(0, 0.8,
                                                          (opt.images_per_identity, opt.latent_dim))))
            z = base_z + variations
            gen_imgs: object = generator(z)
            identity = "".join(random.choices(ascii_letters + digits, k=8))
            while identity in dirs:
                identity = "".join(random.choices(ascii_letters + digits, k=8))

            image_path = current_session_dir / identity
            image_path.mkdir(parents=True)
            dirs.append(identity)
            for i, img in enumerate(gen_imgs):
                gen_img_path = image_path / f"generated_img_{i}.png"
                save_image(img, gen_img_path, normalize=True)
        images_left_to_generate -= number_of_images_to_generate

    print(current_session_dir)
